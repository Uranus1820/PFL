import copy
import time
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from flcore.clients.clientPFL import clientPFL
from utils.data_utils import read_client_data


class PFL(object):
    """服务器：执行GMM聚类、聚合φ/ψ并协调奖励。"""

    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.times = times

        self.model_template = copy.deepcopy(args.model)
        self.layer_idx = args.layer_idx
        self._param_names = [n for n, _ in self.model_template.named_parameters()]
        self._phi_names = self._param_names[:-self.layer_idx]
        self._psi_names = self._param_names[-self.layer_idx:]
        template_state = self.model_template.state_dict()
        self.global_phi = {name: template_state[name].to(self.device) for name in self._phi_names}

        # 客户端记录
        self.clients = []
        self.selected_clients = []
        self.client_psi_vectors = {i: None for i in range(self.num_clients)}
        self.client_psi_states = {i: None for i in range(self.num_clients)}
        self.client_alphas = {i: 0.0 for i in range(self.num_clients)}
        self.client_batches = {i: 1 for i in range(self.num_clients)}
        self.client_similarity = {i: 0.0 for i in range(self.num_clients)}

        # 个性化模型与消息
        self.personalized_cache = {
            i: copy.deepcopy(self.model_template).state_dict() for i in range(self.num_clients)
        }
        self.pending_packets = {i: {"reward": 0.0, "alpha_bar": 0.0, "j_bar": 1, "similarity": 0.0,
                                   "cluster_size": 0} for i in range(self.num_clients)}

        # GMM与奖励参数
        self.init_components = max(1, int(args.gmm_init_components))
        self.merge_epsilon = float(args.epsilon_merge)
        self.sigma = float(args.gmm_sigma)
        self.reward_p0 = float(args.reward_p0)
        self.reward_pJ = float(args.reward_pJ)
        self.reward_palpha = float(args.reward_palpha)
        self.reward_rho = float(args.reward_rho_q)

        # 训练指标
        self.rs_test_acc = []
        self.rs_train_loss = []

        self.set_clients(args, clientPFL)

    # ---------------------- 主循环 ----------------------
    def train(self):
        for rnd in range(self.global_rounds):
            start = time.time()
            self.selected_clients = self.select_clients()
            self.send_personalized_models()
            for client in self.selected_clients:
                client.train()
            self.receive_updates()
            self.aggregate_and_reward()
            print(f"第 {rnd} 轮耗时 {time.time() - start:.2f}s")

    # ---------------------- 客户端管理 ----------------------
    def set_clients(self, args, client_obj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = client_obj(args, cid=i, train_samples=len(train_data), test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        join_num = min(self.join_clients, self.num_clients)
        indices = np.random.choice(range(self.num_clients), join_num, replace=False)
        return [self.clients[i] for i in indices]

    def send_personalized_models(self):
        for client in self.clients:
            state = self.personalized_cache.get(client.id, self.model_template.state_dict())
            client.set_model(state)
            packet = self.pending_packets.get(client.id, {"reward": 0.0})
            client.update_game_state(packet)

    # ---------------------- 接收更新 ----------------------
    def receive_updates(self):
        self.uploaded_updates = []
        for client in self.selected_clients:
            payload = client.get_pf_updates()
            self.client_psi_vectors[client.id] = payload["psi_vector"]
            self.client_psi_states[client.id] = payload["psi"]
            self.client_alphas[client.id] = payload["alpha"]
            self.client_batches[client.id] = payload["batch_size"]
            self.uploaded_updates.append({
                "client_id": client.id,
                "phi": payload["phi"],
                "psi": payload["psi"],
                "psi_vector": payload["psi_vector"],
                "alpha": payload["alpha"],
                "batch": payload["batch_size"],
                "samples": payload["samples"],
            })

    # ---------------------- 聚合与奖励 ----------------------
    def aggregate_and_reward(self):
        template = self.model_template.state_dict()
        restored = self._restore_states(template)
        self._aggregate_global_phi(restored, template)
        cluster_info = self._cluster_clients()
        phi_clusters = self._aggregate_phi_by_cluster(cluster_info, restored, template)
        psi_clusters, similarity = self._aggregate_psi_by_cluster(cluster_info, template)
        self._update_personalized_cache(cluster_info, phi_clusters, psi_clusters, template)
        self._dispatch_packets(cluster_info, similarity)

    def _restore_states(self, template_state):
        restored = {}
        for update in self.uploaded_updates:
            cid = update["client_id"]
            restored[cid] = self._restore_full_state(update, template_state)
        return restored

    def _aggregate_global_phi(self, restored_states, template_state):
        if not restored_states:
            return
        weights = []
        tensors = []
        for cid, state in restored_states.items():
            tensors.append(state)
            weights.append(float(max(1, self.client_batches.get(cid, 1))))
        total = sum(weights)
        phi = {}
        for name in self._phi_names:
            acc = torch.zeros_like(template_state[name])
            for w, st in zip(weights, tensors):
                acc += st[name].to(self.device) * (w / total)
            phi[name] = acc
        self.global_phi = phi

    def _cluster_clients(self):
        client_ids = []
        vectors = []
        for cid, vec in self.client_psi_vectors.items():
            if vec is None:
                continue
            client_ids.append(cid)
            vectors.append(vec)
        if not vectors:
            return {"clusters": {}, "assignments": {}, "means": {}}

        X = np.stack(vectors, axis=0)
        n_components = min(len(client_ids), self.init_components)
        try:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                  reg_covar=1e-6, max_iter=200, random_state=0)
            gmm.fit(X)
            responsibilities = gmm.predict_proba(X)
            means = gmm.means_
        except Exception:
            responsibilities = np.ones((len(client_ids), 1))
            means = np.array([np.mean(X, axis=0)])

        merged = self._merge_components(means)
        if not merged:
            merged = [{"ids": list(range(len(means))), "mean": np.mean(means, axis=0)}]

        comp_map = {}
        for idx, comp in enumerate(merged):
            for old_id in comp["ids"]:
                comp_map[old_id] = idx

        clusters = defaultdict(list)
        assignments = {}
        for idx, cid in enumerate(client_ids):
            comp_scores = np.zeros(len(merged))
            for old_id, cluster_id in comp_map.items():
                if old_id < responsibilities.shape[1]:
                    comp_scores[cluster_id] += responsibilities[idx, old_id]
            cluster_id = int(np.argmax(comp_scores))
            clusters[cluster_id].append(cid)
            assignments[cid] = cluster_id

        means_dict = {i: comp["mean"] for i, comp in enumerate(merged)}
        return {"clusters": dict(clusters), "assignments": assignments, "means": means_dict}

    def _merge_components(self, means):
        comps = [{"ids": [i], "mean": means[i]} for i in range(len(means))]
        changed = True
        while changed and len(comps) > 1:
            changed = False
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    dist = np.linalg.norm(comps[i]["mean"] - comps[j]["mean"]) ** 2
                    if dist <= self.merge_epsilon:
                        merged = {
                            "ids": comps[i]["ids"] + comps[j]["ids"],
                            "mean": (comps[i]["mean"] + comps[j]["mean"]) / 2.0,
                        }
                        comps[i] = merged
                        comps.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return comps

    def _aggregate_phi_by_cluster(self, cluster_info, restored_states, template_state):
        phi_clusters = {}
        for cluster_id, members in cluster_info["clusters"].items():
            acc = {name: torch.zeros_like(template_state[name]) for name in self._phi_names}
            total = 0.0
            for cid in members:
                if cid not in restored_states:
                    continue
                total += 1.0
                for name in self._phi_names:
                    acc[name] += restored_states[cid][name].to(self.device)
            if total > 0:
                for name in acc:
                    acc[name] /= total
                phi_clusters[cluster_id] = acc
            else:
                phi_clusters[cluster_id] = {name: tensor.clone() for name, tensor in self.global_phi.items()}
        return phi_clusters

    def _aggregate_psi_by_cluster(self, cluster_info, template_state):
        psi_clusters = {}
        similarity = {}
        for cluster_id, members in cluster_info["clusters"].items():
            mu = cluster_info["means"].get(cluster_id)
            if mu is None:
                continue
            sigma2 = max(self.sigma ** 2, 1e-6)
            acc = {name: torch.zeros_like(template_state[name]) for name in self._psi_names}
            weight_sum = 0.0
            for cid in members:
                vec = self.client_psi_vectors.get(cid)
                state = self.client_psi_states.get(cid)
                if vec is None or state is None:
                    continue
                dist2 = float(np.sum((vec - mu) ** 2))
                w = np.exp(-dist2 / sigma2)
                similarity[cid] = w
                weight_sum += w
                for name in self._psi_names:
                    acc[name] += state[name].to(self.device) * w
            if weight_sum > 0:
                for name in acc:
                    acc[name] /= weight_sum
                psi_clusters[cluster_id] = acc
            else:
                psi_clusters[cluster_id] = {name: template_state[name].to(self.device) for name in self._psi_names}
        self.client_similarity.update(similarity)
        return psi_clusters, similarity

    def _update_personalized_cache(self, cluster_info, phi_clusters, psi_clusters, template_state):
        new_cache = {}
        for cid in range(self.num_clients):
            cluster_id = cluster_info["assignments"].get(cid)
            combined = OrderedDict()
            for name in template_state:
                if cluster_id is not None and name in self._phi_names:
                    combined[name] = phi_clusters[cluster_id][name]
                elif cluster_id is not None and name in self._psi_names:
                    combined[name] = psi_clusters[cluster_id][name]
                else:
                    combined[name] = template_state[name].to(self.device)
            new_cache[cid] = combined
        self.personalized_cache = new_cache

    def _dispatch_packets(self, cluster_info, similarity):
        packets = {}
        for cluster_id, members in cluster_info["clusters"].items():
            if not members:
                continue
            alpha_bar = float(np.mean([self.client_alphas.get(cid, 0.0) for cid in members]))
            j_bar = float(np.mean([self.client_batches.get(cid, 1) for cid in members]))
            for cid in members:
                reward = (self.reward_p0 +
                          self.reward_pJ * (self.client_batches.get(cid, 1) - j_bar) +
                          self.reward_palpha * (self.client_alphas.get(cid, 0.0) - alpha_bar) -
                          self.reward_rho * (1.0 - similarity.get(cid, 0.0)))
                packets[cid] = {
                    "reward": reward,
                    "alpha_bar": alpha_bar,
                    "j_bar": j_bar,
                    "similarity": similarity.get(cid, 0.0),
                    "cluster_size": len(members),
                }
        for cid in range(self.num_clients):
            if cid not in packets:
                packets[cid] = {"reward": 0.0, "alpha_bar": 0.0, "j_bar": 1,
                                "similarity": 0.0, "cluster_size": 0}
        self.pending_packets = packets

    # ---------------------- 工具函数 ----------------------
    def _restore_full_state(self, update, template_state):
        restored = OrderedDict()
        for name, tensor in template_state.items():
            if name in update["psi"]:
                restored[name] = update["psi"][name].to(tensor.device, dtype=tensor.dtype)
            elif name in update["phi"]:
                restored[name] = self._restore_tensor(update["phi"][name], tensor)
            else:
                restored[name] = torch.zeros_like(tensor)
        return restored

    def _restore_tensor(self, info, template_tensor):
        target = torch.zeros_like(template_tensor)
        indices = info["indices"]
        values = info["values"].to(template_tensor.device, dtype=template_tensor.dtype)
        if indices is None:
            if values.shape == target.shape:
                return values.clone()
            target.copy_(values.view_as(target))
            return target
        idx = indices.to(torch.long).to(template_tensor.device)
        slicer = [idx] + [slice(None)] * (target.dim() - 1)
        target[tuple(slicer)] = values
        return target

    # ---------------------- 评估（可选） ----------------------
    def evaluate(self):
        ids = []
        correct = []
        total = []
        for client in self.clients:
            model_state = self.personalized_cache.get(client.id)
            temp_model = copy.deepcopy(self.model_template)
            if model_state is not None:
                temp_model.load_state_dict(model_state)
            ct, ns, _ = client.test_metrics(temp_model)
            ids.append(client.id)
            correct.append(ct)
            total.append(ns)
        overall = sum(correct) / max(1, sum(total))
        self.rs_test_acc.append(overall)
        print(f"平均准确率 {overall:.4f}")
