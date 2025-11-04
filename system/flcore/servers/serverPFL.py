import copy
import math
import time
from collections import OrderedDict, defaultdict

import numpy as np
import openpyxl as op
import torch
from sklearn.mixture import GaussianMixture

from flcore.clients.clientPFL import clientPFL
from utils.data_utils import read_client_data


class PFL(object):
    """服务器端执行 GMM 聚类与资源感知的个性化聚合。"""

    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.num_clusters = getattr(args, "num_clusters", 2)
        self.layer_idx = args.layer_idx
        self.gmm_sigma = float(getattr(args, "gmm_sigma", 1.0))
        self.resource_only_interval = int(getattr(args, "resource_only_interval", 0))

        self.gmm = GaussianMixture(n_components=self.num_clusters, covariance_type='diag')
        self.gmm_fitted = False

        self.cluster_models = [copy.deepcopy(args.model) for _ in range(self.num_clusters)]

        self.client_psi_vectors = {i: None for i in range(self.num_clients)}
        self.client_cluster_ids = {i: np.random.randint(0, self.num_clusters) for i in range(self.num_clients)}
        self.client_alphas = {i: 0.0 for i in range(self.num_clients)}
        self.client_samples = {i: 0 for i in range(self.num_clients)}

        self.uploaded_updates = []
        self.current_round = 0

        self.set_clients(args, clientPFL)

        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"GMM Clusters: {self.num_clusters}")
        print("Finished creating server and clients.")

        self.Budget = []

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------
    def train(self):
        for rnd in range(self.global_rounds + 1):
            self.current_round = rnd
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if rnd > 0:
                self.perform_gmm_clustering()

            self.send_cluster_models()

            if rnd % self.eval_gap == 0:
                print(f"\n-------------Round number: {rnd}-------------")
                print("\nEvaluate cluster models")
                self.evaluate(nonprint=None)

            for client in self.selected_clients:
                client.train()

            self.receive_pf_updates()
            self.aggregate_cluster_models()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    # ------------------------------------------------------------------
    # 客户端与模型发送/接收
    # ------------------------------------------------------------------
    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_cluster_models(self):
        assert self.clients
        for client in self.clients:
            cluster_id = self.client_cluster_ids[client.id]
            model_k = self.cluster_models[cluster_id]
            client.set_model(model_k.state_dict())

    def receive_pf_updates(self):
        assert self.selected_clients
        self.uploaded_updates = []

        s_t = time.time()
        for client in self.selected_clients:
            payload = client.get_pf_updates()
            psi_vector = payload['psi_vector']

            self.client_psi_vectors[client.id] = psi_vector
            self.client_alphas[client.id] = payload['alpha']
            self.client_samples[client.id] = payload['samples']

            update_record = {
                'client_id': client.id,
                'cluster': self.client_cluster_ids[client.id],
                'phi': payload['phi'],
                'psi': payload['psi'],
                'psi_vector': psi_vector,
                'alpha': payload['alpha'],
                'samples': payload['samples'],
            }
            self.uploaded_updates.append(update_record)

        print("GMM Vector Reception:")
        print('-' * 50, time.time() - s_t)

    # ------------------------------------------------------------------
    # GMM 聚类
    # ------------------------------------------------------------------
    def perform_gmm_clustering(self):
        s_t = time.time()
        psi_vectors = [vec for vec in self.client_psi_vectors.values() if vec is not None]
        client_ids = [cid for cid, vec in self.client_psi_vectors.items() if vec is not None]

        if len(psi_vectors) < self.num_clusters:
            print("GMM Warning: 客户端数量不足以聚类，跳过本轮聚类。")
            return

        self.gmm.fit(psi_vectors)
        self.gmm_fitted = True

        labels = self.gmm.predict(psi_vectors)
        for cid, label in zip(client_ids, labels):
            self.client_cluster_ids[cid] = int(label)

        print("GMM Clustering complete:")
        print(f"Cluster assignment: {self.client_cluster_ids}")
        print('-' * 50, time.time() - s_t)

    # ------------------------------------------------------------------
    # 集群模型聚合
    # ------------------------------------------------------------------
    def aggregate_cluster_models(self):
        s_t = time.time()

        cluster_to_updates = defaultdict(list)
        for update in self.uploaded_updates:
            cluster_to_updates[update['cluster']].append(update)

        for cluster_id in range(self.num_clusters):
            updates = cluster_to_updates.get(cluster_id, [])
            if not updates:
                continue

            template_state = self.cluster_models[cluster_id].state_dict()
            aggregated_state = OrderedDict((name, torch.zeros_like(param)) for name, param in template_state.items())

            weights = []
            restored_states = []
            for update in updates:
                weight = self._compute_attention_weight(update, cluster_id)
                if weight <= 0:
                    continue
                restored = self._restore_full_state(update, template_state)
                weights.append(weight)
                restored_states.append(restored)

            if not weights:
                continue

            total_weight = sum(weights)
            for w, state in zip(weights, restored_states):
                for name in aggregated_state:
                    aggregated_state[name] += state[name].to(self.device) * (w / total_weight)

            self.cluster_models[cluster_id].load_state_dict(aggregated_state)

        print("Aggregate clusters:")
        print('-' * 50, time.time() - s_t)

    def _restore_full_state(self, update, template_state):
        restored = OrderedDict()
        phi_updates = update['phi']
        psi_updates = update['psi']

        for name, template_tensor in template_state.items():
            if name in psi_updates:
                tensor = psi_updates[name].to(template_tensor.device, dtype=template_tensor.dtype)
            elif name in phi_updates:
                tensor = self._restore_tensor_from_sparse(phi_updates[name], template_tensor)
            else:
                tensor = torch.zeros_like(template_tensor)
            restored[name] = tensor
        return restored

    def _restore_tensor_from_sparse(self, info, template_tensor):
        target = torch.zeros_like(template_tensor)
        indices = info['indices']
        values = info['values']

        if indices is None:
            data = values.to(template_tensor.device, dtype=template_tensor.dtype)
            if data.shape == target.shape:
                target = data.clone()
            else:
                target.copy_(data.view_as(target))
            return target

        if isinstance(indices, torch.Tensor):
            indices_tensor = indices.to(torch.long)
        else:
            indices_tensor = torch.tensor(indices, dtype=torch.long)

        if indices_tensor.numel() == 0:
            return target

        data = values.to(template_tensor.device, dtype=template_tensor.dtype)
        slicer = [indices_tensor.to(template_tensor.device)] + [slice(None)] * (target.dim() - 1)
        target[tuple(slicer)] = data
        return target

    def _compute_attention_weight(self, update, cluster_id):
        alpha = max(0.0, float(update['alpha']))
        if alpha <= 0:
            return 0.0

        if self.resource_only_interval > 0 and self.current_round % self.resource_only_interval == 0:
            return alpha

        if not self.gmm_fitted or not hasattr(self.gmm, 'means_'):
            return alpha

        sigma_sq = max(1e-12, self.gmm_sigma ** 2)
        mu_k = self.gmm.means_[cluster_id]
        diff = update['psi_vector'] - mu_k
        dist_sq = float(np.dot(diff, diff))
        w_gmm = math.exp(-dist_sq / sigma_sq)
        return alpha * w_gmm

    # ------------------------------------------------------------------
    # 评估与监控
    # ------------------------------------------------------------------
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        for client in self.clients:
            cluster_id = self.client_cluster_ids[client.id]
            model_k = self.cluster_models[cluster_id]
            ct, ns, auc = client.test_metrics(model=model_k)
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            accs.append(ct * 1.0 / ns if ns > 0 else 0.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, accs

    def train_metrics(self):
        num_samples = []
        losses = []
        for client in self.clients:
            cluster_id = self.client_cluster_ids[client.id]
            model_k = self.cluster_models[cluster_id]
            cl, ns = client.train_metrics(model=model_k)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None, nonprint=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / max(sum(stats[1]), 1)
        test_auc = sum(stats[3]) * 1.0 / max(sum(stats[1]), 1)
        train_loss = sum(stats_train[2]) * 1.0 / max(sum(stats_train[1]), 1)
        accs = [a / n if n > 0 else 0.0 for a, n in zip(stats[2], stats[1])]
        aucs = [a / n if n > 0 else 0.0 for a, n in zip(stats[3], stats[1])]

        if nonprint is None:
            if acc is None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)
            if loss is None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)
            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return accs, test_acc

    def set_parameters(self, model, parameters):
        for new_param, old_param in zip(parameters, model.parameters()):
            old_param.data = torch.tensor(new_param, dtype=torch.float).to(self.device)

    def get_parameters(self, model):
        return [val.data.cpu().numpy() for val in model.parameters()]
