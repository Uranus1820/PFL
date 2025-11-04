import copy
import math
import time
from collections import OrderedDict

import numpy as np
import openpyxl as op
import torch
from sklearn.mixture import GaussianMixture

from flcore.clients.clientPFL import clientPFL
from utils.data_utils import read_client_data


class PFL(object):
    """基于每客户端的两高斯混合模型（2-GMM）实现个性化的服务器。
    - 从各客户端收集分类器（psi）参数。
    - 对每个客户端 k，计算其与其他客户端的 psi 向量距离，并在这一维距离向量上拟合二元高斯混合模型（GMM），
      将其他客户端划分为 候选/非候选 两组。
    - 对候选集合执行 FedAvg 得到个性化分类器（psi）。
    - 对特征提取器（phi）使用所有已上传的更新执行 FedAvg 得到全局 phi。
    """

    def __init__(self, args, times):
        # 基础配置
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        # 客户端列表与本轮选择结果
        self.clients = []
        self.selected_clients = []

        # 评估指标缓存
        self.rs_test_acc = []
        self.rs_train_loss = []

        # 训练流程控制
        self.times = times
        self.eval_gap = args.eval_gap

        # 个性化相关参数
        self.layer_idx = args.layer_idx
        self.resource_only_interval = int(getattr(args, "resource_only_interval", 0))

        # 模型模板与参数名称划分（用于区分 phi/psi）
        self.model_template = copy.deepcopy(args.model)
        self._param_names = [n for n, _ in self.model_template.named_parameters()]
        self._psi_names = self._param_names[-self.layer_idx:] if self.layer_idx > 0 else []
        self._phi_names = self._param_names[:-self.layer_idx] if self.layer_idx > 0 else self._param_names

        # 客户端状态缓存
        self.client_psi_vectors = {i: None for i in range(self.num_clients)}
        self.client_psi_states = {i: None for i in range(self.num_clients)}
        self.client_alphas = {i: 0.0 for i in range(self.num_clients)}
        self.client_samples = {i: 0 for i in range(self.num_clients)}

        # 聚合中间缓存
        self.uploaded_updates = []
        self.current_round = 0
        self.global_phi_state = None
        self.personalized_models_cache = {i: copy.deepcopy(self.model_template).state_dict() for i in range(self.num_clients)}

        # 创建客户端实例
        self.set_clients(args, clientPFL)

        # 结果记录工作簿
        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Two-component GMM personalization enabled")
        print("Finished creating server and clients.")

        # 时间开销记录
        self.Budget = []

    # ------------------------------
    # 训练主循环
    # ------------------------------
    def train(self):
        for rnd in range(self.global_rounds + 1):
            self.current_round = rnd
            s_t = time.time()

            # 本轮选择参与的客户端
            self.selected_clients = self.select_clients()

            # 下发模型（初始为模板；之后为个性化缓存）
            self.send_personalized_models()

            # 周期性评估
            if rnd % self.eval_gap == 0:
                print(f"\n-------------Round number: {rnd}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(nonprint=None)

            # 客户端本地训练
            for client in self.selected_clients:
                client.train()

            # 接收上传并计算下一轮个性化参数
            self.receive_pf_updates()
            self.compute_personalized_models()

            # 时间记录
            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    # ------------------------------
    # 客户端管理
    # ------------------------------
    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, id=i, train_samples=len(train_data), test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_personalized_models(self):
        assert self.clients
        for client in self.clients:
            state = self.personalized_models_cache.get(client.id)
            if state is None:
                state = self.model_template.state_dict()
            client.set_model(state)

    # ------------------------------
    # 更新数据的收发（I/O）
    # ------------------------------
    def receive_pf_updates(self):
        assert self.selected_clients
        self.uploaded_updates = []

        s_t = time.time()
        for client in self.selected_clients:
            payload = client.get_pf_updates()
            psi_vector = payload['psi_vector']

            # 记录客户端的最新状态
            self.client_psi_vectors[client.id] = psi_vector
            self.client_psi_states[client.id] = payload['psi']
            self.client_alphas[client.id] = payload['alpha']
            self.client_samples[client.id] = payload['samples']

            update_record = {
                'client_id': client.id,
                'phi': payload['phi'],
                'psi': payload['psi'],
                'psi_vector': psi_vector,
                'alpha': payload['alpha'],
                'samples': payload['samples'],
            }
            self.uploaded_updates.append(update_record)

        print("Updates reception completed:")
        print('-' * 50, time.time() - s_t)

    # ------------------------------
    # 个性化聚合逻辑
    # ------------------------------
    def compute_personalized_models(self):
        """基于两高斯混合模型的距离划分，计算全局 phi 与每客户端个性化 psi。"""
        s_t = time.time()

        template_state = self.model_template.state_dict()

        # 全局 phi 聚合（对已上传更新执行 FedAvg）
        if self.uploaded_updates:
            restored_list = []
            weights = []
            for upd in self.uploaded_updates:
                restored = self._restore_full_state(upd, template_state)
                restored_list.append(restored)
                weights.append(float(max(1, upd.get('samples', 0))))
            total_w = sum(weights) if weights else 1.0
            global_phi = {}
            for name in self._phi_names:
                acc = torch.zeros_like(template_state[name])
                for w, st in zip(weights, restored_list):
                    acc += st[name].to(self.device) * (w / total_w)
                global_phi[name] = acc
            self.global_phi_state = global_phi
        elif self.global_phi_state is None:
            self.global_phi_state = {name: template_state[name].to(self.device) for name in self._phi_names}

        # 具备 psi 向量与状态的可用客户端列表
        available_ids = [cid for cid in range(self.num_clients)
                         if self.client_psi_states.get(cid) is not None and self.client_psi_vectors.get(cid) is not None]

        new_cache = {}
        all_ids = list(range(self.num_clients))
        for k in all_ids:
            base_vec = self.client_psi_vectors.get(k)
            distances = []
            other_ids = []
            if base_vec is not None:
                for j in available_ids:
                    if j == k:
                        continue
                    vj = self.client_psi_vectors.get(j)
                    if vj is None:
                        continue
                    diff = base_vec - vj
                    d = float(np.sqrt(np.dot(diff, diff)))
                    distances.append(d)
                    other_ids.append(j)

            # 基于距离的一维数据，用 2-GMM 划分候选集合
            candidate_ids = []
            if len(distances) >= 2:
                X = np.array(distances, dtype=np.float64).reshape(-1, 1)
                try:
                    gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
                    gmm.fit(X)
                    labels = gmm.predict(X)
                    means = gmm.means_.reshape(-1)
                    low_idx = int(np.argmin(means))
                    candidate_ids = [oid for oid, lab in zip(other_ids, labels) if int(lab) == low_idx]
                except Exception:
                    med = float(np.median(distances))
                    candidate_ids = [oid for oid, d in zip(other_ids, distances) if d <= med]
            else:
                candidate_ids = [j for j in available_ids if j != k]

            # 个性化 psi 聚合（对候选集合执行 FedAvg）
            psi_sources = []
            psi_weights = []
            for cid in candidate_ids:
                st = self.client_psi_states.get(cid)
                if st is None:
                    continue
                psi_sources.append(st)
                psi_weights.append(float(max(1, self.client_samples.get(cid, 0))))
            if not psi_sources:
                self_state = self.client_psi_states.get(k)
                if self_state is not None:
                    psi_sources = [self_state]
                    psi_weights = [float(max(1, self.client_samples.get(k, 0)))]
                else:
                    tmpl = {name: template_state[name].to(self.device) for name in self._psi_names}
                    psi_sources = [tmpl]
                    psi_weights = [1.0]

            total_w = float(sum(psi_weights)) if psi_weights else 1.0
            personalized_psi = {}
            for name in self._psi_names:
                acc = torch.zeros_like(template_state[name])
                for w, st in zip(psi_weights, psi_sources):
                    acc += st[name].to(self.device) * (w / total_w)
                personalized_psi[name] = acc

            # 组合全局 phi 与个性化 psi，得到每个客户端的个性化模型参数
            combined = OrderedDict()
            for name in template_state:
                if name in self._phi_names:
                    combined[name] = self.global_phi_state[name]
                elif name in self._psi_names:
                    combined[name] = personalized_psi[name]
                else:
                    combined[name] = template_state[name].to(self.device)
            new_cache[k] = combined

        self.personalized_models_cache = new_cache
        print("Personalized aggregation complete:")
        print('-' * 50, time.time() - s_t)

    # ------------------------------
    # 稀疏还原的实用函数
    # ------------------------------
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

    # ------------------------------
    # 评估辅助函数
    # ------------------------------
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        for client in self.clients:
            model_state = self.personalized_models_cache.get(client.id)
            temp_model = copy.deepcopy(self.model_template)
            if model_state is not None:
                temp_model.load_state_dict(model_state)
            ct, ns, auc = client.test_metrics(model=temp_model)
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
            model_state = self.personalized_models_cache.get(client.id)
            temp_model = copy.deepcopy(self.model_template)
            if model_state is not None:
                temp_model.load_state_dict(model_state)
            cl, ns = client.train_metrics(model=temp_model)
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

