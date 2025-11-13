import copy
import math
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data


class clientPFL(object):
    """客户端：执行拆分模型训练并在马尔可夫势博弈下选择 (α, J)。"""

    def __init__(self, args, cid, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.device = args.device
        self.dataset = args.dataset
        self.id = cid

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.layer_idx = args.layer_idx
        self.loss = nn.CrossEntropyLoss()

        # 决策空间：α ∈ (0,1]，J ∈ [1, J_max]
        self.alpha_min = max(1e-3, float(args.alpha_min))
        self.alpha_max = min(1.0, float(args.alpha_max))
        self.alpha_grid = np.linspace(self.alpha_min, self.alpha_max,
                                      max(2, int(args.alpha_samples))).tolist()
        self.j_min = 1
        self.j_max = max(1, int(args.max_batch_size))
        self.j_step = max(1, int(args.batch_step))
        self.j_grid = list(range(self.j_min, self.j_max + 1, self.j_step))
        if self.j_grid[-1] != self.j_max:
            self.j_grid.append(self.j_max)
        self.action_space = [(round(float(a), 4), int(j)) for a in self.alpha_grid for j in self.j_grid]

        # 成本模型参数
        self.proc_cost = float(args.proc_cost)
        self.comm_cost = float(args.comm_cost)

        # 马尔可夫博弈相关参数
        self.discount = float(args.discount_factor)
        self.td_lr = float(args.td_learning_rate)
        self.policy_eps = float(args.policy_epsilon)
        self.q_table = {}
        self.current_state = ("init",)
        self.pending_state = None
        self.pending_action = None
        self.pending_cost = 0.0

        # 记录当前策略
        self.alpha = self.alpha_max
        self.batch_size = self.j_max

        # 稀疏上传辅助缓存
        self._selected_channels = {}
        self._phi_masks = {}

    # ---------------------- 服务器交互 ----------------------
    def set_model(self, state_dict):
        """接收服务器下发的个性化模型。"""
        self.model.load_state_dict(state_dict)

    def update_game_state(self, packet):
        """接收服务器的奖励与集群状态，并执行TD更新。"""
        next_state = self._encode_state(packet)
        reward = float(packet.get("reward", 0.0)) if packet else 0.0

        if self.pending_action is not None and self.pending_state is not None:
            prev_q = self._ensure_q(self.pending_state, self.pending_action)
            next_q = self._max_q(next_state)
            utility = reward - self.pending_cost
            updated = prev_q + self.td_lr * (utility + self.discount * next_q - prev_q)
            self.q_table[self.pending_state][self.pending_action] = updated
            self.pending_action = None
            self.pending_state = None
            self.pending_cost = 0.0

        self.current_state = next_state

    def train(self):
        """依据当前状态选择 (α, J) 并执行本地训练。"""
        action = self._select_action(self.current_state)
        self.alpha = action[0]
        self.batch_size = action[1]
        self.pending_state = self.current_state
        self.pending_action = action
        self.pending_cost = self._calc_cost(action)

        trainloader = self.load_train_data(self.batch_size)
        if len(trainloader) == 0:
            return

        self.model.train()
        named_params = list(self.model.named_parameters())
        phi_named = named_params[:-self.layer_idx]
        psi_named = named_params[-self.layer_idx:]

        self._selected_channels = self._select_top_channels(phi_named, self.alpha)
        self._phi_masks = self._build_phi_masks(phi_named, self._selected_channels)

        phi_params = [param for name, param in phi_named if name in self._phi_masks]
        psi_params = [param for _, param in psi_named]

        optim_groups = []
        if phi_params:
            optim_groups.append({'params': phi_params, 'lr': self.learning_rate * 0.1})
        if psi_params:
            optim_groups.append({'params': psi_params, 'lr': self.learning_rate})
        if not optim_groups:
            return

        optimizer = torch.optim.SGD(optim_groups)

        for _ in range(self.local_steps):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                self._apply_phi_gradient_masks()
                optimizer.step()

    def get_pf_updates(self):
        """上传部分 φ 与完整 ψ 以及本轮决策。"""
        named_params = list(self.model.named_parameters())
        phi_named = named_params[:-self.layer_idx]
        psi_named = named_params[-self.layer_idx:]

        psi_vector = torch.cat([param.detach().view(-1) for _, param in psi_named]).cpu().numpy()
        psi_state = OrderedDict((name, param.detach().cpu().clone()) for name, param in psi_named)
        phi_updates = self._extract_sparse_phi_updates(phi_named)

        return {
            "phi": phi_updates,
            "psi": psi_state,
            "psi_vector": psi_vector,
            "alpha": float(self.alpha),
            "batch_size": int(self.batch_size),
            "samples": int(self.train_samples),
        }

    # ---------------------- 数据加载 ----------------------
    def load_train_data(self, batch_size):
        data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size):
        data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)

    # ---------------------- 指标计算 ----------------------
    def test_metrics(self, model=None):
        """返回 (正确样本数, 总样本数, AUC)。"""
        loader = self.load_test_data(self.batch_size)
        model = model or self.model
        model.eval()
        correct, total = 0, 0
        y_prob, y_true = [], []

        with torch.no_grad():
            for x, y in loader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                correct += (torch.argmax(out, dim=1) == y).sum().item()
                total += y.shape[0]
                y_prob.append(F.softmax(out, dim=1).cpu().numpy())
                classes = self.num_classes if self.num_classes > 2 else 3
                lb = label_binarize(y.cpu().numpy(), classes=np.arange(classes))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        if total == 0:
            return 0, 0, 0.0
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return correct, total, auc

    def train_metrics(self, model=None):
        """返回 (损失总和, 样本数)。"""
        loader = self.load_train_data(self.batch_size)
        model = model or self.model
        model.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                loss = self.loss(out, y)
                total_loss += loss.item() * y.shape[0]
                total += y.shape[0]
        return total_loss, total

    # ---------------------- 决策/博弈辅助 ----------------------
    def _encode_state(self, packet):
        if not packet:
            return ("init",)
        alpha_bar = round(float(packet.get("alpha_bar", 0.0)), 4)
        j_bar = int(packet.get("j_bar", 1))
        w = round(float(packet.get("similarity", 0.0)), 4)
        cluster_size = int(packet.get("cluster_size", 0))
        return ("state", alpha_bar, j_bar, w, cluster_size)

    def _ensure_q(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}
        elif action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]

    def _max_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}
        return max(self.q_table[state].values()) if self.q_table[state] else 0.0

    def _select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}
        if random.random() < self.policy_eps:
            return random.choice(self.action_space)
        return max(self.q_table[state].items(), key=lambda kv: kv[1])[0]

    def _calc_cost(self, action):
        alpha, batch = action
        proc = self.proc_cost * alpha * batch * max(1, self.local_steps)
        comm = self.comm_cost * alpha
        return proc + comm

    # ---------------------- 稀疏上传工具 ----------------------
    def _select_top_channels(self, phi_named_params, ratio):
        selected = {}
        ratio = max(0.0, min(1.0, float(ratio)))
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            tensor = param.detach()
            if tensor.dim() == 0:
                indices = torch.zeros(0, dtype=torch.long)
            else:
                out_dim = tensor.shape[0]
                k = max(1, int(math.ceil(out_dim * ratio)))
                norms = torch.norm(tensor.view(out_dim, -1), p=2, dim=1)
                _, topk = torch.topk(norms, min(k, out_dim))
                indices = torch.sort(topk).values
            selected[base] = indices.cpu()
        return selected

    def _build_phi_masks(self, phi_named_params, selected_indices):
        masks = {}
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            indices = selected_indices.get(base)
            if indices is None or indices.numel() == 0:
                continue
            tensor = param.detach()
            mask = torch.zeros_like(tensor)
            slicer = [indices.to(tensor.device)] + [slice(None)] * (tensor.dim() - 1)
            mask[tuple(slicer)] = 1.0
            masks[name] = mask
        return masks

    def _apply_phi_gradient_masks(self):
        if not self._phi_masks:
            return
        named_params = dict(self.model.named_parameters())
        for name, mask in self._phi_masks.items():
            if name not in named_params:
                continue
            param = named_params[name]
            if param.grad is not None:
                param.grad.mul_(mask.to(param.grad.device))

    def _extract_sparse_phi_updates(self, phi_named_params):
        updates = OrderedDict()
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            indices = self._selected_channels.get(base, torch.zeros(0, dtype=torch.long))
            tensor = param.detach().cpu()
            if tensor.dim() == 0 or indices.numel() == tensor.shape[0]:
                values = tensor.clone()
                payload_idx = None
            else:
                slicer = [indices] + [slice(None)] * (tensor.dim() - 1)
                values = tensor[tuple(slicer)].clone()
                payload_idx = indices.clone()
            updates[name] = {
                "indices": payload_idx,
                "values": values,
                "shape": torch.tensor(tensor.shape, dtype=torch.long),
            }
        return updates

    @staticmethod
    def _base_param_name(name):
        return name.rsplit('.', 1)[0] if '.' in name else name
