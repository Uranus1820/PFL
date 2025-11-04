import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from system.utils.data_utils import read_client_data


class clientPFL(object):
    """客户端执行资源自适应的局部训练，并上传解耦的特征提取/分类层参数。"""

    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.args = args

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.layer_idx = args.layer_idx  # 切分特征提取层和分类层

        # --- 资源感知训练与效用建模参数 ---
        self.gamma = float(getattr(args, "gamma", 1.0))
        self.epsilon = float(getattr(args, "epsilon", 1.0))
        self.alpha_min = max(0.0, float(getattr(args, "alpha_min", 0.0)))
        self.alpha_max = min(1.0, float(getattr(args, "alpha_max", 1.0)))
        self.resource_cost = max(1e-6, float(getattr(args, "resource_cost", 1.0)))

        # 引入轻微扰动来模拟客户端偏好差异
        self.gamma *= float(1.0 + np.random.uniform(-0.1, 0.1))
        self.resource_cost *= float(1.0 + np.random.uniform(-0.1, 0.1))

        self.alpha = self.alpha_max  # 当前轮的训练比例
        self.optimizer = None

        # 数据质量 Q_i = q_i * d_i (提前计算)
        self.data_quality = self._estimate_data_quality()

        # 用于缓存每轮选择的通道与掩码
        self._selected_channel_indices = {}
        self._phi_masks = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def set_model(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.model_before = copy.deepcopy(self.model)

    def train(self):
        trainloader = self.load_train_data()
        if len(trainloader) == 0:
            return

        self.model.train()

        # 计算最优训练比例 alpha
        self.alpha = self._determine_optimal_alpha()

        named_params = list(self.model.named_parameters())
        phi_named = named_params[:-self.layer_idx]
        psi_named = named_params[-self.layer_idx:]

        # 选择通道并生成掩码
        self._selected_channel_indices = self._select_top_channels(phi_named, self.alpha)
        self._phi_masks = self._build_phi_masks(phi_named, self._selected_channel_indices)

        phi_params_for_optim = [
            param for name, param in phi_named
            if name in self._phi_masks and torch.count_nonzero(self._phi_masks[name]) > 0
        ]
        psi_params_for_optim = [param for _, param in psi_named]

        params_for_optimizer = []
        if phi_params_for_optim:
            params_for_optimizer.append({'params': phi_params_for_optim, 'lr': self.learning_rate * 0.1})
        if psi_params_for_optim:
            params_for_optimizer.append({'params': psi_params_for_optim, 'lr': self.learning_rate})

        if not params_for_optimizer:
            return

        self.optimizer = torch.optim.SGD(params_for_optimizer)

        for _ in range(self.local_steps):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self._apply_phi_gradient_masks()
                self.optimizer.step()

    def get_pf_updates(self):
        named_params = list(self.model.named_parameters())
        phi_named = named_params[:-self.layer_idx]
        psi_named = named_params[-self.layer_idx:]

        psi_vector = torch.cat([
            param.data.view(-1) for _, param in psi_named
        ]).detach().cpu().numpy()

        psi_state = OrderedDict(
            (name, param.detach().cpu().clone()) for name, param in psi_named
        )

        phi_updates = self._extract_sparse_phi_updates(phi_named)

        return {
            'phi': phi_updates,
            'psi': psi_state,
            'psi_vector': psi_vector,
            'alpha': float(self.alpha),
            'samples': int(self.train_samples),
        }

    # ------------------------------------------------------------------
    # 数据集相关辅助方法
    # ------------------------------------------------------------------
    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model is None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model is None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _estimate_data_quality(self):
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = []
        for _, y in train_data:
            if isinstance(y, torch.Tensor):
                labels.append(int(y.item()))
            else:
                labels.append(int(y))

        d_i = max(1, len(labels))
        label_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        prob = label_counts / max(label_counts.sum(), 1.0)
        non_zero = prob > 0
        entropy = -np.sum(prob[non_zero] * np.log(prob[non_zero] + 1e-12))
        max_entropy = math.log(min(self.num_classes, np.count_nonzero(label_counts)) or 1)
        q_i = (entropy / max_entropy) if max_entropy > 0 else 0.0
        q_i = float(max(q_i, 1e-6))
        return q_i * d_i

    def _utility(self, alpha):
        if alpha <= 0:
            return -self.resource_cost * max(alpha, 0.0)
        contribution = 1.0 - math.exp(-self.epsilon * alpha)
        return self.gamma * self.data_quality * contribution - alpha * self.resource_cost

    def _determine_optimal_alpha(self):
        feasible_min = max(0.0, self.alpha_min)
        feasible_max = min(1.0, max(feasible_min, self.alpha_max))
        if feasible_max <= 0:
            return 0.0

        numerator = self.epsilon * self.gamma * self.data_quality
        vertex = float('-inf')
        if numerator > 0 and self.resource_cost > 0:
            ratio = numerator / self.resource_cost
            if ratio > 0:
                vertex = math.log(ratio) / self.epsilon

        if vertex < feasible_min:
            candidate = feasible_min
            return 0.0 if self._utility(candidate) < 0 else candidate

        if vertex > feasible_max:
            return feasible_max

        return max(0.0, vertex)

    def _select_top_channels(self, phi_named_params, ratio):
        selected = {}
        ratio = max(0.0, min(1.0, float(ratio)))
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            if name.endswith('bias') and base in selected:
                continue

            tensor = param.detach()
            out_dim = tensor.shape[0] if tensor.dim() > 0 else 1
            if out_dim == 0:
                indices = torch.zeros(0, dtype=torch.long)
            elif ratio <= 0.0:
                indices = torch.zeros(0, dtype=torch.long)
            elif ratio >= 1.0:
                indices = torch.arange(out_dim, dtype=torch.long)
            else:
                k = max(1, int(math.ceil(out_dim * ratio)))
                view_tensor = tensor.view(out_dim, -1)
                norms = torch.norm(view_tensor, p=2, dim=1)
                k = min(k, out_dim)
                _, topk = torch.topk(norms, k)
                indices = torch.sort(topk).values
            selected[base] = indices.cpu()
        return selected

    def _build_phi_masks(self, phi_named_params, selected_indices):
        masks = {}
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            if base not in selected_indices:
                continue

            indices = selected_indices[base]
            tensor = param.detach()
            if tensor.dim() == 0:
                mask = torch.ones_like(tensor)
            else:
                mask = torch.zeros_like(tensor)
                if indices.numel() > 0:
                    slicer = [indices.to(tensor.device)] + [slice(None)] * (tensor.dim() - 1)
                    mask[tuple(slicer)] = 1.0
            masks[name] = mask
        return masks

    def _apply_phi_gradient_masks(self):
        if not self._phi_masks:
            return
        named_params = dict(self.model.named_parameters())
        for name, mask in self._phi_masks.items():
            param = named_params[name]
            if param.grad is not None:
                param.grad.mul_(mask.to(param.grad.device))

    def _extract_sparse_phi_updates(self, phi_named_params):
        updates = OrderedDict()
        for name, param in phi_named_params:
            base = self._base_param_name(name)
            indices = self._selected_channel_indices.get(base, torch.zeros(0, dtype=torch.long))
            tensor = param.detach().cpu()

            if tensor.dim() == 0:
                values = tensor.clone()
                payload_indices = None
            elif indices.numel() == 0:
                values = torch.zeros((0,), dtype=tensor.dtype)
                payload_indices = torch.zeros(0, dtype=torch.long)
            elif indices.numel() == tensor.shape[0]:
                values = tensor.clone()
                payload_indices = None
            else:
                slicer = [indices] + [slice(None)] * (tensor.dim() - 1)
                values = tensor[tuple(slicer)].clone()
                payload_indices = indices.clone()

            updates[name] = {
                'indices': payload_indices,
                'values': values,
                'shape': torch.tensor(tensor.shape, dtype=torch.long)
            }
        return updates

    @staticmethod
    def _base_param_name(name: str) -> str:
        if name.endswith('.weight') or name.endswith('.bias'):
            return name.rsplit('.', 1)[0]
        return name
