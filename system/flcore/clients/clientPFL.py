# 导入标准库
import copy  # 深拷贝功能
import math  # 数学函数
from collections import OrderedDict  # 有序字典

# 导入第三方库
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from sklearn import metrics  # 评估指标
from sklearn.preprocessing import label_binarize  # 标签二值化
from torch.utils.data import DataLoader  # 数据加载器

# 导入项目自定义模块
from system.utils.data_utils import read_client_data


class clientPFL(object):
    """客户端执行资源自适应的局部训练，并上传解耦的特征提取/分类层参数。"""

    def __init__(self, args, id, train_samples, test_samples):
        """
        初始化客户端
        
        Args:
            args: 配置参数对象
            id: 客户端唯一标识符
            train_samples: 训练样本数量
            test_samples: 测试样本数量
        """
        # 深拷贝模型以避免引用问题
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset  # 数据集名称
        self.device = args.device  # 计算设备（CPU/CUDA）
        self.id = id  # 客户端ID
        self.args = args  # 保存参数对象

        # 训练相关参数
        self.num_classes = args.num_classes  # 分类类别数
        self.train_samples = train_samples  # 训练样本数
        self.test_samples = test_samples  # 测试样本数
        self.batch_size = args.batch_size  # 批次大小
        self.learning_rate = args.local_learning_rate  # 本地学习率
        self.local_steps = args.local_steps  # 本地训练步数

        # 损失函数和层切分
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.layer_idx = args.layer_idx  # 切分特征提取层和分类层

        # --- 资源感知训练与效用建模参数 ---
        self.gamma = float(getattr(args, "gamma", 1.0))
        self.epsilon = float(getattr(args, "epsilon", 1.0))
        self.alpha_min = max(0.0, float(getattr(args, "alpha_min", 0.0)))
        self.alpha_max = min(1.0, float(getattr(args, "alpha_max", 1.0)))
        self.resource_cost = max(1e-6, float(getattr(args, "resource_cost", 1.0)))

        # 引入轻微扰动来模拟客户端偏好差异（增加异构性）
        self.gamma *= float(1.0 + np.random.uniform(-0.1, 0.1))
        self.resource_cost *= float(1.0 + np.random.uniform(-0.1, 0.1))

        # 当前轮的训练比例，初始化为最大值
        self.alpha = self.alpha_max
        # 优化器，将在训练时初始化
        self.optimizer = None

        # 数据质量 Q_i = q_i * d_i (提前计算)
        # q_i: 数据熵质量，d_i: 数据量
        self.data_quality = self._estimate_data_quality()

        # 用于缓存每轮选择的通道索引与掩码
        # 避免重复计算，提高效率
        self._selected_channel_indices = {}
        self._phi_masks = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def set_model(self, model_state_dict):
        """
        设置客户端模型参数

        Args:
            model_state_dict: 服务器下发的全局模型
        """
        # 加载模型参数
        self.model.load_state_dict(model_state_dict)
        # 保存训练前的模型状态，用于后续计算更新
        self.model_before = copy.deepcopy(self.model)

    def train(self):
        """
        执行本地训练，使用资源自适应的更新策略
        """
        # 加载训练数据
        trainloader = self.load_train_data()
        # 如果没有数据则直接返回
        if len(trainloader) == 0:
            return

        # 设置为训练模式
        self.model.train()

        # 计算最优训练比例 alpha（基于效用函数）
        self.alpha = self._determine_optimal_alpha()

        # 获取所有命名参数
        named_params = list(self.model.named_parameters())
        # 切分参数：phi为特征提取层，psi为分类层
        phi_named = named_params[:-self.layer_idx]  # 特征提取层
        psi_named = named_params[-self.layer_idx:]  # 分类层

        # 根据alpha选择要更新的通道索引
        self._selected_channel_indices = self._select_top_channels(phi_named, self.alpha)
        # 构建掩码，只更新选中的通道
        self._phi_masks = self._build_phi_masks(phi_named, self._selected_channel_indices)

        # 筛选出需要优化的phi参数（有非零掩码的）
        phi_params_for_optim = [
            param for name, param in phi_named
            if name in self._phi_masks and torch.count_nonzero(self._phi_masks[name]) > 0
        ]
        # psi参数全部参与优化
        psi_params_for_optim = [param for _, param in psi_named]

        # 构建优化器参数组（不同层使用不同学习率）
        params_for_optimizer = []
        # phi层使用较小的学习率（0.1倍）
        if phi_params_for_optim:
            params_for_optimizer.append({'params': phi_params_for_optim, 'lr': self.learning_rate * 0.1})
        # psi层使用正常学习率
        if psi_params_for_optim:
            params_for_optimizer.append({'params': psi_params_for_optim, 'lr': self.learning_rate})

        # 如果没有可优化的参数则返回
        if not params_for_optimizer:
            return

        # 创建SGD优化器
        self.optimizer = torch.optim.SGD(params_for_optimizer)

        # 执行本地训练步骤
        for _ in range(self.local_steps):
            # 遍历训练数据批次
            for x, y in trainloader:
                # 处理输入数据（支持列表格式，如文本数据）
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播
                output = self.model(x)
                # 计算损失
                loss = self.loss(output, y)

                # 反向传播和参数更新
                self.optimizer.zero_grad()  # 清零梯度
                loss.backward()  # 反向传播
                # 应用掩码，只更新选中的通道
                self._apply_phi_gradient_masks()
                self.optimizer.step()  # 更新参数

    def get_pf_updates(self):
        """
        获取客户端更新，包括稀疏的phi更新和完整的psi更新
        
        Returns:
            dict: 包含phi更新、psi更新、psi向量、alpha和样本数的字典
        """
        # 获取模型参数
        named_params = list(self.model.named_parameters())
        phi_named = named_params[:-self.layer_idx]  # 特征提取层
        psi_named = named_params[-self.layer_idx:]  # 分类层

        # 将psi参数展平为向量（用于GMM聚类）
        psi_vector = torch.cat([
            param.data.view(-1) for _, param in psi_named
        ]).detach().cpu().numpy()

        # 保存psi参数的完整状态字典
        psi_state = OrderedDict(
            (name, param.detach().cpu().clone()) for name, param in psi_named
        )

        # 提取稀疏的phi更新（只包含选中的通道）
        phi_updates = self._extract_sparse_phi_updates(phi_named)

        # 返回更新信息
        return {
            'phi': phi_updates,  # 稀疏的phi更新
            'psi': psi_state,  # 完整的psi参数
            'psi_vector': psi_vector,  # psi向量（用于聚类）
            'alpha': float(self.alpha),  # 训练比例
            'samples': int(self.train_samples),  # 样本数
        }

    # ------------------------------------------------------------------
    # 数据集相关辅助方法
    # ------------------------------------------------------------------
    def load_train_data(self, batch_size=None):
        """
        加载训练数据
        
        Args:
            batch_size: 批次大小，如果为None则使用默认值
            
        Returns:
            DataLoader: 训练数据加载器
        """
        if batch_size is None:
            batch_size = self.batch_size
        # 读取客户端训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # 创建数据加载器（不shuffle，drop_last丢弃最后一个不完整批次）
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        """
        加载测试数据
        
        Args:
            batch_size: 批次大小，如果为None则使用默认值
            
        Returns:
            DataLoader: 测试数据加载器
        """
        if batch_size is None:
            batch_size = self.batch_size
        # 读取客户端测试数据
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        # 创建数据加载器
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        """
        计算测试集上的评估指标
        
        Args:
            model: 要评估的模型，如果为None则使用当前模型
            
        Returns:
            tuple: (正确预测数, 总样本数, AUC分数)
        """
        # 加载测试数据
        testloader = self.load_test_data()
        # 如果没有指定模型，使用当前模型
        if model is None:
            model = self.model
        # 设置为评估模式
        model.eval()

        # 初始化统计变量
        test_acc = 0  # 正确预测数
        test_num = 0  # 总样本数
        y_prob = []  # 预测概率
        y_true = []  # 真实标签

        # 在无梯度模式下进行评估
        with torch.no_grad():
            for x, y in testloader:
                # 处理输入数据
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播
                output = model(x)
                # 计算准确率
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                # 保存预测概率
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())

                # 处理标签二值化（用于AUC计算）
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1  # 二分类需要特殊处理
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # 合并所有批次的预测结果
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # 计算AUC分数
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        """
        计算训练集上的损失指标
        
        Args:
            model: 要评估的模型，如果为None则使用当前模型
            
        Returns:
            tuple: (总损失, 样本数)
        """
        # 加载训练数据
        trainloader = self.load_train_data()
        # 如果没有指定模型，使用当前模型
        if model is None:
            model = self.model
        # 设置为评估模式
        model.eval()

        # 初始化统计变量
        train_num = 0  # 样本数
        losses = 0  # 累积损失
        
        # 在无梯度模式下计算损失
        with torch.no_grad():
            for x, y in trainloader:
                # 处理输入数据
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 前向传播计算损失
                output = self.model(x)
                loss = self.loss(output, y)
                # 累积统计信息
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _estimate_data_quality(self):
        """
        估计数据质量 Q_i = q_i * d_i
        q_i: 基于熵的标签分布质量
        d_i: 数据量
        
        Returns:
            float: 数据质量分数
        """
        # 读取训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # 提取所有标签
        labels = []
        for _, y in train_data:
            if isinstance(y, torch.Tensor):
                labels.append(int(y.item()))
            else:
                labels.append(int(y))

        # 数据量
        d_i = max(1, len(labels))
        # 统计各类别样本数
        label_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        # 计算类别概率
        prob = label_counts / max(label_counts.sum(), 1.0)
        # 计算熵（衡量数据分布的均匀性）
        non_zero = prob > 0
        entropy = -np.sum(prob[non_zero] * np.log(prob[non_zero] + 1e-12))
        # 最大熵（完全均匀分布时的熵）
        max_entropy = math.log(min(self.num_classes, np.count_nonzero(label_counts)) or 1)
        # 归一化熵作为质量分数（0-1之间）
        q_i = (entropy / max_entropy) if max_entropy > 0 else 0.0
        q_i = float(max(q_i, 1e-6))  # 避免为0
        # 返回数据质量（质量分数 × 数据量）
        return q_i * d_i

    def _utility(self, alpha):
        """
        计算效用函数 U(alpha) = gamma * Q_i * (1 - exp(-epsilon * alpha)) - alpha * C
        
        Args:
            alpha: 训练比例
            
        Returns:
            float: 效用值
        """
        # 如果alpha <= 0，返回负的资源消耗
        if alpha <= 0:
            return -self.resource_cost * max(alpha, 0.0)
        # 计算贡献项（饱和函数）
        contribution = 1.0 - math.exp(-self.epsilon * alpha)
        # 效用 = 性能收益 - 资源消耗
        return self.gamma * self.data_quality * contribution - alpha * self.resource_cost

    def _determine_optimal_alpha(self):
        """
        通过求解效用函数的最优化问题来确定最优训练比例alpha
        
        Returns:
            float: 最优的alpha值
        """
        # 确定可行域
        feasible_min = max(0.0, self.alpha_min)
        feasible_max = min(1.0, max(feasible_min, self.alpha_max))
        if feasible_max <= 0:
            return 0.0

        # 计算效用函数的导数为0的点（最优解）
        numerator = self.epsilon * self.gamma * self.data_quality
        vertex = float('-inf')
        if numerator > 0 and self.resource_cost > 0:
            ratio = numerator / self.resource_cost
            if ratio > 0:
                vertex = math.log(ratio) / self.epsilon

        # 如果最优解小于下界，检查下界点
        if vertex < feasible_min:
            candidate = feasible_min
            # 如果下界点的效用为负，返回0（不训练）
            return 0.0 if self._utility(candidate) < 0 else candidate

        # 如果最优解大于上界，返回上界
        if vertex > feasible_max:
            return feasible_max

        # 返回最优解（在可行域内）
        return max(0.0, vertex)

    def _select_top_channels(self, phi_named_params, ratio):
        """
        根据比例选择top-k通道（基于L2范数）
        
        Args:
            phi_named_params: phi层的命名参数列表
            ratio: 选择比例（0-1之间）
            
        Returns:
            dict: 每个参数对应的选中通道索引字典
        """
        selected = {}
        # 确保ratio在[0, 1]范围内
        ratio = max(0.0, min(1.0, float(ratio)))
        
        for name, param in phi_named_params:
            # 获取基础参数名（去除.weight/.bias后缀）
            base = self._base_param_name(name)
            # bias参数与对应的weight共享通道索引
            if name.endswith('bias') and base in selected:
                continue

            tensor = param.detach()
            # 获取输出维度（通道数）
            out_dim = tensor.shape[0] if tensor.dim() > 0 else 1
            
            # 处理边界情况
            if out_dim == 0:
                indices = torch.zeros(0, dtype=torch.long)
            elif ratio <= 0.0:
                indices = torch.zeros(0, dtype=torch.long)  # 不选择任何通道
            elif ratio >= 1.0:
                indices = torch.arange(out_dim, dtype=torch.long)  # 选择所有通道
            else:
                # 计算要选择的通道数
                k = max(1, int(math.ceil(out_dim * ratio)))
                # 将参数重塑为(out_dim, -1)以便计算每个通道的范数
                view_tensor = tensor.view(out_dim, -1)
                # 计算每个通道的L2范数
                norms = torch.norm(view_tensor, p=2, dim=1)
                k = min(k, out_dim)
                # 选择范数最大的k个通道
                _, topk = torch.topk(norms, k)
                indices = torch.sort(topk).values  # 排序以便后续使用
            selected[base] = indices.cpu()
        return selected

    def _build_phi_masks(self, phi_named_params, selected_indices):
        """
        根据选中的通道索引构建掩码
        
        Args:
            phi_named_params: phi层的命名参数列表
            selected_indices: 选中的通道索引字典
            
        Returns:
            dict: 参数名到掩码的映射
        """
        masks = {}
        for name, param in phi_named_params:
            # 获取基础参数名
            base = self._base_param_name(name)
            # 如果该参数没有对应的选中索引，跳过
            if base not in selected_indices:
                continue

            indices = selected_indices[base]
            tensor = param.detach()
            # 处理标量参数
            if tensor.dim() == 0:
                mask = torch.ones_like(tensor)  # 标量总是更新
            else:
                # 创建全零掩码
                mask = torch.zeros_like(tensor)
                if indices.numel() > 0:
                    # 构建切片索引，只选中指定通道
                    slicer = [indices.to(tensor.device)] + [slice(None)] * (tensor.dim() - 1)
                    # 将选中通道位置设为1
                    mask[tuple(slicer)] = 1.0
            masks[name] = mask
        return masks

    def _apply_phi_gradient_masks(self):
        """
        应用掩码到梯度，只更新选中的通道
        """
        # 如果没有掩码则直接返回
        if not self._phi_masks:
            return
        # 获取所有参数
        named_params = dict(self.model.named_parameters())
        # 对每个有掩码的参数应用掩码
        for name, mask in self._phi_masks.items():
            param = named_params[name]
            # 如果梯度存在，则应用掩码
            if param.grad is not None:
                param.grad.mul_(mask.to(param.grad.device))  # 原位乘法

    def _extract_sparse_phi_updates(self, phi_named_params):
        """
        提取稀疏的phi更新（只包含选中的通道）
        
        Args:
            phi_named_params: phi层的命名参数列表
            
        Returns:
            OrderedDict: 稀疏更新字典，包含索引、值和形状信息
        """
        updates = OrderedDict()
        for name, param in phi_named_params:
            # 获取基础参数名和对应的选中索引
            base = self._base_param_name(name)
            indices = self._selected_channel_indices.get(base, torch.zeros(0, dtype=torch.long))
            tensor = param.detach().cpu()

            # 处理不同情况
            if tensor.dim() == 0:
                # 标量参数：直接保存值
                values = tensor.clone()
                payload_indices = None
            elif indices.numel() == 0:
                # 没有选中任何通道：返回空张量
                values = torch.zeros((0,), dtype=tensor.dtype)
                payload_indices = torch.zeros(0, dtype=torch.long)
            elif indices.numel() == tensor.shape[0]:
                # 选中了所有通道：不需要索引，直接保存完整张量
                values = tensor.clone()
                payload_indices = None
            else:
                # 部分选中：保存选中通道的值和索引
                slicer = [indices] + [slice(None)] * (tensor.dim() - 1)
                values = tensor[tuple(slicer)].clone()
                payload_indices = indices.clone()

            # 保存更新信息
            updates[name] = {
                'indices': payload_indices,  # 通道索引（如果部分选中）
                'values': values,  # 参数值
                'shape': torch.tensor(tensor.shape, dtype=torch.long)  # 原始形状
            }
        return updates

    @staticmethod
    def _base_param_name(name: str) -> str:
        """
        提取参数的基础名称（去除.weight/.bias后缀）
        
        Args:
            name: 完整参数名
            
        Returns:
            str: 基础参数名
        """
        if name.endswith('.weight') or name.endswith('.bias'):
            return name.rsplit('.', 1)[0]
        return name
