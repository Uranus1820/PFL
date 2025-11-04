import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from system.utils.data_utils import read_client_data


# 注意：我们不再导入 FLAYER_aggregation，因为服务器会处理聚合

class clientPFL(object):
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

        # 核心参数
        self.layer_idx = args.layer_idx  # 用于切分 phi / psi
        self.alpha = args.alpha  # 用于资源感知训练

        # 优化器将在 train() 方法中动态定义
        self.optimizer = None

    def set_model(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.model_before = copy.deepcopy(self.model)  # 保存训练前的状态

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        # 1. 使用 layer_idx 区分参数
        all_params = list(self.model.parameters())
        phi_params = all_params[:-self.layer_idx]  # 特征提取层 (phi)
        psi_params = all_params[-self.layer_idx:]  # 特征分类层 (psi)

        # 2. 实现 "资源感知的模型收缩"

        num_phi_layers_to_train = int(len(phi_params) * self.alpha)


        # 如果 alpha < 1.0，只训练 phi 的后几层
        if num_phi_layers_to_train < len(phi_params):
            split_idx = len(phi_params) - num_phi_layers_to_train
            params_to_freeze = phi_params[:split_idx]
            params_to_train_phi = phi_params[split_idx:]
        else:
            params_to_freeze = []
            params_to_train_phi = phi_params

        # 冻结 phi 的前几层
        for param in params_to_freeze:
            param.requires_grad = False

        # 3. 定义优化器
        #    总是训练完整的 psi (用于 GMM) 和部分的 phi
        params_for_optimizer = [
            {'params': params_to_train_phi, 'lr': self.learning_rate * 0.1},  # 给 phi 较小的 LR
            {'params': psi_params, 'lr': self.learning_rate}  # 给 psi 正常的 LR
        ]
        self.optimizer = torch.optim.SGD(params_for_optimizer)

        # 4. 训练循环
        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 5. 解冻所有参数，以便下次训练
        for param in params_to_freeze:
            param.requires_grad = True

    def get_pf_updates(self):
        """
        (替换 get_parameters_sparse)
        返回 GMM 聚类 所需的 psi 向量，以及聚合所需的完整参数。
        """
        all_params = list(self.model.parameters())

        # 1. 获取 psi (分类层) 参数
        psi_params = all_params[-self.layer_idx:]  #

        # 2. 为 GMM 聚类 创建展平的 psi 向量
        psi_vector = torch.cat(
            [p.data.view(-1) for p in psi_params]
        ).detach().cpu().numpy()

        # 3. 返回更新
        #    “部分 phi 上传”，




        #
        return self.model.state_dict(), psi_vector, self.alpha, self.train_samples

    # --- (load_train_data, load_test_data, test_metrics, train_metrics 保持不变) ---
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(F.softmax(output).detach().cpu().numpy())
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
        if model == None:
            model = self.model
        model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num