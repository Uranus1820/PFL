import copy
import numpy as np
import torch
import time
import openpyxl as op
from sklearn.mixture import GaussianMixture  # <-- 导入 GMM
from flcore.clients.clientPFL import clientPFL  # <-- 导入新客户端
from utils.data_utils import read_client_data
from threading import Thread


class PFL(object):  # <-- 重命名为 PFL
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        # self.global_model = copy.deepcopy(args.model) # <-- 移除: 没有单一全局模型
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        # self.uploaded_weights = [] # <-- 移除
        # self.uploaded_ids = [] # <-- 移除
        # self.uploaded_models = [] # <-- 移除
        # self.aggregate_params = [] # <-- 移除

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        # --- GMM 和集群模型所需的新增属性 ---
        self.num_clusters = args.num_clusters
        self.layer_idx = args.layer_idx
        self.gmm = GaussianMixture(n_components=self.num_clusters, covariance_type='diag')

        # 维护 K 个集群模型
        self.cluster_models = [
            copy.deepcopy(args.model) for _ in range(self.num_clusters)
        ]

        # 存储客户端的 psi 向量和集群分配
        self.client_psi_vectors = {i: None for i in range(self.num_clients)}
        self.client_cluster_ids = {i: np.random.randint(0, self.num_clusters) for i in
                                   range(self.num_clients)}  # 随机初始分配

        self.uploaded_updates = []  # 存储 (state_dict, w_i)
        # --- 结束新增 ---

        self.set_clients(args, clientPFL)  # <-- 使用新的 clientPFL

        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"GMM Clusters: {self.num_clusters}")  # <-- 新增
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        # best_acc = 0.0
        # accs = [0.0] * self.num_clients # <-- FLAYER 的 acc 不再需要

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # 1. (新) GMM 聚类 (在发送模型之前)
            if i > 0:  # 从第二轮开始聚类
                self.perform_gmm_clustering()

            # 2. (修改) 发送集群模型
            self.send_cluster_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate cluster models")
                # (评估逻辑也需要修改)
                self.evaluate(nonprint=None)

            for client in self.selected_clients:
                client.train()

            # 3. (修改) 接收个性化更新
            self.receive_pf_updates()

            # 4. (修改) 聚合 K 个集群模型
            self.aggregate_cluster_models()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def set_clients(self, args, clientObj):  # 保持不变
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):  # 保持不变
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_cluster_models(self):
        """ (替换 send_models)
        发送客户端所属集群的最新模型
        """
        assert (len(self.clients) > 0)
        for client in self.clients:  # 发送给所有客户端
            k = self.client_cluster_ids[client.id]  # 获取客户端的集群ID
            model_k = self.cluster_models[k]  # 获取对应的集群模型
            client.set_model(model_k.state_dict())  # 发送

    # def save_models(self, i=None): # (可选，保持不变)
    #     for client in self.clients:
    #         client.save_models(i)

    def receive_pf_updates(self):
        """ (替换 receive_models)
        接收 GMM 向量 和模型更新
        """
        assert (len(self.selected_clients) > 0)
        self.uploaded_updates = []

        s_t = time.time()
        for client in self.selected_clients:
            # 1. 获取客户端更新
            state_dict, psi_vector, alpha, samples = client.get_pf_updates()

            # 2. 存储 GMM 向量 以备下次聚类
            self.client_psi_vectors[client.id] = psi_vector

            # 3. 计算聚合权重 w_i
            #    我们使用 .md 中的 w_i = w_i^res * (其他)
            #    简化版：w_i = (alpha * 样本数)
            #    (alpha 是 w_i^res，样本数是数据质量的代理)
            w_i = alpha * samples

            # 4. 存储 (state_dict, 权重, 集群ID)
            k = self.client_cluster_ids[client.id]
            self.uploaded_updates.append((state_dict, w_i, k))

        print(f"GMM Vector Reception:")
        print('-' * 50, time.time() - s_t)

    def perform_gmm_clustering(self):
        """ (新方法)
        使用 client_psi_vectors 拟合 GMM 并更新 client_cluster_ids
        """
        s_t = time.time()
        # 1. 收集所有已上传的 psi 向量
        psi_vectors = [vec for vec in self.client_psi_vectors.values() if vec is not None]
        client_ids = [cid for cid, vec in self.client_psi_vectors.items() if vec is not None]

        if len(psi_vectors) < self.num_clusters:
            print("GMM Warning: 客户端数量不足以聚类，跳过本轮聚类。")
            return

        # 2. 拟合 GMM
        self.gmm.fit(psi_vectors)

        # 3. 分配新集群
        labels = self.gmm.predict(psi_vectors)
        for cid, label in zip(client_ids, labels):
            self.client_cluster_ids[cid] = label

        print("GMM Clustering complete:")
        print(f"Cluster assignment: {self.client_cluster_ids}")
        print('-' * 50, time.time() - s_t)

    def aggregate_cluster_models(self):
        """ (替换 aggregate_parameters)
        执行 K 个集群的个性化聚合
        """
        s_t = time.time()

        # 1. K 个新的聚合模型 (清零)
        new_cluster_models = [
            copy.deepcopy(model) for model in self.cluster_models
        ]
        for model in new_cluster_models:
            for param in model.parameters():
                param.data.zero_()

        total_weights = {k: 1e-9 for k in range(self.num_clusters)}  # 避免除零

        # 2. 累加加权参数
        for (state_dict, w_i, k) in self.uploaded_updates:
            total_weights[k] += w_i

            temp_model = copy.deepcopy(self.cluster_models[k])
            temp_model.load_state_dict(state_dict)

            for (agg_param, client_param) in zip(new_cluster_models[k].parameters(), temp_model.parameters()):
                agg_param.data += client_param.data * w_i

        # 3. 执行平均
        for k in range(self.num_clusters):
            for param in new_cluster_models[k].parameters():
                param.data /= total_weights[k]
            # 更新集群模型
            self.cluster_models[k] = new_cluster_models[k]

        print("Aggregate clusters:")
        print('-' * 50, time.time() - s_t)

    # ... (add_parameters 已不再需要) ...

    def test_metrics(self):
        """ (修改) 评估所有客户端在其 *所属集群模型* 上的表现 """
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        for c in self.clients:
            k = self.client_cluster_ids[c.id]  # 获取集群ID
            model_k = self.cluster_models[k]  # 获取集群模型

            # 在集群模型上评估客户端 c
            ct, ns, auc = c.test_metrics(model=model_k)

            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            accs.append(ct * 1.0 / ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, accs

    def train_metrics(self):
        """ (修改) 评估所有客户端在其 *所属集群模型* 上的表现 """
        num_samples = []
        losses = []
        for c in self.clients:
            k = self.client_cluster_ids[c.id]  # 获取集群ID
            model_k = self.cluster_models[k]  # 获取集群模型

            cl, ns = c.train_metrics(model=model_k)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    # evaluate, set_parameters, get_parameters 基本保持不变
    # (注意：evaluate 内部调用的 test_metrics 和 train_metrics 已被修改，所以评估是正确的)

    def evaluate(self, acc=None, loss=None, nonprint=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        losses = [a / n for a, n in zip(stats_train[2], stats_train[1])]

        if nonprint == None:
            if acc == None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)
            if loss == None:
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

# --- 删掉文件末尾的 aggregate_sparse 函数 ---