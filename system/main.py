import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

import multiprocessing

from flcore.servers.serverPFL import PFL
from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635  # AG News 数据集的词汇表大小
max_len = 200       # 文本最大长度
hidden_dim = 32     # 隐藏层维度

def run(args):

    time_list = []
    model_str = args.model
    args.model_str = model_str

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        # CNN模型: 根据数据集调整输入通道数
        # MNIST: 1通道 (灰度图)
        # CIFAR: 3通道 (彩色图)
        # 其他: 3通道
        # ResNet模型: 使用预训练的 ResNet-18
        # fastText模型: 用于文本分类
        if model_str == "cnn":
            if args.dataset[:5] == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset[:5] == "Cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError
                            
        print(args.model)
        

        if args.algorithm == "PFL":
            server = PFL(args, i)
        else:
            raise NotImplementedError
            
        server.train()
        
        # torch.cuda.empty_cache()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])  # 计算设备选择
    parser.add_argument('-did', "--device_id", type=str, default="0")  # GPU设备ID
    
    # 4.2 数据和模型参数
    parser.add_argument('-data', "--dataset", type=str, default="mnist")  # 数据集名称
    parser.add_argument('-nb', "--num_classes", type=int, default=10)  # 分类数量
    parser.add_argument('-m', "--model", type=str, default="cnn")  # 模型类型 (cnn/resnet/fastText)
    
    # 4.3 训练超参数
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)  # 批次大小
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")  # 本地学习率
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)  # 全局训练轮数
    parser.add_argument('-ls', "--local_steps", type=int, default=1)  # 本地训练步数
    
    # 4.4 联邦学习算法参数
    parser.add_argument('-algo', "--algorithm", type=str, default="PFL")  # 算法类型
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 每轮参与客户端比例
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")  # 是否随机选择参与比例
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")  # 总客户端数量
    parser.add_argument('--num_clusters', type=int, default=2,
                        help="Number of GMM clusters for personalized aggregation")

    # 4.5 实验控制参数
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")  # 之前运行次数
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")  # 总运行次数
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")  # 评估间隔轮数
    
    # 4.6 FLAYER算法特定参数
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")  # 层级索引，用于稀疏更新

    parser.add_argument('--gamma', type=float, default=1.0,
                        help="Client preference for performance gain in utility function")
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help="Saturation coefficient in the contribution function")
    parser.add_argument('--alpha_min', type=float, default=0.0,
                        help="Minimum feasible training proportion per client")
    parser.add_argument('--alpha_max', type=float, default=1.0,
                        help="Maximum feasible training proportion per client")
    parser.add_argument('--resource_cost', type=float, default=1.0,
                        help="Baseline resource cost per unit training proportion")
    parser.add_argument('--gmm_sigma', type=float, default=1.0,
                        help="Bandwidth used in resource-aware GMM attention weighting")
    parser.add_argument('--resource_only_interval', type=int, default=0,
                        help="Interval for resource-only aggregation rounds (0 disables the feature)")
    args = parser.parse_args()

    # 4.7 设备环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id  # 设置可见的GPU设备
    # torch.cuda.set_device(int(args.device_id))  # 备用GPU设置方法

    # 4.8 CUDA可用性检查
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"  # 如果CUDA不可用，自动切换到CPU

    # 4.9 启动联邦学习训练
    run(args)
