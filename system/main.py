import argparse
import os
import time
import warnings

import numpy as np
import torch
import torchvision

from flcore.servers.serverPFL import PFL
from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
torch.manual_seed(0)

VOCAB_SIZE = 98635
HIDDEN_DIM = 32


def run(args):
    time_list = []
    model_str = args.model
    args.model_str = model_str

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "cnn":
            if args.dataset[:5].lower() == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset[:5].lower() == "cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "fastText":
            args.model = fastText(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
                                  num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError(f"Unknown model type {model_str}")

        print(args.model)

        server = PFL(args, i)

        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # 设备配置
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run training on.")
    parser.add_argument('-did', "--device_id", type=str, default="0",
                        help="Visible CUDA device id.")

    # 数据集与模型
    parser.add_argument('-data', "--dataset", type=str, default="mnist",
                        help="Dataset name.")
    parser.add_argument('-nb', "--num_classes", type=int, default=10,
                        help="Number of labels.")
    parser.add_argument('-m', "--model", type=str, default="cnn",
                        help="Model backbone (cnn/resnet/fastText).")

    # 本地训练
    parser.add_argument('-lbs', "--batch_size", type=int, default=10,
                        help="Default local batch size.")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate.")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200,
                        help="Number of global FL rounds.")
    parser.add_argument('-ls', "--local_steps", type=int, default=1,
                        help="Local epochs per round.")

    # 客户端参与
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients sampled per round.")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients.")

    # 实验控制
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previously finished runs (for resume).")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Total number of experiments.")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Evaluation frequency in global rounds.")

    # 模型切分
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="Number of layers treated as the classification head.")

    parser.add_argument('--alpha_min', type=float, default=0.0,
                        help="Minimum feasible training ratio.")
    parser.add_argument('--alpha_max', type=float, default=1.0,
                        help="Maximum feasible training ratio.")
    parser.add_argument('--gmm_sigma', type=float, default=1.0,
                        help="Bandwidth used for similarity attention weights.")
    parser.add_argument('--gmm_init_components', type=int, default=0,
                        help="Initial number of components for dynamic GMM (0 defaults to num_clients).")
    parser.add_argument('--epsilon_merge', type=float, default=1e-2,
                        help="Merge threshold for redundant GMM components.")

    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help="Discount factor for the Markov game.")
    parser.add_argument('--td_learning_rate', type=float, default=0.1,
                        help="TD-learning step size.")
    parser.add_argument('--policy_epsilon', type=float, default=0.1,
                        help="Exploration rate for the client policy.")
    parser.add_argument('--alpha_samples', type=int, default=5,
                        help="Number of discrete alpha candidates.")
    parser.add_argument('--max_batch_size', type=int, default=-1,
                        help="Maximum minibatch size for decision space (<=0 uses batch_size).")
    parser.add_argument('--batch_step', type=int, default=1,
                        help="Step between candidate minibatch sizes.")
    parser.add_argument('--proc_cost', type=float, default=1.0,
                        help="Processing cost coefficient in the private cost model.")
    parser.add_argument('--comm_cost', type=float, default=0.1,
                        help="Communication cost coefficient in the private cost model.")

    parser.add_argument('--reward_p0', type=float, default=1.0,
                        help="Base participation reward.")
    parser.add_argument('--reward_pJ', type=float, default=0.1,
                        help="Reward factor for extra data usage.")
    parser.add_argument('--reward_palpha', type=float, default=0.1,
                        help="Reward factor for larger training ratios.")
    parser.add_argument('--reward_rho_q', type=float, default=0.5,
                        help="Penalty coefficient linked to similarity shortfall.")

    args = parser.parse_args()

    if args.max_batch_size <= 0:
        args.max_batch_size = args.batch_size
    if args.gmm_init_components <= 0:
        args.gmm_init_components = args.num_clients

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)
