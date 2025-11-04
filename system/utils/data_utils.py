# 导入标准库
import numpy as np  # 数值计算
import os  # 操作系统接口
import torch  # PyTorch深度学习框架

# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    """
    将数据分批处理
    
    Args:
        data: 数据字典，包含'x'和'y'两个numpy数组（单个客户端的数据）
        batch_size: 批次大小
        
    Yields:
        tuple: (batched_x, batched_y)，每个都是长度为batch_size的numpy数组
    """
    data_x = data['x']
    data_y = data['y']

    # 随机打乱数据（保持x和y的对应关系）
    ran_state = np.random.get_state()  # 保存随机状态
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)  # 恢复随机状态以确保x和y同步打乱
    np.random.shuffle(data_y)

    # 循环生成小批次
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    """
    随机获取一个批次的数据样本
    
    Args:
        data_x: 特征数据数组
        data_y: 标签数据数组
        batch_size: 批次大小
        
    Returns:
        tuple: (batch_x, batch_y) 随机批次的数据
    """
    # 计算可以划分的批次数量
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        # 随机选择一个批次索引
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        # 如果索引超出范围，返回从索引到末尾的数据
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            # 返回指定批次的数据
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        # 如果数据量小于批次大小，返回全部数据
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    """
    获取第一个批次的数据样本（打乱后）
    
    Args:
        data: 数据字典，包含'x'和'y'
        batch_size: 批次大小
        
    Returns:
        tuple: (batched_x, batched_y) 第一个批次的数据
    """
    data_x = data['x']
    data_y = data['y']

    # 随机打乱数据（保持x和y的对应关系）
    # np.random.seed(100)  # 可选：设置随机种子
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # 获取第一个批次
    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx, is_train=True):
    """
    从磁盘读取客户端数据文件
    
    Args:
        dataset: 数据集名称
        idx: 客户端索引
        is_train: 是否读取训练数据（True为训练数据，False为测试数据）
        
    Returns:
        dict: 数据字典，包含'x'和'y'键
    """
    if is_train:
        # 构建训练数据目录路径
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        # 构建文件路径
        train_file = train_data_dir + str(idx) + '.npz'
        # 读取.npz文件
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        # 构建测试数据目录路径
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        # 构建文件路径
        test_file = test_data_dir + str(idx) + '.npz'
        # 读取.npz文件
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    """
    读取客户端数据并转换为PyTorch张量格式
    
    Args:
        dataset: 数据集名称
        idx: 客户端索引
        is_train: 是否读取训练数据
        
    Returns:
        list: 数据对列表，每个元素为 (x, y) 元组
    """
    # 如果是文本数据集（ag news或SS），使用专门的文本读取函数
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx)

    if is_train:
        # 读取训练数据
        train_data = read_data(dataset, idx, is_train)
        # 转换为PyTorch张量
        X_train = torch.Tensor(train_data['x']).type(torch.float32)  # 特征为float32
        y_train = torch.Tensor(train_data['y']).type(torch.int64)  # 标签为int64

        # 转换为数据对列表
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        # 读取测试数据
        test_data = read_data(dataset, idx, is_train)
        # 转换为PyTorch张量
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        # 转换为数据对列表
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    """
    读取文本数据集的客户端数据并转换为PyTorch张量格式
    
    Args:
        dataset: 数据集名称（文本数据集）
        idx: 客户端索引
        is_train: 是否读取训练数据
        
    Returns:
        list: 数据对列表，每个元素为 ((text, text_length), y) 元组
    """
    if is_train:
        # 读取训练数据
        train_data = read_data(dataset, idx, is_train)
        # 解包文本序列和长度
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        # 转换为PyTorch张量
        X_train = torch.Tensor(X_train).type(torch.int64)  # 文本序列为int64（词汇索引）
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)  # 序列长度为int64
        y_train = torch.Tensor(train_data['y']).type(torch.int64)  # 标签为int64

        # 转换为数据对列表，输入为 (text, text_length) 元组
        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        # 读取测试数据
        test_data = read_data(dataset, idx, is_train)
        # 解包文本序列和长度
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        # 转换为PyTorch张量
        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        # 转换为数据对列表
        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data
