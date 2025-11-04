import torch
import torch.nn as nn


# 批次大小（未使用）
batch_size = 16

class LocalModel(nn.Module):
    """
    本地模型类，由特征提取器和分类头组成
    """
    def __init__(self, feature_extractor, head):
        """
        初始化本地模型

        Args:
            feature_extractor: 特征提取器网络
            head: 分类头网络
        """
        super(LocalModel, self).__init__()
        # 特征提取层
        self.feature_extractor = feature_extractor
        # 分类层
        self.head = head
    def forward(self, x, feat=False):
        """
        前向传播

        Args:
            x: 输入数据
            feat: 如果为True，只返回特征；否则返回分类结果
            
        Returns:
            torch.Tensor: 特征或分类结果
        """
        # 提取特征
        out = self.feature_extractor(x)
        if feat:
            # 只返回特征
            return out
        else:
            # 通过分类头得到最终输出
            out = self.head(out)
            return out


class FedAvgCNN(nn.Module):
    """
    FedAvg使用的CNN模型，适用于图像分类任务
    结构：Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> FC
    """
    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        """
        初始化CNN模型

        Args:
            in_features: 输入通道数（1为灰度图，3为彩色图）
            num_classes: 分类类别数
            dim: 第一个全连接层的输入维度
            dim1: 第一个全连接层的输出维度（也是第二个全连接层的输入维度）
        """
        super().__init__()
        # 第一个卷积块：输入通道 -> 32通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),  # 原地激活以节省内存
            nn.MaxPool2d(kernel_size=(2, 2))  # 2x2最大池化
        )
        # 第二个卷积块：32通道 -> 64通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # 第一个全连接层：dim -> dim1
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1), 
            nn.ReLU(inplace=True)
        )
        # 分类层：dim1 -> num_classes
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        # 第一个卷积块
        out = self.conv1(x)
        # 第二个卷积块
        out = self.conv2(out)
        # 展平特征图
        out = torch.flatten(out, 1)
        # 第一个全连接层
        out = self.fc1(out)
        # 分类层
        out = self.fc(out)
        return out


class fastText(nn.Module):
    """
    fastText模型，用于文本分类任务
    结构：Embedding -> Mean Pooling -> FC -> FC
    """
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        """
        初始化fastText模型
        
        Args:
            hidden_dim: 隐藏层维度（也是embedding维度）
            padding_idx: padding索引
            vocab_size: 词汇表大小
            num_classes: 分类类别数
        """
        super(fastText, self).__init__()
        
        # 词嵌入层：将词汇索引转换为向量
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # 隐藏层（全连接层）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层（分类层）
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入元组 (text, text_lengths)
                - text: 文本序列 [batch_size, seq_len]
                - text_lengths: 文本长度 [batch_size]
            
        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        text, text_lengths = x

        # 词嵌入：将词汇索引转换为向量
        embedded_sent = self.embedding(text)  # [batch_size, seq_len, hidden_dim]
        # 平均池化：对序列维度求平均
        h = self.fc1(embedded_sent.mean(1))  # [batch_size, hidden_dim]
        # 分类输出
        z = self.fc(h)  # [batch_size, num_classes]
        out = z

        return out
