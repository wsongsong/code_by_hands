import torch
from torch import nn
import numpy as np

class MutiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # 初始化qkv投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        # 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, mask=None):
        batch_size = hidden_state.size()[0]
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim = -1)
        output = torch.matmul(attention_probs, value)
        # 对注意力输出进行拼接
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.head_dim*self.num_heads)
        output = self.o_linear(output)
        return output
    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim)
    
# 快速测试版本
def quick_test():
    # 简单测试配置
    batch_size = 2
    seq_len = 3
    hidden_size = 64
    num_heads = 4
    
    # 创建模型
    attention = MutiHeadAttention(hidden_size, num_heads)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    output = attention(x)
    
    print("快速测试结果:")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入输出形状匹配: {x.shape == output.shape}")
    print(f"注意力层工作正常: {not torch.isnan(output).any()}")
    
    return attention, x, output

# 运行快速测试
model, input_tensor, output_tensor = quick_test()