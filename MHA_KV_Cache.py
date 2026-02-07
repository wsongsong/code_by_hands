import torch
import torch.nn as nn
import torch.nn.functional as F

class MutiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_batch_size=512, max_seq_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads

        # 初始化qkv投影矩阵，用来获得查询、键、值
        self.q_linear = nn.Linear(hidden_size,hidden_size)
        # 注意kv只用一个头
        self.k_linear = nn.Linear(hidden_size,hidden_size)
        self.v_linear = nn.Linear(hidden_size,hidden_size)
        self.o_linear = nn.Linear(hidden_size,hidden_size)

        # 添加KV缓存
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, self.num_heads, self.head_dim)).cuda()  # 缓存键
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, self.num_heads, self.head_dim)).cuda()  # 缓存值

    def forward(self, hidden_state, mask=None, start_pos=0):
        batch_size = hidden_state.size()[0]
        seq_len = hidden_state.size()[1]
        # 计算qkv
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # 重新调整qkv的形状以适应多头注意力机制
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)

        # 将KV缓存移动到当前设备
        self.cache_k = self.cache_k.to(query)
        self.cache_v = self.cache_v.to(query)
        # 更新KV缓存
        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = key
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = value

        # 从缓存中获取所有键值
        query = query.transpose(1,2)
        key = self.cache_k[:batch_size, :start_pos+seq_len].transpose(1,2)
        value = self.cache_v[:batch_size, :start_pos+seq_len].transpose(1,2)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # 对注意力分数进行归一化(最后一个维度)
        attention_probs = torch.softmax(attention_scores, dim = -1)
        output = torch.matmul(attention_probs, value)
        # 对注意力输出进行拼接
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.head_dim*self.num_heads)
        output = self.o_linear(output)
        return output
    
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