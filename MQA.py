import torch
from torch import nn

class MutiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads

        # 初始化qkv投影矩阵，用来获得查询、键、值
        self.q_linear = nn.Linear(hidden_size,hidden_size)
        # 注意kv只用一个头
        self.k_linear = nn.Linear(hidden_size,self.head_dim)
        self.v_linear = nn.Linear(hidden_size,self.head_dim)
        self.o_linear = nn.Linear(hidden_size,hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        # kv只用一个头，需要扩展到多头
        key = key.view(batch_size, -1, self.head_dim).unsqueeze(1)
        key = key.expand(-1, self.num_heads, -1, -1)
        value = value.view(batch_size, -1, self.head_dim).unsqueeze(1)
        value = value.expand(-1, self.num_heads, -1, -1)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
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
    attention = MutiQueryAttention(hidden_size, num_heads)
    
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
