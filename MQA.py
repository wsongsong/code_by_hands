import torch
from torch import nn

class MutiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiQueryAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads

        self.q_linear = nn.Linear(hidden_size,hidden_size)
        self.k_linear = nn.Linear(hidden_size,self.head_dim)
        self.v_linear = nn.Linear(hidden_size,self.head_dim)
        self.o_linear = nn.Linear(hidden_size,hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        query = self.split_head(query)
        key = self.split(key, 1)
        value = self.split(value, 1)

        key = key.expand(-1, self.num_heads, -1, -1)
        value = value.expand(-1, self.num_heads, -1, -1)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)
        return output
    def split_head(self, x, head_num = None):
        batch_size = x.size()[0]
        if head_num is None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1,2)
