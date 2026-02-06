import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, group_size, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = hidden_size // num_heads
        self.group_heads = num_heads // group_size

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.group_size*self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.group_size*self.head_dim)
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_state, attention_mask=None):
        batch_size, seq_size, _ = hidden_state.size()
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        query = query.view(batch_size, seq_size, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, seq_size, self.group_size, self.head_dim).permute(0,2,3,1)
        value = value.view(batch_size, seq_size, self.group_size, self.head_dim).transpose(1,2)

        key = key.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        key = key.view(batch_size, self.num_heads, self.head_dim, seq_size)
        value = value.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        value = value.view(batch_size, self.num_heads, seq_size, self.head_dim)

        attn_scores = torch.matmul(query,key)
        attn_scores = attn_scores/torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask is not None:
            mask = attention_mask.view(batch_size, 1, 1, seq_size)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_size, -1)
        return self.o_linear(output)

