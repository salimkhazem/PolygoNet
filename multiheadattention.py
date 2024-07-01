import math
import torch
import torch.nn as nn


class SplineMultiHead(nn.Module):
    def __init__(self, input_dim, num_heads, d_k, out_size, batch_first=False):
        super(SplineMultiHead, self).__init__()
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_k
        self.query_project = nn.Linear(input_dim, d_k * num_heads)
        self.key_project = nn.Linear(input_dim, d_k * num_heads)
        self.value_project = nn.Linear(input_dim, d_k * num_heads)
        self.attention = nn.MultiheadAttention(
            d_k * num_heads, num_heads, batch_first=True
        )
        self.out_size = out_size
        self.query = nn.Parameter(torch.empty((out_size, d_k * num_heads)))
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

    def forward(self, x):
        bs, _, _ = x.size()
        query = self.query.unsqueeze(dim=0).expand(
            (bs, self.out_size, self.d_k * self.num_heads)
        )
        key = self.key_project(x)  # --> BS, N, num_head * embed (BS, N, 4)
        value = self.value_project(x)  # --> BS, N, num_head * embed (BS, N, 4)
        attention, attn_weights = self.attention(query, key, value)
        attention = attention.view(bs, self.out_size, -1, self.num_heads)
        return attention, attn_weights


if __name__ == "__main__":

    query, key = torch.randn(64, 256, 2), torch.randn(64, 300, 2)
    value = torch.randn(64, 300, 2)
    model = SplineMultiHead(2, 1, d_k=2, out_size=128, batch_first=True)
    out1, out2 = model(key)
    out3, out4 = model(query)
    print(f"{out1.shape} | {out2.shape} | {out3.shape} | {out4.shape}")
