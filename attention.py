import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(SplineAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Linear(input_size, hidden_size * num_heads)
        self.key = nn.Linear(input_size, hidden_size * num_heads)
        self.value = nn.Linear(input_size, hidden_size * num_heads)

    def forward(self, input_tensor):
        """
        Arguments:
            input_tensor: torch.Tensor of shape (B, N, input_size)

        Returns:
            attended_values: torch.Tensor of shape (B, num_points, num_heads*num_values)
            attention_weights: torch.Tensor of shape (B, num_points....) TODO
        """
        batch_size, num_points, _ = input_tensor.size()

        q = self.query(input_tensor).view(
            batch_size, num_points, self.num_heads, self.hidden_size
        )
        k = self.key(input_tensor).view(
            batch_size, num_points, self.num_heads, self.hidden_size
        )
        v = self.value(input_tensor).view(
            batch_size, num_points, self.num_heads, self.hidden_size
        )

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.hidden_size**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, v)
        attended_values = attended_values.view(batch_size, num_points, -1)

        return attended_values, attention_weights


if __name__ == "__main__":
    input_size = 2  # Input size for each point (x, y)
    hidden_size = 32  # Hidden size of the attention module
    num_heads = 4  # Number of attention heads
    attention = SplineAttention(input_size, hidden_size, num_heads)
    model = nn.Sequential(
        nn.Conv1d(128, 256, 3, padding_mode="circular", padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 512, 3, padding_mode="circular", padding=1),
        nn.ReLU(inplace=True),
    )
    input_tensor = torch.randn(64, 300, input_size)  # BS, N, 2
    output, w1 = attention(input_tensor)

    input_2 = torch.randn(64, 250, input_size)  # BS, N, 2
    output_2, w2 = attention(input_2)
    output_3 = model(output_2.permute(0, 2, 1))
    print(
        f"Output 1: {output.shape} | Output 2: {output_2.shape} | Output 3: {output_3.shape}"
    )
    print(f"Weights 1: {w1.shape} | Weights 2: {w2.shape}")
