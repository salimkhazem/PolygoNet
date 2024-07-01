import torch
import torch.nn as nn
import numpy as np
from multiheadattention import SplineMultiHead


class DeepNetwork(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=128,
        d_k=2,
        num_heads=2,
        out_size=128,
        num_classes=5,
        use_positional_encoding=True,
        requires_grad=False,
    ):
        super().__init__()
        self.out_size = out_size
        self.num_classes = num_classes
        self.attention = SplineMultiHead(
            input_size, num_heads, d_k=d_k, out_size=out_size, batch_first=True
        )
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding = self.positional_encoding_2d(512, 2)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.pos_encoding = nn.Parameter(
                self.pos_encoding, requires_grad=self.requires_grad
            )
        dummy_tensor = torch.randn(64, 256, 2)
        out, _ = self.attention(dummy_tensor)
        out_dim = out.shape[1]
        self.conv = nn.Sequential(
            nn.Conv1d(out_dim, 64, 3, padding_mode="circular", padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding_mode="circular", padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, padding_mode="circular", padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 3, padding_mode="circular", padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 3, padding_mode="circular", padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Linear(1024 * num_heads * d_k, self.num_classes)
        )

    @staticmethod
    def positional_encoding_2d(num_points, embedding_dim=2):
        position = torch.arange(0, num_points).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-np.log(10000.0) / embedding_dim)
        )
        pos_encoding = torch.zeros((num_points, embedding_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, x):
        bs, n, _ = x.size()
        if self.use_positional_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding[:n, :].to(x.device).unsqueeze(0)
        out1, _ = self.attention(x)
        out_cnn = self.conv(out1.view(bs, self.out_size, -1))
        output = self.output(out_cnn.view(bs, -1))
        return output


if __name__ == "__main__":
    model = DeepNetwork(requires_grad=True).to(torch.device("cuda"))
    x1 = torch.randn(64, 300, 2).to(torch.device("cuda"))
    x2 = torch.randn(64, 256, 2).to(torch.device("cuda"))
    out_1 = model(x1)
    out_2 = model(x2)
    print(out_1.shape)
    print(out_2.shape)
