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
        super(DeepNetwork, self).__init__()
        self.out_size = out_size
        self.num_classes = num_classes
        self.attention = SplineMultiHead(
            input_size, num_heads, d_k=d_k, out_size=out_size, batch_first=True
        )
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding = self.positional_encoding_2d(8048, 2)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.pos_encoding = nn.Parameter(
                self.pos_encoding, requires_grad=self.requires_grad
            )
        dummy_tensor = torch.randn(64, 256, 2)
        out, _ = self.attention(dummy_tensor)
        out_dim = out.shape[1]
        self.encoder = nn.Sequential(
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

        self.decoder_mask = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16), 
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8), 
            nn.Conv2d(8, 1, 3, padding=1)
        ) 
        self.decoder_coord = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(512, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
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
        BS, N, _ = x.size()
        if self.use_positional_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding[:N, :].to(x.device).unsqueeze(0)
        out1, out2 = self.attention(x)
        out1 = out1.view(BS, self.out_size, -1) 
        out_cnn = self.encoder(out1)
        out_coord = out_cnn
        BS, C, L = out_cnn.size()
        H = W = int(np.ceil(np.sqrt(L)))
        if H * W > L:
            # Pad with zeros to match H * W
            padding = torch.zeros(BS, C, H * W - L).to(out_cnn.device)
            out_cnn = torch.cat([out_cnn, padding], dim=2)
        out_cnn = out_cnn.view(BS, C, H, W)  # x shape: (BS, 1024, H, W)
        out_decoder_coord = self.decoder_coord(out_coord).view(BS, -1, 2)
        out_decoder_mask = self.decoder_mask(out_cnn)

        if out_decoder_coord.size(1) > N:
            out_decoder_coord = out_decoder_coord[:, :N, :]
        elif out_decoder_coord.size(1) < N:
            padding = torch.zeros(BS, N - out_decoder_coord.size(1), 2).to(out_decoder_coord.device)
            out_decoder_coord = torch.cat([out_decoder_coord, padding], dim=1)
        return out_decoder_mask, out_decoder_coord


if __name__ == "__main__":
    model = DeepNetwork(requires_grad=True).to(torch.device("cuda"))
    x1 = torch.randn(64, 300, 2).to(torch.device("cuda"))
    x2 = torch.randn(64, 256, 2).to(torch.device("cuda"))
    out_1_mask, out_1_coord = model(x1)
    out_2_mask, out_2_coord = model(x2)
    print(f"Input: {x1.shape} ---> Coord: ({out_1_coord.shape}) | Mask: ({out_1_mask.shape})")
    print(f"Input: {x2.shape} ---> Coord: ({out_2_coord.shape}) | Mask: ({out_2_mask.shape})")
    print("---------------------")
