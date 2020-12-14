import torch


class ClassificationModel(torch.nn.Module):
    def __init__(self, backbone, embeddings_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeddings_dim, embeddings_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embeddings_dim, 1)
        )

    def forward(self, x):
        x = self.backbone(x)[0]

        # backbone output shape: N, L, C (number of samples, number_of_channels, length)
        # pooling input shape: N, C, L
        x = x.permute(0, 2, 1)
        x = self.pooling(x)

        # pooling output shape: N, C, 1
        x = x.squeeze(2)
        x = self.head(x)
        return x
