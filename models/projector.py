import torch


class Projector(torch.nn.Module):
    def __init__(self, shape=()):
        super(Projector, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = torch.nn.Sequential(
            torch.nn.Linear(shape[0], shape[1]),
            torch.nn.BatchNorm1d(shape[1]),
            # torch.nn.LayerNorm(shape[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(shape[1], shape[2])
        )

    def forward(self, x):
        return self.main(x)