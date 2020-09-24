import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        L = []
        layers = [32, 64]
        c = 3
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, 3, padding=1, stride=2))
            L.append(torch.nn.ReLU())
            c = l
        # L.append(torch.nn.Conv2d(c, c, kernel_size=1))
        self.layers = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        res = self.layers(x).mean(dim=[2, 3])
        res = self.classifier(res)
        return res


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn1.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn1.th'), map_location='cpu'))
    return r
