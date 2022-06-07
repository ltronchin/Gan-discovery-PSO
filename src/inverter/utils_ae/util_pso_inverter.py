import torch.nn as nn

class Identity(nn.Module):  # utils to skip a layer of a pretrained model
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def change_classifier_n_class(model, n_class=2):
    model.fc = nn.Linear(in_features=2048, out_features=n_class)
    return model