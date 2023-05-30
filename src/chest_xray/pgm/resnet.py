import torch
import torchvision
import torch.nn as nn


class ResNets_custom(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        name="resnet18",
        pretrained=False,
        context_dim=0,
    ):
        super(ResNets_custom, self).__init__()
        # bring resnet
        if name == "resnet18":
            resnet = torchvision.models.resnet18(weights=pretrained)
        elif name == "resnet34":
            resnet = torchvision.models.resnet34(weights=pretrained)
        elif name == "resnet50":
            resnet = torchvision.models.resnet50(weights=pretrained)
        else:
            NotImplementedError
        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Our case
        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Extract feature extraction layers
        modules = list(resnet.children())[:-1]
        self.model_fe = nn.Sequential(*modules)
        # Set output layers
        num_ftrs = resnet.fc.in_features
        # Custom output layer
        self.fc = nn.Linear(num_ftrs + context_dim, out_channels)

    def forward(self, x, y=None):
        x = self.model_fe(x).squeeze(-1).squeeze(-1)
        if y is not None:
            x = torch.cat([x, y], dim=1)
        return self.fc(x)


if __name__ == "__main__":
    my_resnet = ResNets_custom(in_channels=1, out_channels=1, context_dim=1)
    input = torch.randn((16, 1, 244, 244))
    y = torch.randn((16, 1))
    output = my_resnet(input, y)
    print(output.shape)
