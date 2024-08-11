import torch
from torch import nn

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64,  64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(nn.Module):
    def __init__(self, arch: list[int | str], in_channels: int = 3, num_classes: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels: int = in_channels
        self.conv_layers: nn.Sequential = self.create_conv_layer(arch)
        self.flatten: nn.Flatten = nn.Flatten()

        self.adaptive_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7, 7))

        self.fully_connected: nn.Sequential = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fully_connected(x)

    def create_conv_layer(self, arch) -> nn.Sequential:
        layers = []
        in_channels = self.in_channels

        for x in arch:
            if isinstance(x, int):
                out_channels = x
                layers.append(nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))

                # It is not in VGG architecture but it improves the performance
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                in_channels = out_channels

            elif isinstance(x, str) and x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                raise Exception(f'Wrong architecture value: {x}')

        return nn.Sequential(*layers)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGNet(arch=VGG_types['VGG19'], in_channels=3, num_classes=1000).to(device)

    x = torch.randn(1, 3, 224, 224).to(device)
    print(model(x).shape)


if __name__ == '__main__':
    main()
