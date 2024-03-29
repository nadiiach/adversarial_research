import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Softplus()]
            else:
                layers += [conv2d, nn.Softplus()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
          512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, w=None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False

    if w is None or len(w) == 0:
        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    else:
        _w = cfgs[cfg]
        nex = 0
        for i, item in enumerate(_w):
            if isinstance(item, int):
                _w[i] = w[nex]
                nex += 1
        assert nex == len(w)
        w = _w
        model = VGG(make_layers(w, batch_norm=batch_norm), **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, w=None, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, w=w, **kwargs)


def vgg11(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)