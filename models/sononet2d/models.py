"""
PyTorch implementation for a 2D version of the SonoNet model proposed in:
Baumgartner et al. "SonoNet: real-time detection and localisation of fetal standard scan planes in freehand ultrasound."
IEEE transactions on medical imaging 36.11 (2017): 2204-2215.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SonoNet2D(nn.Module):
    """
    PyTorch implementation for a 2D version of the SonoNet model.

    Args:
        in_channels (int, optional): Number of input channels in the data. Default is 1.
        hid_features (int, optional): Number of features in the first hidden layer that defines network arhcitecture.
            In fact, features in all subsequent layers are set accordingly by using multiples of this value,
            (i.e. x2, x4 and x8). Default is 16.
        out_labels (int, optional): Number of output labels (length of output vector after adaptation).
            Default is 7. Ignored if features_only=True.
        features_only (bool, optional): If True, only feature layers are initialized and the forward method
            returns the features. Default is False.
         train_classifier_only: (bool, optionale): If True, only the classifier layers are trainable. Default is False.
        trainable_layers (int, optional): Number of trainable layers in the feature extractor.
            If 0, all layers are trainable. Default is 0. 

    Attributes:
        _features (torch.nn.Sequential): Feature extraction CNN
        _adaptation (torch.nn.Sequential): Adaption layers for classification


    """

    def __init__(self, in_channels: int = 1, hid_features: int = 32, out_labels: int = 7,
                 features_only: bool = False, init: str = 'uniform', train_classifier_only: bool = False, trainable_layers: int = 0):
        super().__init__()

        self.in_channels = in_channels
        self.hid_features = hid_features  # number of filters in the first layer (then x2, x4 and x8)
        self.out_labels = out_labels
        self.features_only = features_only
        self.dropout = nn.Dropout(p=0.25)

        self._features = self._make_feature_layers()
        if not features_only:
            self._adaptation = self._make_adaptation_layers()

        assert init in ['normal', 'uniform'], 'The init parameter may only be one between "normal" and "uniform"'
        full_init = self._initialize_normal if init == 'normal' else self._initialize_uniform
        self.apply(full_init)
        if not features_only:
            last_init = nn.init.xavier_normal_ if init == 'normal' else nn.init.xavier_uniform_
            last_init(self._adaptation[3].weight)  # last conv layer has no ReLu, hence Kaiming init is not suitable

            #initially all layers are trainable
            for param in self._features.parameters():
                param.requires_grad = True
            for param in self._adaptation.parameters():
                param.requires_grad = True


            if train_classifier_only:
                for param in self._features.parameters():
                    param.requires_grad = False
                for param in self._adaptation.parameters():
                    param.requires_grad = True
         
            num_blocks = len(list(self._features.children()))
        
            if trainable_layers > 0: 
                for i, layer in enumerate(list(self._features.children())[: (num_blocks-trainable_layers - 1)]):
                    for param in layer.parameters():
                        param.requires_grad = False


    @staticmethod
    def _initialize_normal(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()

    @staticmethod
    def _initialize_uniform(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()

    def forward(self, x):
        x = self._features(x)
        if not self.features_only:
            x = self._adaptation(x)
            try:
                (batch, channel, h, w) = x.size()           #(batch, channel, t, h, w) = x.size()  
            except ValueError:
                (channel, h, w) = x.size()
                batch = 1

            x = self.dropout(x) 
            x = F.avg_pool2d(x, kernel_size=(h, w)).view(batch, channel)  # in=(N,C,H,W) & out=(N,C)
            #x = F.softmax(x, dim=1)  # out=(N,C)
        return x

    @staticmethod
    def _conv_layer(in_channels, out_channels):
        layer = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3), padding="same", bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-4),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layer)

    def _make_feature_layers(self):
        layers = [
            # Convolution stack 1
            self._conv_layer(self.in_channels, self.hid_features),
            self._conv_layer(self.hid_features, self.hid_features),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Convolution stack 2
            self._conv_layer(self.hid_features, self.hid_features * 2),
            self._conv_layer(self.hid_features * 2, self.hid_features * 2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Convolution stack 3
            self._conv_layer(self.hid_features * 2, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Convolution stack 4
            self._conv_layer(self.hid_features * 4, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Convolution stack 5
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
        ]
        return nn.Sequential(*layers)

    def _make_adaptation_layers(self):
        layers = [
            # Adaptation layer 1
            nn.Conv2d(self.hid_features * 8, self.hid_features * 4,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.hid_features * 4),
            nn.ReLU(inplace=True),
            # Adaptation layer 2
            nn.Conv2d(self.hid_features * 4, self.out_labels,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.out_labels)
        ]
        return nn.Sequential(*layers)
    







