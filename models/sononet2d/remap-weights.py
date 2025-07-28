import os
import torch
from collections import OrderedDict


new = ["_features.0.0.weight",
       "_features.0.1.weight", "_features.0.1.bias", "_features.0.1.running_mean", "_features.0.1.running_var",
       "_features.1.0.weight",
       "_features.1.1.weight", "_features.1.1.bias", "_features.1.1.running_mean", "_features.1.1.running_var",
       "_features.3.0.weight",
       "_features.3.1.weight", "_features.3.1.bias", "_features.3.1.running_mean", "_features.3.1.running_var",
       "_features.4.0.weight",
       "_features.4.1.weight", "_features.4.1.bias", "_features.4.1.running_mean", "_features.4.1.running_var",
       "_features.6.0.weight",
       "_features.6.1.weight", "_features.6.1.bias", "_features.6.1.running_mean", "_features.6.1.running_var",
       "_features.7.0.weight",
       "_features.7.1.weight", "_features.7.1.bias", "_features.7.1.running_mean", "_features.7.1.running_var",
       "_features.8.0.weight",
       "_features.8.1.weight", "_features.8.1.bias", "_features.8.1.running_mean", "_features.8.1.running_var",
       "_features.10.0.weight",
       "_features.10.1.weight", "_features.10.1.bias", "_features.10.1.running_mean", "_features.10.1.running_var",
       "_features.11.0.weight",
       "_features.11.1.weight", "_features.11.1.bias", "_features.11.1.running_mean", "_features.11.1.running_var",
       "_features.12.0.weight",
       "_features.12.1.weight", "_features.12.1.bias", "_features.12.1.running_mean", "_features.12.1.running_var",
       "_features.14.0.weight",
       "_features.14.1.weight", "_features.14.1.bias", "_features.14.1.running_mean", "_features.14.1.running_var",
       "_features.15.0.weight",
       "_features.15.1.weight", "_features.15.1.bias", "_features.15.1.running_mean", "_features.15.1.running_var",
       "_features.16.0.weight",
       "_features.16.1.weight", "_features.16.1.bias", "_features.16.1.running_mean", "_features.16.1.running_var",
       "_adaptation.0.weight",
       "_adaptation.1.weight", "_adaptation.1.bias", "_adaptation.1.running_mean", "_adaptation.1.running_var",
       "_adaptation.3.weight",
       "_adaptation.4.weight", "_adaptation.4.bias", "_adaptation.4.running_mean", "_adaptation.4.running_var"]

old = ["features.0.0.0.weight",
       "features.0.0.1.weight", "features.0.0.1.bias", "features.0.0.1.running_mean", "features.0.0.1.running_var",
       "features.0.1.0.weight",
       "features.0.1.1.weight", "features.0.1.1.bias", "features.0.1.1.running_mean", "features.0.1.1.running_var",
       "features.1.0.0.weight",
       "features.1.0.1.weight", "features.1.0.1.bias", "features.1.0.1.running_mean", "features.1.0.1.running_var",
       "features.1.1.0.weight",
       "features.1.1.1.weight", "features.1.1.1.bias", "features.1.1.1.running_mean", "features.1.1.1.running_var",
       "features.2.0.0.weight",
       "features.2.0.1.weight", "features.2.0.1.bias", "features.2.0.1.running_mean", "features.2.0.1.running_var",
       "features.2.1.0.weight",
       "features.2.1.1.weight", "features.2.1.1.bias", "features.2.1.1.running_mean", "features.2.1.1.running_var",
       "features.2.2.0.weight",
       "features.2.2.1.weight", "features.2.2.1.bias", "features.2.2.1.running_mean", "features.2.2.1.running_var",
       "features.3.0.0.weight",
       "features.3.0.1.weight", "features.3.0.1.bias", "features.3.0.1.running_mean", "features.3.0.1.running_var",
       "features.3.1.0.weight",
       "features.3.1.1.weight", "features.3.1.1.bias", "features.3.1.1.running_mean", "features.3.1.1.running_var",
       "features.3.2.0.weight",
       "features.3.2.1.weight", "features.3.2.1.bias", "features.3.2.1.running_mean", "features.3.2.1.running_var",
       "features.4.0.0.weight",
       "features.4.0.1.weight", "features.4.0.1.bias", "features.4.0.1.running_mean", "features.4.0.1.running_var",
       "features.4.1.0.weight",
       "features.4.1.1.weight", "features.4.1.1.bias", "features.4.1.1.running_mean", "features.4.1.1.running_var",
       "features.4.2.0.weight",
       "features.4.2.1.weight", "features.4.2.1.bias", "features.4.2.1.running_mean", "features.4.2.1.running_var",
       "adaption.0.weight",
       "adaption.1.weight", "adaption.1.bias", "adaption.1.running_mean", "adaption.1.running_var",
       "adaption.3.weight",
       "adaption.4.weight", "adaption.4.bias", "adaption.4.running_mean", "adaption.4.running_var"]

old_new_mapping = {old[i]: new[i] for i in range(len(old))}

base_weights_dir = '../../logs/weights4sononet2d/FetalDB'
for num_layers in [16, 32, 64]:
       pretraining_dir = os.path.join(base_weights_dir, f'SonoNet-{num_layers}')
       check_point_old = torch.load(os.path.join(pretraining_dir, 'ckpt_best_loss-old.pth'), map_location='cpu')
       check_point_new = OrderedDict([(old_new_mapping[k], v) for k, v in check_point_old.items()])
       torch.save(check_point_new, os.path.join(pretraining_dir, 'ckpt_best_loss.pth'))
