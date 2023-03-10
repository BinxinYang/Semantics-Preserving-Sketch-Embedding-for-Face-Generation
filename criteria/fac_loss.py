import torch
from torch import nn
from configs.paths_config import model_paths
from models.faceparsing.faceparsing import unet,unet_my
from models.faceparsing.bisenet_model import BiSeNet
import numpy as np
import torch.nn.functional as F


import torch
from torch import nn
from torch import optim
from criteria.layers import SSE, Slim, conv_dw_separable, conv_2d, init_weights

# hacky convenience function to turn a list of values into a dictionary
def zip_params(filter_count_values, initial_conv, num_classes, depth_multiplier):
  return locals()
        
# Network Definition


class SlimNet(nn.Module):
  def __init__(self, filter_count_values=[16,32,48,64], initial_conv=[96,7,2], num_classes=40, depth_multiplier=1):
    super().__init__()
    
    #Store architecture hyper_params for model persistence / loading
    self.hyper_params = zip_params(filter_count_values, initial_conv, num_classes, depth_multiplier)

    self.conv1 = conv_2d(3, initial_conv[0], initial_conv[1], stride=initial_conv[2])
    self.max_pool1 = nn.MaxPool2d(3,2)
    
    self.slim1 = Slim(initial_conv[0], filter_count_values[0])
    self.max_pool2 = nn.MaxPool2d(3,2)
    
    self.slim2 = Slim(filter_count_values[0] * 3, filter_count_values[1])
    self.max_pool3 = nn.MaxPool2d(3,2)

    self.slim3 = Slim(filter_count_values[1] * 3, filter_count_values[2])
    self.max_pool4 = nn.MaxPool2d(3,2)

    self.slim4 = Slim(filter_count_values[2] * 3, filter_count_values[3])
    self.max_pool5 = nn.MaxPool2d(3,2)
    
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(filter_count_values[3] * 3, num_classes)
    
    [module.apply(init_weights) for module in [self.conv1, self.fc]]
        
  def forward(self, x):
    result=[]
    out = self.max_pool1(self.conv1(x))
    result.append(out)
    out = self.max_pool2(self.slim1(out))
    result.append(out)
    out = self.max_pool3(self.slim2(out))
    result.append(out)
    out = self.max_pool4(self.slim3(out))
    result.append(out)
    out = self.global_pool(self.max_pool5(self.slim4(out)))
    
    return self.fc(torch.flatten(out, 1)),result

  def save(self, path, optimizer=None, scheduler=None):
    checkpoint_dictionary = {'hyper_params': self.hyper_params, 'model_state_dict': self.state_dict()}                                                
    if optimizer is not None:
      checkpoint_dictionary['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
      checkpoint_dictionary['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint_dictionary, path)

  @staticmethod
  def load_pretrained(path, optimizer=None, scheduler=None):
    if torch.cuda.is_available():
      checkpoint = torch.load(path)
    else:
      checkpoint = torch.load(path, map_location='cpu')
    hyper_params = checkpoint['hyper_params']
    model = SlimNet(**hyper_params)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
      scheduler.load_state_dict[checkpoint['scheduler_state_dict']]

    if any([optimizer, scheduler]):
      return model, optimizer, scheduler
    else:
      return model




class FACloss(nn.Module):
    def __init__(self):
        super(FACloss, self).__init__()
        print('Loading FAC Net')
        self.parsenet = SlimNet.load_pretrained('/gpfs/home/eeid/tennyson/code/pytorch-slim-cnn-master/checkpoints/model_20.pth').to('cuda:'+str(torch.cuda.current_device()))
        self.parsenet.eval()
        self.criterion = nn.BCEWithLogitsLoss()
        self.L1loss=nn.L1Loss()

    def forward(self, y_hat, y):
        Attr_hat,feat_hat=self.parsenet(y_hat)
        Attr,feat_gt=self.parsenet(y)
        loss=0
        for j in range(4):
            loss += self.L1loss(feat_hat[j], feat_gt[j].detach())
        # Attr=torch.sigmoid(self.parsenet(y))
        # Attr=(Attr>0.5)
        # print(Attr,Attr_hat)
        # loss=self.L1loss(Attr_hat[:,16],Attr[:,16])
        # print(Attr_hat[:,16],Attr[:,16],loss)
        return loss




