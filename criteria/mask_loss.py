import torch
from torch import nn
from configs.paths_config import model_paths
from models.faceparsing.faceparsing import unet,unet_my
from models.faceparsing.bisenet_model import BiSeNet
import numpy as np
import torch.nn.functional as F



def generate_label_plain(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        #pred = pred.reshape((1, 512, 512))
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())
                
    label_batch = np.array(label_batch)

    return label_batch

def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
                
    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch) 

    return label_batch
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.shape
    nt, ht, wt = target.shape

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        print('Loading FaceParsing Net')
        self.parsenet = unet()
        # self.parsenet = BiSeNet(n_classes=19)
        # if isinstance(self.parsenet, torch.nn.DataParallel):
        #     self.parsenet = self.parsenet.module
        self.parsenet.load_state_dict(torch.load(model_paths['parsenet']))
        self.parsenet.eval()
        # self.CE_weight=torch.FloatTensor([0.1,0.1,0.1,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,0.1,0.1,0.1,0.1,])
        self.CEloss=nn.CrossEntropyLoss()
        self.L1loss=nn.L1Loss()


    def forward(self, x, y):
        labels_predict = self.parsenet(x)

        # labels_predict=labels_predict[0]

        c_loss = self.CEloss(labels_predict, y.argmax(1).long())

        # y_plain = generate_label_plain(y,labels_predict.shape[-1])
        index=torch.max(labels_predict,dim=1,keepdim=True).indices
        mask=torch.zeros(labels_predict.shape).to('cuda:'+str(torch.cuda.current_device()))
        mask=mask.scatter_(1,index,1)
        img=mask.float()

                # temp_predict=labels_predict.view(-1,1)
        # temp_gt=y.long().view(-1,1)
        # c_loss=self.L1loss(temp_predict,temp_gt)

        return c_loss,img


class MaskFeatLoss(nn.Module):
    def __init__(self):
        super(MaskFeatLoss, self).__init__()
        print('Loading FaceParsing Net')
        self.parsenet = unet_my()
        print(self.parsenet)
        self.parsenet.load_state_dict(torch.load(model_paths['parsenet']))
        self.parsenet.eval()
        self.CEloss=nn.CrossEntropyLoss()
        self.L1loss=nn.L1Loss()
        # self.upsample= nn.Upsample(scale_factor=2, mode='nearest')
        self.face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))

    def forward(self, y_hat, y):

        y_hat=self.face_pool(y_hat)
        y=self.face_pool(y)
        feat_pred,img_hat = self.parsenet(y_hat)
        feat_gt,img_gt = self.parsenet(y)
        return feat_pred,feat_gt,img_hat,img_gt




