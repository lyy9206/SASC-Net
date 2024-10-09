import torch
import torch.nn as nn

from load_dataset.preprocessing import RandomErasing
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss import ArcFace

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.LAST_STRIDE
        model_path = cfg.PRETRAIN_PATH
        self.cos_layer = cfg.COS_LAYER
        model_name = cfg.MODEL_NAME
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        else:
            print('unsupported backbone! only support resnet50, but got {}'.format(model_name))

        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # nn.Dropout(p=0.5)
        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bn2 = nn.BatchNorm2d(256, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(512, eps=1e-5)
        self.bn_ = nn.BatchNorm2d(2048, eps=1e-5)
        self.tx2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.tx3 = nn.Conv2d(512, 512, 1, 1, 0)
        self.tx5 = nn.Conv2d(2048, 2048, 1, 1, 0)
        self.tx5x = nn.Conv2d(2048, 512, 1, 1, 0)

    def forward(self, x, label=None, is_hr=False):  # label is unused if self.cos_layer == 'no'
        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).unsqueeze(0).cuda()
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
        x = (x - mean) / std
        x5, x3, x2 = self.base(x)
        global_feat = nn.functional.avg_pool2d(x5, x5.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)

        if is_hr == True:
            # tx2 = self.tx2(self.bn2(x2.detach()))
            tx3 = self.tx3(self.bn3(x3.detach()))
            tx5 = self.tx5(self.bn_(x5.detach()))
            tx5x = self.tx5x(self.bn_(x5.detach()))
        else:
            # tx2 = self.tx2(self.bn2(x2))
            tx3 = self.tx3(self.bn3(x3))
            tx5 = self.tx5(self.bn_(x5))
            tx5x = self.tx5x(self.bn_(x5))

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat, tx3, tx5, tx5x  # global feature for triplet loss
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
