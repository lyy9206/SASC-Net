""" CNN for architecture search """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
from models.searched_cell import SearchedCell
from models import genotypes
from torch.nn.parallel._functions import Broadcast
import logging
from config import ClsConfig

cfg = ClsConfig()

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies

def pixelshuffle_block(in_channels, out_channels, upscale_factor):
    upsample_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
    if upscale_factor == 4:
        i = 2
        upscale_factor = 2
    else:
        i = 1
    for _ in range(i):
        upsample_layers += [
            nn.Conv2d(out_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor=upscale_factor),
        ]
    return nn.Sequential(*upsample_layers)

class SearchCNN(nn.Module):
    """ Search CNN model """
    # SearchCNN(3, 64, 1467, 8, 4)
    def __init__(self, input_channels, init_channels, num_layers, num_nodes=4):
        """
        Args:
            input_channels: # of input channels   输入通道数
            init_channels: # of starting model channels 初始通道数
            num_classes: # of classes
            num_layers: # of layers     1个network包括8个cell
            num_nodes: # of intermediate nodes in Cell    默认为4个中间节点
        """
        super().__init__()
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.num_layers = num_layers

        output_channels = init_channels     # 当前Sequential模块的输出通道数
        self.stem = nn.Sequential(
            # nn.Conv2d(input_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=output_channels)
        )

        # for the first cell, stem is used for both s0 and s1
        # ppre_channels and pre_channels is output channel size, but output_channels is input channel size.
        # 48, 48, 16
        ppre_channels, pre_channels, channels = output_channels, output_channels, init_channels

        self.cells = nn.ModuleList()
        for i in range(num_layers):

            # 构建cell  每个cell的input nodes是前前cell和前一个cell的输出
            '''
            layers = 8
            cell[0]: cell = SearchCell(4, 16, 16, 16)
            cell[1]: cell = SearchCell(4, 16, 64, 16)
            cell[2]: cell = SearchCell(4, 64, 64, 16)
            cell[3]: cell = SearchCell(4, 64, 64, 16)
            cell[4]: cell = SearchCell(4, 64, 64, 16)
            cell[5]: cell = SearchCell(4, 64, 64, 16)
            cell[6]: cell = SearchCell(4, 64, 64, 16)
            cell[7]: cell = SearchCell(4, 64, 64, 16)
            '''
            cell = SearchedCell(num_nodes, ppre_channels, pre_channels, channels)
            # print("cell[" + str(i)+ "]: cell = SearchCell({}, {}, {}, {})".format(num_nodes, ppre_channels, pre_channels, channels))
            self.cells.append(cell)
            curr_output_channels = channels * num_nodes
            # pre_channels = 4 * curr_input_channels 是因为每个cell的输出是4个中间节点concat的，这个concat是在通道这个维度，所以输出的通道数变为原来的4倍
            ppre_channels, pre_channels = pre_channels, curr_output_channels


        self.upsampler = pixelshuffle_block(pre_channels, channels, upscale_factor=2)
        # Final output block
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, weights_normal):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_normal
            s0, s1 = s1, cell(s0, s1)

        out = self.upsampler(s1)
        sr = self.final_conv(out)
        return sr


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    # (3, 64, 1467, 8, L1Loss(), 4, [0])
    def __init__(self, input_channels, init_channels, num_layers, criterion, num_nodes=4, device_ids=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        num_ops = len(genotypes.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()

        for i in range(num_nodes):
            # torch.randn(a, b)返回一个axb的张量，该张量由均值为0、方差为1的正态分布中的随机数填充。
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, num_ops)))

        '''
        ParameterList(
            (0): Parameter containing: [torch.float32 of size 2x8]
            (1): Parameter containing: [torch.float32 of size 3x8]
            (2): Parameter containing: [torch.float32 of size 4x8]
            (3): Parameter containing: [torch.float32 of size 5x8]
        )
        '''

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        '''
        self._alphas[0]:
        ('alpha_normal.0', Parameter containing:
        tensor([[-0.0010,  0.0009, -0.0013, -0.0011, -0.0012,  0.0012, -0.0011, -0.0001],
                [-0.0009,  0.0003, -0.0006, -0.0002,  0.0010, -0.0008, -0.0002, -0.0004]],
               requires_grad=True))
        '''

        # SearchCNN(3, 64, 1467, 8, 4)
        self.net = SearchCNN(input_channels, init_channels, num_layers, num_nodes)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        sr = self.forward(X)
        return self.criterion(sr, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = genotypes.parse(self.alpha_normal, k=2)
        concat = range(2, 2+self.num_nodes) # concat all intermediate nodes

        return genotypes.Genotype(normal=gene_normal, normal_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
