import os
from torch import Tensor
from typing import List
import torch
import json
import os.path as osp
import errno


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

 # output: (bs, num_class) 是16行1367列,  target: (bs, 1)， topk=(1,5)
def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)  # 5
    batch_size = target.size(0)

    # maxk=5，表示dim=1按行取值
    # output的值是精度，选top5是选这一行精度最大的五个对应的列，也就是属于哪一类
    # pred是(bs,5) 值为类别号，0，1，...,9
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()     # 转置，pred:(5, bs)
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    # pred和target对应位置值相等返回1，不等返回0
    # target原来是64行1列，值为类别；target.view(1, -1)把target拉成一行，expand_as(pred)又把target变成5行64列
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:  # k=1 和 k=5
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # zouxing: view -> .contiguous().view
        res.append(correct_k.mul_(1.0 / batch_size))
    # res里是两个值，一个是top1的概率，一个是top5的概率
    return res

def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def unnormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    _assert_image_tensor(tensor)

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.mul_(std).add_(mean)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))