import os
import argparse
import torch
from functools import partial

class TypeArgs:
    parameter: str

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

def get_parser(name):
    """ make default formatted parser """
    # ArgumentDefaultsHelpFormatter 自动添加默认的值的信息到每一个帮助信息的参数中
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always        partial() 用来"冻结"一个函数的参数，并返回"冻结"参数后的新函数。
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--dataset', default='Market-1501', help='CUHK03 / Market-1501 / DukeMTMC-reID / VIPeR / caviar')
        parser.add_argument('--w_lr', type=float, default=1e-3, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=1e-4, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=1e-3,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=10, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--height', type=int, default=256, help='# of workers')
        parser.add_argument('--width', type=int, default=128, help='# of workers')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './dataset/'
        self.path = os.path.join('searchs', self.dataset)
        self.plot_path = os.path.join(self.path, 'plots')
        # print(self.gpus) ----------- 0
        self.gpus = parse_gpus(self.gpus)
        # print(self.gpus) ----------- [0]

class ClsConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Cls config")

        parser.add_argument('--PROJECT_NAME', type=str, default='myProject', help='project name')
        parser.add_argument('--LOG_DIR', type=str, default='./log', help='log directory')
        parser.add_argument('--OUTPUT_DIR', type=str, default='./output', help='saved model directory')

        parser.add_argument('--CFG_NAME', type=str, default='baseline', help='log directory')

        parser.add_argument('--LOG_PERIOD', type=int, default=10, help='iteration of displaying training log')
        parser.add_argument('--CHECKPOINT_PERIOD', type=int, default=5, help='saving model period')
        parser.add_argument('--EVAL_PERIOD', type=int, default=5, help='validation period')
        parser.add_argument('--MAX_EPOCHS', type=int, default=200, help='max training epochs')

        # pre-train
        parser.add_argument('--pretrain_dataset', type=str, default='DIV2K')
        parser.add_argument('--pretrain_datadir', type=str, default='../dataset/DIV2K/DIV2K_train_HR')
        parser.add_argument('--croped_path', type=str, default='../dataset/MyDIV2K/DIV2K_train_HR/')
        # parser.add_argument('--scale_list', type=list, default=[2])
        parser.add_argument('--crop_size', type=list, default=[256, 128])
        parser.add_argument('--image_num', type=int, default=800)

        # data
        parser.add_argument('--DATASET', default='Market-1501', help='CUHK03 / Market-1501 / DukeMTMC-reID / VIPeR / caviar')
        parser.add_argument('--DATA_DIR', type=str, default='../dataset/Market-1501-v15.09.15/', help='dataset path: Market-1501-v15.09.15 / DukeMTMC-reID / VIPeR / cuhk03_release')
        parser.add_argument('--DATALOADER_NUM_WORKERS', type=int, default=8, help='number of dataloader workers')
        parser.add_argument('--SAMPLER', type=str, default='triplet', help='batch sampler, option: triplet, softmax')
        parser.add_argument('--BATCH_SIZE', type=int, default=32, help='MxN, M: number of persons, N: number of images of per person')
        parser.add_argument('--NUM_IMG_PER_ID', type=int, default=4, help='N, number of images of per person')
        parser.add_argument('--SPLIT_ID', type=int, default=0, help="split index")
        # CUHK03-specific setting
        parser.add_argument('--CUHK03_CLASSIC_SPLIT',type=bool, default=True,
                            help="whether to use classic split by Li et al. CVPR'14 (default: False)")

        # model
        parser.add_argument('--INPUT_SIZE', type=list, default=[256, 128], help='HxW')
        parser.add_argument('--UPSCALE_FACTOR', type=list, default=[2, 3, 4])
        parser.add_argument('--LAST_STRIDE', type=int, default=1, help='the stride of the last layer of resnet50')
        parser.add_argument('--PRETRAIN_MODEL', type=str, default='./output/DIV2K/pretrained_layers-10.pth')
        parser.add_argument('--PRETRAIN_PATH', type=str, default='./models/resnet50-11ad3fa6.pth', help='pretrained weight path')
        parser.add_argument('--SR_MODEL_NAME', type=str, default='srnet', help='')
        parser.add_argument('--MODEL_NAME', type=str, default='resnet50', help='backbone name, option: resnet50')

        # loss
        parser.add_argument('--COS_LAYER', type=bool, default=False, help='')
        parser.add_argument('--LOSS_TYPE', type=str, default='triplet+softmax+center', help='option: triplet+softmax, softmax+center, triplet+softmax+center')
        parser.add_argument('--LOSS_LABELSMOOTH', type=str, default='on', help='using labelsmooth, option: on, off')
        parser.add_argument('--MARGIN', type=float, default=0.3, help='triplet loss margin')
        parser.add_argument('--HARD_FACTOR', type=float, default=0.0, help='harder example mining')

        # solver
        parser.add_argument('--CE_LOSS_WEIGHT', type=float, default=1.0, help='weight of softmax loss')
        parser.add_argument('--TRIPLET_LOSS_WEIGHT', type=float, default=1.0, help='weight of triplet loss')
        parser.add_argument('--CENTER_LOSS_WEIGHT', type=float, default=0.0005, help='weight of center loss')

        parser.add_argument('--OPTIMIZER', type=str, default='Adam', help='optimizer')
        parser.add_argument('--BASE_LR', type=float, default=0.00035, help='base learning rate')
        parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0005, help='')
        parser.add_argument('--BIAS_LR_FACTOR', type=float, default=1.0, help='')
        parser.add_argument('--WEIGHT_DECAY_BIAS', type=float, default=0.0005, help='')
        parser.add_argument('--MOMENTUM', type=float, default=0.9, help='')
        parser.add_argument('--CENTER_LR', type=float, default=0.5, help='learning rate for the weights of center loss')

        parser.add_argument('--STEPS', type=list, default=[40, 70, 130], help='')
        parser.add_argument('--GAMMA', type=float, default=0.1, help='decay factor of learning rate')
        parser.add_argument('--WARMUP_FACTOR', type=float, default=0.01, help='')
        parser.add_argument('--WARMUP_EPOCHS', type=int, default=10, help='warm up epochs')
        parser.add_argument('--WARMUP_METHOD', type=str, default='linear', help='option: linear, constant')

        # test
        parser.add_argument('--TEST_IMS_PER_BATCH', type=int, default=32, help='')
        parser.add_argument('--FEAT_NORM', type=str, default='yes', help='')
        parser.add_argument('--TEST_SR_WEIGHT', type=str, default='./output/CUHK03/sr_54.pth', help='')
        parser.add_argument('--TEST_CLS_WEIGHT', type=str, default='./output/CUHK03/cls_54.pth', help='')
        parser.add_argument('--TEST_LR_CLS_WEIGHT', type=str, default='./output/CUHK03/lr_cls_54.pth', help='')
        parser.add_argument('--TEST_HR_CLS_WEIGHT', type=str, default='./output/CUHK03/hr_cls_54.pth', help='')
        parser.add_argument('--TEST_METHOD', type=str, default='cosine', help='')
        parser.add_argument('--FLIP_FEATS', type=str, default='off', help='using fliped feature for testing, option: on, off')
        parser.add_argument('--RERANKING', type=bool, default=False, help='re-ranking')


        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))