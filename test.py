import logging
import os
import torch
from torch import nn
from torch.backends import cudnn
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from load_dataset import make_dataloader
from models import make_model
from config import ClsConfig, SearchConfig
from models.search_cnn import SearchCNNController
from utils.logger import setup_logger
from utils.metrics import R1_mAP

from eval_metrics import evaluate

config = SearchConfig()
cfg = ClsConfig()
log_dir = cfg.LOG_DIR
logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir)

def test():
    cudnn.benchmark = True

    device = torch.device("cuda")

    train_loader, query_loader, gallery_loader, num_query, num_classes = make_dataloader(cfg, mode='test')

    loss = nn.L1Loss().to(device)
    model = SearchCNNController(config.input_channels, config.init_channels, config.layers,
                                    loss, device_ids=config.gpus)
    model.load_state_dict(torch.load(cfg.TEST_SR_WEIGHT))

    cls_model = make_model(cfg, num_classes)
    cls_model.load_param(cfg.TEST_CLS_WEIGHT)
    lr_cls_model = make_model(cfg, num_classes)
    lr_cls_model.load_param(cfg.TEST_LR_CLS_WEIGHT)

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM,
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()

    resize_ = T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC)

    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
            cls_model = nn.DataParallel(cls_model)
        model.to(device)
        cls_model.to(device)
        lr_cls_model.to(device)

    model.eval()
    cls_model.eval()
    lr_cls_model.eval()

    img_path_list = []
    for n_iter, (hr_imgs, imgs, pids, camids, imgpath) in enumerate(query_loader):
        with torch.no_grad():
            # img, img2, img3, img4 = img.to(device), img2.to(device), img3.to(device), img4.to(device)
            imgs = imgs.to(device)
            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(imgs.size(0), 4096).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(imgs.size(3) - 1, -1, -1).long().cuda()
                        imgs = imgs.index_select(3, inv_idx)
                    sr_img = model(imgs)
                    lr_img = resize_(imgs)
                    lr_features = lr_cls_model(lr_img)
                    f = cls_model(sr_img)
                    f = torch.cat((lr_features, f), dim=1)
                    feat = feat + f
            else:
                sr_img = model(imgs)
                lr_img = resize_(imgs)
                lr_features = lr_cls_model(lr_img)
                feat = cls_model(sr_img)
                feat = torch.cat((lr_features, feat), dim=1)

            evaluator.update((feat, pids, camids))
            img_path_list.extend(imgpath)

    for n_iter, (hr_imgs, imgs, pids, camids, imgpath) in enumerate(gallery_loader):
        with torch.no_grad():
            # img, img2, img3, img4 = img.to(device), img2.to(device), img3.to(device), img4.to(device)
            imgs = imgs.to(device)
            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(imgs.size(0), 4096).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(imgs.size(3) - 1, -1, -1).long().cuda()
                        imgs = imgs.index_select(3, inv_idx)
                    sr_img = model(imgs)
                    lr_img = resize_(imgs)
                    lr_features = lr_cls_model(lr_img)
                    f = cls_model(sr_img)
                    f = torch.cat((lr_features, f), dim=1)
                    feat = feat + f
            else:
                sr_img = model(imgs)
                lr_img = resize_(imgs)
                lr_features = lr_cls_model(lr_img)
                feat = cls_model(sr_img)
                feat = torch.cat((lr_features, feat), dim=1)

            evaluator.update((feat, pids, camids))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, q_pids, g_pids, q_camids, g_camids = evaluator.compute()

    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=True)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

if __name__ == '__main__':
    test()