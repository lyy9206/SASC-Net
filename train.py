import logging
import os
import random

import numpy as np
import scipy
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
import time

from torchvision.transforms import InterpolationMode

import eval_metrics
from eval_metrics import evaluate
from load_dataset import make_dataloader
from loss import make_loss
from models import make_model
from models.search_cnn import SearchCNNController
from config import SearchConfig, ClsConfig
from torch import nn
from models.architect import Architect
from solver import make_optimizer, WarmupMultiStepLR
from utils import tools
from utils.logger import setup_logger
from utils.metrics import R1_mAP
from utils.visualize import plot
import torchvision.transforms as T

config = SearchConfig()
cfg = ClsConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))

if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.LOG_DIR)
logger.info("Running with config:\n{}".format(cfg.CFG_NAME))

def train():
    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    # 参数相当于给定标志，总是按一定的顺序生成随机数，并不是真正意义上的随机
    np.random.seed(config.seed)
    # 为 CPU 设置种子，生成随机数：
    torch.manual_seed(config.seed)
    # 为特定 GPU 设置种子，生成随机数：
    # torch.cuda.manual_seed(config.seed)
    # 为所有 GPU 设置种子，生成随机数：
    torch.cuda.manual_seed_all(config.seed)
    cudnn.benchmark = True

    train_loader, train_val_loader, num_query, num_classes = make_dataloader(cfg, mode='train_sr')

    # 设置损失函数
    l1loss = nn.L1Loss().to(device)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    def SRD_loss(PHR, HR):
        batch = PHR.size(0)
        # print(PHR.size())
        #     loss = Variable(torch.zeros(1).cuda())
        SP = PHR.view(PHR.size(0), PHR.size(1), -1)
        # print(SP.size())
        SP_norm = torch.norm(SP, p=2, dim=1)
        # print(SP_norm.size())
        sp = SP / (SP_norm.unsqueeze(1) + 1e-5)
        # print(sp.size())
        Sp = torch.matmul(sp.transpose(1, 2), sp)
        # print(Sp.shape)

        SH = HR.view(HR.size(0), HR.size(1), -1)
        SH_norm = torch.norm(SH, p=2, dim=1)
        sh = SH / (SH_norm.unsqueeze(1) + 1e-5)
        Sh = torch.matmul(sh.transpose(1, 2), sh)

        loss = torch.norm(Sp - Sh, p=1, dim=1).sum() / batch

        return loss

    def L1_loss(PHR5, HR5, PHR3, HR3):
        batch = PHR5.size(0)
        # print(PHR5.size())
        #     loss = Variable(torch.zeros(1).cuda())
        SP = PHR5.view(PHR5.size(0), PHR5.size(1), -1)
        # print(SP.size())
        SP_norm = torch.norm(SP, p=2, dim=1)
        # print(SP_norm.size())
        sp = SP / (SP_norm.unsqueeze(1) + 1e-5)
        # print(sp.size())
        SP2 = PHR3.view(PHR3.size(0), PHR3.size(1), -1)
        # print(SP2.size())
        SP2_norm = torch.norm(SP2, p=2, dim=1)
        # print(SP2_norm.size())
        sp2 = SP2 / (SP2_norm.unsqueeze(1) + 1e-5)
        # print(sp2.size())

        Sp = torch.matmul(sp.transpose(1, 2), sp2)
        # print(Sp.shape)

        SH = HR5.view(HR5.size(0), HR5.size(1), -1)
        SH_norm = torch.norm(SH, p=2, dim=1)
        sh = SH / (SH_norm.unsqueeze(1) + 1e-5)
        SH2 = HR3.view(HR3.size(0), HR3.size(1), -1)
        SH2_norm = torch.norm(SH2, p=2, dim=1)
        sh2 = SH2 / (SH2_norm.unsqueeze(1) + 1e-5)

        Sh = torch.matmul(sh.transpose(1, 2), sh2)

        loss = torch.norm(Sp - Sh, p=1, dim=1).sum() / batch

        return loss

    # 构建网络，即包括8个cell的supernet
    # SearchCNNController(3, 64, 1467, 8, L1Loss(), [0])
    model = SearchCNNController(config.input_channels, config.init_channels, config.layers,
                                l1loss, device_ids=config.gpus)
    model.load_state_dict(torch.load(cfg.PRETRAIN_MODEL))
    # model = T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC)

    cls_model = make_model(cfg, num_classes)
    lr_cls_model = make_model(cfg, num_classes)
    hr_cls_model = make_model(cfg, num_classes)
    # cls_model.load_param(cfg.TEST_CLS_WEIGHT)
    # lr_cls_model.load_param(cfg.TEST_LR_CLS_WEIGHT)
    # hr_cls_model.load_param(cfg.TEST_HR_CLS_WEIGHT)

    resize_ = T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC)

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            cls_model = nn.DataParallel(cls_model)
        model.to(device)
        cls_model.to(device)
        lr_cls_model.to(device)
        hr_cls_model.to(device)

    # weights optimizer 用于网络参数 w 的优化器
    # w_lr 初始值是0.01，使用的余弦退火调度更新学习率，每个epoch的学习率都不一样
    # w_momentum = 0.9 常用参数
    # w_weight_decay = 1e-3  正则化参数
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)


    # alphas optimizer 用于结构参数 α 的优化器
    # alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.9, 0.999),
    #                                weight_decay=config.alpha_weight_decay)

    optimizer, optimizer_center = make_optimizer(cfg, cls_model, center_criterion)
    lr_optimizer, lr_optimizer_center = make_optimizer(cfg, lr_cls_model, center_criterion)
    hr_optimizer, hr_optimizer_center = make_optimizer(cfg, hr_cls_model, center_criterion)

    # 学习率更新参数，每次迭代调整不同的学习率   使用余弦退火调度设置各组参数组的学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, T_max=config.epochs, eta_min=config.w_lr_min)

    cls_scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)
    lr_cls_scheduler = WarmupMultiStepLR(lr_optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)
    hr_cls_scheduler = WarmupMultiStepLR(hr_optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)

    # 创建用于更新 α 的architect
    # architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_rank1 = 0.
    # 经历 epochs 次搜索后结束
    start_time = time.time()
    for epoch in range(cfg.MAX_EPOCHS):

        log_period = cfg.LOG_PERIOD
        checkpoint_period = cfg.CHECKPOINT_PERIOD

        # training 先更新alpha，再更新w
        top1 = tools.AverageMeter()  # 保存前 1 预测正确的概率
        top5 = tools.AverageMeter()  # 保存前 5 预测正确的概率
        sr_losses = tools.AverageMeter()  # 保存loss值
        cls_losses = tools.AverageMeter()  # 保存loss值
        hr_cls_losses = tools.AverageMeter()  # 保存loss值
        lr_cls_losses = tools.AverageMeter()
        feat_losses = tools.AverageMeter()  # 保存loss值
        srd_losses = tools.AverageMeter()  # 保存loss值
        losses = tools.AverageMeter()  # 保存loss值

        cur_step = epoch * len(train_loader)

        model.train()
        cls_model.train()
        lr_cls_model.train()
        hr_cls_model.train()
        # 每个step取出一个batch，batchsize是64
        # enumerate(): 可以同时获得索引和值
        if epoch < config.epochs:
            for step, ((trn_hr_imgs, trn_imgs, trn_targets), (val_hr_imgs, val_imgs, _)) in enumerate(zip(train_loader, train_val_loader)):
            # for step, (img, vid) in enumerate(train_loader):
                trn_hr_imgs, trn_imgs, trn_targets = trn_hr_imgs.to(device, non_blocking=True), trn_imgs.to(device, non_blocking=True), trn_targets.to(device, non_blocking=True)
                # 用于架构参数alpha 更新的一个batch, 使用iter(dataloader)返回的是一个迭代器, 然后可以使用next访问
                val_hr_imgs, val_imgs = val_hr_imgs.to(device, non_blocking=True), val_imgs.to(device, non_blocking=True)

                N = trn_hr_imgs.size(0)  # N = 64

                # phase 2. architect step (alpha) 对应伪代码的第 1 步, 架构参数梯度下降
                # alpha_optim.zero_grad()  # 清除之前学到的梯度的参数
                # lr = scheduler.get_last_lr()[0]
                # # lr = scheduler.get_lr()[0]
                # writer.add_scalar('train/lr', lr, cur_step)
                # architect.unrolled_backward(trn_imgs, trn_hr_imgs, val_imgs, val_hr_imgs, lr, w_optim)
                # alpha_optim.step()

                # phase 1. child network step (w) 对应伪代码的第 2 步, 网络参数梯度下降
                w_optim.zero_grad()  # 清除之前学到的梯度的参数
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                lr_optimizer.zero_grad()
                lr_optimizer_center.zero_grad()
                hr_optimizer.zero_grad()
                hr_optimizer_center.zero_grad()

                trn_sr = model(trn_imgs)

                sr_loss = 10 * model.criterion(trn_sr, trn_hr_imgs)  # 计算的是点对点的损失，图像层面

                trn_imgs = resize_(trn_imgs)
                lr_score, lr_feat, lr_x3, lr_x5, lr_x5x = lr_cls_model(trn_imgs, trn_targets)
                score, feat, sr_x3, sr_x5, sr_x5x = cls_model(trn_sr, trn_targets)
                hr_score, hr_feat, hr_x3, hr_x5, hr_x5x = hr_cls_model(trn_hr_imgs, trn_targets, is_hr=True)

                cls_loss = loss_func(score, feat, trn_targets)
                hr_cls_loss = loss_func(hr_score, hr_feat, trn_targets)
                lr_cls_loss = loss_func(lr_score, lr_feat, trn_targets)

                feat_loss = 0.003 * L1_loss(sr_x5x, hr_x5x.detach(), sr_x3, hr_x3.detach())
                # print(feat_loss)
                srd1 = SRD_loss(sr_x5, hr_x5.detach())
                srd2 = SRD_loss(sr_x3, hr_x3.detach())
                # print(srd1, srd2)
                srd_loss = 0.003 * (srd1 + 0.1 * srd2)

                loss = sr_loss + cls_loss + hr_cls_loss + lr_cls_loss + feat_loss + srd_loss
                # loss = 10 * sr_loss + cls_loss + hr_cls_loss + lr_cls_loss + 0.003 * srd_loss
                loss.backward()

                # gradient clipping  梯度裁剪
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)

                w_optim.step()  # 应用梯度
                optimizer.step()
                lr_optimizer.step()
                hr_optimizer.step()

                if 'center' in cfg.LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                    optimizer_center.step()
                    lr_optimizer_center.step()
                    hr_optimizer_center.step()

                prec1, prec5 = tools.accuracy(score, trn_targets, topk=(1, 5))
                sr_losses.update(sr_loss.item(), N)
                cls_losses.update(cls_loss.item(), N)
                hr_cls_losses.update(hr_cls_loss.item(), N)
                lr_cls_losses.update(lr_cls_loss.item(), N)
                feat_losses.update(feat_loss.item(), N)
                srd_losses.update(srd_loss.item(), N)
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % log_period == 0 or step == len(train_loader) - 1:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} SR_Loss={sr_losses.avg:.3f} Cls_Loss={cls_losses.avg:.3f} HR_cls_Loss={hr_cls_losses.avg:.3f} LR_cls_Loss={lr_cls_losses.avg:.3f} Feat_loss={feat_losses.avg:.3f} SRD_loss={srd_losses.avg:.3f} Total_Loss={losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.2%}, {top5.avg:.2%})".format(
                            epoch + 1, cfg.MAX_EPOCHS, step, len(train_loader) - 1, sr_losses=sr_losses,
                            cls_losses=cls_losses, hr_cls_losses=hr_cls_losses,
                            lr_cls_losses=lr_cls_losses, feat_losses=feat_losses, srd_losses=srd_losses, losses=losses,
                            top1=top1, top5=top5))

                # if step == 1641:
                #     MEAN = [0.485, 0.456, 0.406]
                #     STD = [0.229, 0.224, 0.225]
                #     var_img = unnormalize(trn_sr, MEAN, STD)
                #     writer.add_images('train/sr', var_img, cur_step)
                # writer.add_scalar('train/loss', loss.item(), cur_step)
                # writer.add_scalar('train/top1', prec1.item(), cur_step)
                # writer.add_scalar('train/top5', prec5.item(), cur_step)
                cur_step += 1
        else:
            train_loader, _, _, num_query, num_classes = make_dataloader(cfg, mode='train')
            for step, (trn_hr_imgs, trn_imgs, trn_targets) in enumerate(train_loader):
                # for step, (img, vid) in enumerate(train_loader):
                trn_hr_imgs, trn_imgs, trn_targets = trn_hr_imgs.to(device, non_blocking=True), trn_imgs.to(device, non_blocking=True), trn_targets.to(device, non_blocking=True)

                N = trn_imgs.size(0)  # N = 64

                # lr = scheduler.get_last_lr()[0]
                # lr = scheduler.get_lr()[0]
                # writer.add_scalar('train/lr', lr, cur_step)

                # phase 1. child network step (w) 对应伪代码的第 2 步, 网络参数梯度下降
                w_optim.zero_grad()  # 清除之前学到的梯度的参数

                trn_sr = model(trn_imgs)
                sr_loss = 10 * model.criterion(trn_sr, trn_hr_imgs)

                optimizer.zero_grad()
                optimizer_center.zero_grad()
                lr_optimizer.zero_grad()
                lr_optimizer_center.zero_grad()
                hr_optimizer.zero_grad()
                hr_optimizer_center.zero_grad()

                trn_imgs = resize_(trn_imgs)
                lr_score, lr_feat, lr_x3, lr_x5, lr_x5x = lr_cls_model(trn_imgs, trn_targets)
                score, feat, sr_x3, sr_x5, sr_x5x = cls_model(trn_sr, trn_targets)
                hr_score, hr_feat, hr_x3, hr_x5, hr_x5x = hr_cls_model(trn_hr_imgs, trn_targets, is_hr=True)

                cls_loss = loss_func(score, feat, trn_targets)
                hr_cls_loss = loss_func(hr_score, hr_feat, trn_targets)
                lr_cls_loss = loss_func(lr_score, lr_feat, trn_targets)

                feat_loss = 0.003 * L1_loss(sr_x5x, hr_x5x.detach(), sr_x3, hr_x3.detach())
                # print(feat_loss)
                srd1 = SRD_loss(sr_x5, hr_x5.detach())
                srd2 = SRD_loss(sr_x3, hr_x3.detach())
                # print(srd1, srd2)
                srd_loss = 0.003 * (srd1 + 0.1 * srd2)

                loss = sr_loss + cls_loss + hr_cls_loss + lr_cls_loss + feat_loss + srd_loss
                # loss = 10 * sr_loss + cls_loss + hr_cls_loss + lr_cls_loss + 0.003 * srd_loss
                loss.backward()

                # gradient clipping  梯度裁剪
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)

                w_optim.step()  # 应用梯度
                optimizer.step()
                lr_optimizer.step()
                hr_optimizer.step()

                if 'center' in cfg.LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                    optimizer_center.step()
                    lr_optimizer_center.step()
                    hr_optimizer_center.step()

                prec1, prec5 = tools.accuracy(score, trn_targets, topk=(1, 5))
                sr_losses.update(sr_loss.item(), N)
                cls_losses.update(cls_loss.item(), N)
                hr_cls_losses.update(hr_cls_loss.item(), N)
                lr_cls_losses.update(lr_cls_loss.item(), N)
                feat_losses.update(feat_loss.item(), N)
                srd_losses.update(srd_loss.item(), N)
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % log_period == 0 or step == len(train_loader) - 1:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} SR_Loss={sr_losses.avg:.3f} Cls_Loss={cls_losses.avg:.3f} HR_cls_Loss={hr_cls_losses.avg:.3f} LR_cls_Loss={lr_cls_losses.avg:.3f} Feat_loss={feat_losses.avg:.3f} SRD_loss={srd_losses.avg:.3f} Total_Loss={losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.2%}, {top5.avg:.2%})".format(
                            epoch + 1, cfg.MAX_EPOCHS, step, len(train_loader) - 1, sr_losses=sr_losses,
                            cls_losses=cls_losses, hr_cls_losses=hr_cls_losses,
                            lr_cls_losses=lr_cls_losses, feat_losses=feat_losses, srd_losses=srd_losses, losses=losses,
                            top1=top1, top5=top5))

                # if step == 1641:
                #     MEAN = [0.485, 0.456, 0.406]
                #     STD = [0.229, 0.224, 0.225]
                #     var_img = unnormalize(trn_sr, MEAN, STD)
                #     writer.add_images('train/sr', var_img, cur_step)
                # writer.add_scalar('train/loss', loss.item(), cur_step)
                # writer.add_scalar('train/top1', prec1.item(), cur_step)
                # writer.add_scalar('train/top5', prec5.item(), cur_step)
                cur_step += 1
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        # var_img = unnormalize(trn_sr, MEAN, STD)
        # writer.add_images('train/sr', var_img, cur_step)
        logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.MAX_EPOCHS, top1.avg))

        scheduler.step()
        cls_scheduler.step()
        lr_cls_scheduler.step()
        hr_cls_scheduler.step()

        # log
        # genotype
        # genotype = model.genotype()  # 对应论文2.4 选出来权重值大的两个前驱节点，并把(操作，前驱节点)存下来
        # logger.info("genotype = {}".format(genotype))

        # genotype as a image
        # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        # caption = "Epoch {}".format(epoch + 1)
        # plot(genotype.normal, plot_path + "-normal", caption)

        # test
        if (epoch + 1) > config.epochs:
            if cfg.DATASET == 'CUHK03':
                rank1 = test_cuhk03(model, lr_cls_model, cls_model, resize_)
            else:
                rank1 = test(model, lr_cls_model, cls_model, resize_)
        else:
            rank1 = 0.

        # save
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET)):
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET))

        if (epoch + 1) <= config.epochs:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'sr_checkpoint.pth'))
            torch.save(cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'cls_checkpoint.pth'))
            torch.save(lr_cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'lr_cls_checkpoint.pth'))
            torch.save(hr_cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'hr_cls_checkpoint.pth'))

        if best_rank1 < rank1 and (epoch + 1) > config.epochs:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'sr_{}.pth'.format(epoch + 1)))
            torch.save(cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'cls_{}.pth'.format(epoch + 1)))
            torch.save(lr_cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'lr_cls_{}.pth'.format(epoch + 1)))
            torch.save(hr_cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, 'hr_cls_{}.pth'.format(epoch + 1)))
            best_rank1 = rank1
            # best_genotype = genotype

        # if epoch % checkpoint_period == 0:
        #     torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, cfg.SR_MODEL_NAME + '_{}.pth'.format(epoch)))
        #     torch.save(cls_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.DATASET, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

    logger.info("Final best Rank-1 = {:.1%}".format(best_rank1))
    # logger.info("Best Genotype = {}".format(best_genotype))
    end_time = time.time()
    logger.info("Training Times : %dh %dm %ds" %((end_time - start_time)//3600, (end_time - start_time)//60%60, (end_time - start_time)%60))

def test_cuhk03(model, lr_cls_model, cls_model, resize_):
    _, query_loader, gallery_loader, num_query, num_classes = make_dataloader(cfg, mode='test')

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM,
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()

    logger.info("Enter inferencing")

    model.eval()
    cls_model.eval()
    lr_cls_model.eval()

    batch_time = tools.AverageMeter()
    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for n_iter, (hr_imgs, imgs, pids, camids, imgpath) in enumerate(query_loader):
            imgs = imgs.cuda()

            end = time.time()
            sr_img = model(imgs)
            lr_img = resize_(imgs)
            lr_features = lr_cls_model(lr_img)
            features = cls_model(sr_img)
            features = torch.cat((lr_features, features), dim=1)
            batch_time.update(time.time() - end)

            features = features.data
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for n_iter, (hr_imgs, imgs, pids, camids, imgpath) in enumerate(gallery_loader):
            imgs = imgs.cuda()

            end = time.time()
            sr_img = model(imgs)
            lr_img = resize_(imgs)
            lr_features = lr_cls_model(lr_img)
            features = cls_model(sr_img)
            features = torch.cat((lr_features, features), dim=1)
            batch_time.update(time.time() - end)

            features = features.data
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, 32))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.cpu().numpy()

    test_distance = 'global'

    if not test_distance == 'global':
        print("Only using global branch")
        from utils.distance import low_memory_local_dist
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
        if test_distance == 'local':
            print("Only using local branch")
            distmat = local_distmat
        if test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat + distmat
    print("Computing CMC and mAP")
    cmc, mAP = eval_metrics.evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=True)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in [1, 5, 10, 20]:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, cls_model, lr_cls_model, dataloaders):
    resize_ = T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC)

    features = torch.FloatTensor()
    count = 0
    with torch.no_grad():
        for data in dataloaders:
            hr_imgs, imgs, pids, camids, imgpath = data
            n, c, h, w = imgs.size()
            count += n
            ff = torch.FloatTensor(n, 4096).zero_()
            for i in range(2):
                if (i == 1):
                    imgs = fliplr(imgs)
                input_img = Variable(imgs.cuda())
                # img=img.unsqueeze(0)outputs_4
                sr_img = model(input_img)
                lr_img = resize_(input_img)
                lr_features = lr_cls_model(lr_img)
                f = cls_model(sr_img)
                f = torch.cat((lr_features, f), dim=1).cpu()
                ff = ff + f
            # norm feature

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
    return features

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

# Evaluate
def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def test(model, lr_cls_model, cls_model, resize_):
    _, query_loader, gallery_loader, num_query, num_classes = make_dataloader(cfg, mode='test')

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM,
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()

    logger.info("Enter inferencing")

    model.eval()
    cls_model.eval()
    lr_cls_model.eval()

    # Extract feature
    query_feature = extract_feature(model, cls_model, lr_cls_model, query_loader)
    gallery_feature = extract_feature(model, cls_model, lr_cls_model, gallery_loader)

    query_label, query_cam = [], []
    gallery_label, gallery_cam = [], []

    for item in query_loader.dataset:
        query_label.append(item[2])
        query_cam.append(item[3])

    for item in gallery_loader.dataset:
        gallery_label.append(item[2])
        gallery_cam.append(item[3])

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    # print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label,
                                   gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    return CMC[0]

if __name__ == '__main__':
    train()