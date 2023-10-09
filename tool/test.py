import os
import time
import datetime
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util.s3dis import swDataset, load_adj
from util.data_util import collate_fn,saveGiiLabel
from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize

random.seed(123)
np.random.seed(123)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    T_k = load_adj()
    test(model, criterion, names, T_k)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'swDataset':
        data_list = swDataset(split='', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=None, shuffle_index=False, loop=1)
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    with open(os.path.join(args.data_root,"datalist_sw_t1.txt"),"r") as f:
        sub_list = f.readlines()
        sub_list = [ix[:-1].split() for ix in sub_list]

    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list, sub_list


def test(model, criterion, names, T_k):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list, sub_list = data_prepare()
    res_filename = "output" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join(os.getcwd(),res_filename))
    test_loader = torch.utils.data.DataLoader(data_list, batch_size=args.batch_size_test, shuffle=None, num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False, collate_fn=collate_fn)
    for idx, item in enumerate(test_loader):
        s_i, e_i = idx * args.batch_size_test, min((idx + 1) * args.batch_size_test, len(sub_list))
        data_idx = sub_list[s_i:e_i]
        coord_part, feat_part, label, offset_part = item
        coord_part = coord_part.cuda(non_blocking=True)
        feat_part = feat_part.cuda(non_blocking=True)
        offset_part = offset_part.cuda(non_blocking=True)
        with torch.no_grad():
            pred_part = model([coord_part, feat_part, offset_part], T_k)  # (n, k)

        torch.cuda.empty_cache()
        pred = pred_part
        # logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))
        loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
        pred = pred.max(1)[1].data.cpu().numpy()
        for save_idx in range(len(data_idx)):
            pred_si, pred_se = save_idx * len(pred)//len(data_idx), (save_idx + 1) * len(pred)//len(data_idx)
            pred_session = pred[pred_si:pred_se]
            savepath = os.path.join(res_filename,'{}_{}.indi.{}.label.gii'.format(data_idx[save_idx][0], data_idx[save_idx][1], "L"))
            saveGiiLabel(pred_session, "L", savepath)
        # calculation 1: add per room predictions

    #     intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
    #     intersection_meter.update(intersection)
    #     union_meter.update(union)
    #     target_meter.update(target)

    #     accuracy = sum(intersection) / (sum(target) + 1e-10)
    #     batch_time.update(time.time() - end)
    #     logger.info('Test: [{}/{}]-{} '
    #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
    #                 'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time, accuracy=accuracy))
    #     pred_save.append(pred); label_save.append(label)
    #     np.save(pred_save_path, pred); np.save(label_save_path, label)

    # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
    #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
    #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # calculation 1
    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU1 = np.mean(iou_class)
    # mAcc1 = np.mean(accuracy_class)
    # allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # # calculation 2
    # intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    # iou_class = intersection / (union + 1e-10)
    # accuracy_class = intersection / (target + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection) / (sum(target) + 1e-10)
    # logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    # logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    # for i in range(args.classes):
    #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
