import os
import sys
import time
import torch
import pathlib
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.modelsummary import get_model_summary
from utils.train_utils import AverageMeter, get_confusion_matrix, adjust_learning_rate, create_logger


def train(
    cfg, 
    dataloader, 
    model, 
    loss_fn, 
    optimizer, 
    lr_scheduler, 
    scaler, 
    writer_dict,
    epoch, 
):
    model.train()
    
    ave_loss = AverageMeter()
    steps_tot = epoch*len(dataloader) 
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    
    for step, batch in enumerate(dataloader):
        X, y, _, _ = batch
        X, y = X.cuda(), y.long().cuda() 
        
        # Compute prediction and loss
        with torch.cuda.amp.autocast():
            pred = model(X)
            losses = loss_fn(pred, y)
        loss = losses.mean()
        
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # update average loss
        ave_loss.update(loss.item())
        
        # update learning schedule
        lr_scheduler.before_train_iter()
        lr = lr_scheduler.get_lr(int(steps_tot+step), cfg.TRAIN.BASE_LR)
        #lr = adjust_learning_rate(optimizer, cfg['BASE_LR'], cfg['END_LR'], step+steps_tot, cfg['DECAY_STEPS'])
        
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

            

def validate(cfg, dataloader, model, loss_fn, writer_dict):
    model.eval()
    
    ave_loss = AverageMeter()
    iter_steps = len(dataloader.dataset) // cfg.BATCH_SIZE
    confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES, 1))
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, y, _, _ = batch
            size = y.size()
            X, y = X.cuda(), y.long().cuda()
            
            pred = model(X)
            losses = loss_fn(pred, y)
            loss = losses.mean()   
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]    
            for i, x in enumerate(pred):
                confusion_matrix[..., i] += get_confusion_matrix(
                    y, x, size, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_CLASSES)
            ave_loss.update(loss.item())
            
    pos = confusion_matrix[..., 0].sum(1)
    res = confusion_matrix[..., 0].sum(0)
    tp = np.diag(confusion_matrix[..., 0])
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    for key, val in trainid2label.items():
        if key != cfg.DATASET.IGNORE_LABEL and key != -1:
            writer.add_scalar('valid_mIoU_{}'.format(val.name), IoU_array[key], global_steps)    
    writer_dict['valid_global_steps'] = global_steps + 1
        
    return ave_loss.average(), mean_IoU, IoU_array



def testval(cfg, testloader, model, sv_dir='', sv_pred=False, sliding_inf=False):
    model.eval()
    confusion_matrix = np.zeros((cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_CLASSES))
    
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            if sliding_inf:
                pred = testloader.dataset.sliding_inference(model, image)
            else:
                pred = testloader.dataset.inference(model, image)

            confusion_matrix += get_confusion_matrix(
                label, pred, size, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_CLASSES)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
                
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                if sliding_inf:
                    # print(pred.shape)
                    pred = np.squeeze(pred, 0)
                testloader.dataset.save_pred(image, pred, sv_path, name)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def testvideo(cfg, testloader, model, sv_dir='', sv_pred=False):
    model.eval()

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, _, name, *border_padding = batch
            size = image.size()
            pred = testloader.dataset.inference(model, image)
                
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'video_frames')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                testloader.dataset.save_pred(image, pred, sv_path, name)

    print("done!")
