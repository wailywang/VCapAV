#!/usr/bin/env python
"""
Script to train, dev, and eval of Countermeasure model for anti-spoofing task. 

Usage:

 -eval_scores: path to the score file 
 -eval_data_name: path to the directory that has CM protocol.
 -model_path: the path to the model checkpoint, which is used to save the output EER file. 

Example:
For training:
$: python main.py --gpu=0 --track=ASVspoof --comment=no_rawboost --algo=0 --seed=1 \
    --data_aug=False --vad=False --model_pretrain=None --dropout=0.1 --use_amp=True \
    --start_epoch=0 --num_epochs=100 --batch_size=768 --warm_up_steps=10 
For evaluation:
$: python main.py --gpu=0 --eval=True --model_path=/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/exp_/ASVspoof_celoss_0EpochStart_128bs_0.001lr_augFalse_rvbFalse_seed1_ResNet18_ASP_debug \
    --eval_epoch=1 --eval_batch_size=128

Author: "Yikang Wang"
Email: "wwm1995@alps-lab.org"
"""

import os, random, time, math, sys, pdb
import argparse
import numpy as np
import pickle as pk
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
# from nemo.core.optim.lr_scheduler import CosineAnnealing
from omegaconf import OmegaConf

import modules.model_spk as models, modules.back_classifier as classifiers
from model_utils import save_checkpoint, save_ramdom_state
from dataset.dataset import TrainDevDataset_ASVspoof5_offset, EvalDataset_offset
from dataset.sampler import WavBatchSampler
import dataset.feats as featurextractor
from infer_from_score import compute_eer
from other_utils import ProgressMeter, AverageMeter, accuracy, str_to_bool, get_lr, change_lr
from utils.pit_criterion import cal_loss as sisnr_loss
import csv
import pandas as pd

def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle
    
        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def main(args):
    # additional args

    #make experiment reproducible
    set_random_seed(args.seed)

    #GPU device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(args.device))
    track = args.track # FAD or ASVspoof

    #define model saving path
    args.model_tag = f'{track}_{args.loss}loss_{args.start_epoch}EpochStart_{args.batch_size}bs_{args.lr}lr_aug{args.data_aug}_rvb{args.reverb}_seed{args.seed}_{args.model}'
    if args.comment:
        args.model_tag = args.model_tag + '_{}'.format(args.comment)

    args.save_dir = os.path.join(f'{args.exp_dir}/', args.model_tag)
    
    if not os.path.exists(args.save_dir):
        os.system(f'mkdir -p {args.save_dir}')
        os.system(f'cp {sys.argv[0]} {args.save_dir}')
    if not os.path.exists(f'{args.log_dir}/{args.model_tag}'):
            os.system(f'mkdir -p {args.log_dir}/{args.model_tag}')
    # make a copy of original stdout route
    stdout_backup = sys.stdout
    # define the log file that receives your log info
    log_file = open(f'{args.log_dir}/{args.model_tag}/train.log', "w")
    # redirect print output to log file
    sys.stdout = log_file

    # define dataset
    noise_dict = {}
    
    key2int = {'bonafide':1,'spoof':0,'genuine':1,'fake':0}
    
    ## training data
    utt2wav = [line.split() for line in open(f'/your own path/{args.trn_data_name}/wav.scp')]
    utt2label = [line.split() for line in open(f'/your own path/{args.trn_data_name}/utt2label')]
    
    ##########partial data debug##############
    # 将两个列表合并为一个列表
    combined = list(zip(utt2wav, utt2label))
    # 打乱这个列表
    random.shuffle(combined)
    # 取出partial个数据
    if args.partial_data:
        utt2wav[:], utt2label[:] = zip(*combined[:args.partial_data])
    else:
        utt2wav[:], utt2label[:] = zip(*combined)
    ##########partial data debug##############
    
    utt2label = {u:key2int[s] for u, s in utt2label}

    # 检查键是否存在
    key_to_check = 'Jix2dM4O0jQ_000361_ldm2'
    if key_to_check in utt2label:
        print(f"Key '{key_to_check}' is present in utt2label.")
    else:
        print(f"Key '{key_to_check}' is NOT present in utt2label.")

    trn_dataset = TrainDevDataset_ASVspoof5_offset(args, utt2wav, utt2label, fs=args.fs, preemph=args.preemph, is_aug=args.data_aug, aug_rate=args.aug_rate, snr_range=args.snr_range, noise_dict=noise_dict, is_specaug=args.is_specaug,vad=args.vad, speed=args.speed_aug, reverb=args.reverb)
    trn_sampler = WavBatchSampler(trn_dataset, args.dur_range, shuffle=True, batch_size=args.batch_size, drop_last=True)
    trn_loader = DataLoader(trn_dataset, batch_sampler=trn_sampler, num_workers=args.workers, pin_memory=True)

    del trn_dataset 
    
    ## dev data
    utt2wav = [line.split() for line in open(f'/your own path/{args.dev_data_name}/wav.scp')]
    utt2label = [line.split() for line in open(f'/your own path/{args.dev_data_name}/utt2label')]
    
    ##########partial data debug##############
    # 将两个列表合并为一个列表
    combined = list(zip(utt2wav, utt2label))
    # 打乱这个列表
    random.shuffle(combined)
    # 取出partial个数据
    if args.partial_data:
        dev_partial = args.partial_data//2
        utt2wav[:], utt2label[:] = zip(*combined[:dev_partial])
    else:
        utt2wav[:], utt2label[:] = zip(*combined)
    ##########partial data debug##############
    
    utt2label = {u:key2int[s] for u, s in utt2label}

    # 再次检查键是否存在于开发数据集
    if key_to_check in utt2label:
        print(f"Key '{key_to_check}' is present in dev utt2label.")
    else:
        print(f"Key '{key_to_check}' is NOT present in dev utt2label.")

    dev_dataset = TrainDevDataset_ASVspoof5_offset(args, utt2wav, utt2label, fs=args.fs, preemph=args.preemph, is_aug=args.data_aug, aug_rate=args.aug_rate, snr_range=args.snr_range, noise_dict=noise_dict, is_specaug=args.is_specaug,vad=args.vad, speed=args.speed_aug, reverb=args.reverb)
    dev_sampler = WavBatchSampler(dev_dataset, args.dur_range, shuffle=False, batch_size=args.batch_size, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_sampler=dev_sampler, num_workers=args.workers, pin_memory=True)

    del dev_dataset


    ## eval data
    if args.partial_data:
        utt2wav = utt2wav[:args.partial_data]

    eval_dataset = EvalDataset_offset(sorted(utt2wav), fs=args.fs, frame_level=args.frame_level, preemph=args.preemph, vad=args.vad)
    eval_sampler = WavBatchSampler(eval_dataset, args.dur_range, shuffle=False, batch_size=args.eval_batch_size, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    del eval_dataset
    
    # model config file
    cfg = OmegaConf.load(args.model_cfg)
    if args.num_pretrain_layer is not None:
        cfg['encoder']['n_layers'] = args.num_pretrain_layer
   
    # spectral feature calculation
    featCal = getattr(featurextractor, args.feat)(
                      sample_rate = cfg['preprocessor']['sample_rate'],
                      n_fft = cfg['preprocessor']['n_fft'],
                      win_length = int(cfg['preprocessor']['window_size'] * 16000),
                      hop_length = int(cfg['preprocessor']['window_stride'] * 16000),
                      window_fn = cfg['preprocessor']['window'],
                      n_mels = cfg['preprocessor']['features'],
                      trim = args.trim).cuda()
    featCal.eval()

    # define core model
    model = getattr(models, args.model)(cfg['encoder'], dropout=args.dropout)
        
    # define back-end classifier
    embd_dim = cfg['encoder']['embedding_size']
    classifier = getattr(classifiers, args.classifier)(embd_dim, out_features=2, m=args.angular_m, s=args.angular_s)
    
    model = model.to(args.device)
    classifier = classifier.to(args.device)

    # define loss function (criterion) and optimizer
    mse_loss = torch.nn.MSELoss()
    if args.track == 'ASVspoof5':
        weight = torch.FloatTensor([0.1, 0.8]).to(args.device)
    else:
        weight = torch.FloatTensor([0.5, 0.5]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=weight).to(args.device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                                  lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=args.min_lr)
    batch_per_epoch = len(trn_loader)
    args.lr_lambda = lambda x: args.lr / (batch_per_epoch * args.warm_up_epoch) * (x + 1)
    for g in optimizer.param_groups:
        g['lr'] = args.lr
    
    # resume model [need input args.model_path]
    if args.start_epoch > 0 or args.eval:
        if args.eval:
            resume_model_path = os.path.join(args.model_path, f'epoch_{args.eval_epoch}.pkl')
        elif args.start_epoch > 0:
            resume_model_path = os.path.join(args.model_path, f'epoch_{args.start_epoch-1}.pkl')
            args.save_dir = args.model_path

        
        checkpoint = torch.load(resume_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        classifier.load_state_dict(checkpoint['classifier'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

        checkpoint = torch.load(os.path.join(args.model_path, 'random_state.pkl'), map_location='cpu')
        random.setstate(checkpoint['random'])
        np.random.set_state(checkpoint['np'])
        torch.set_rng_state(checkpoint['torch'])
        torch.cuda.set_rng_state_all(checkpoint['torch_cuda'])
        print(f'Model loaded : {resume_model_path}')
        print('LR %.8f'  % get_lr(optimizer))
    else:
        print(str(model) + '\n' + str(classifier) + '\n', flush=True)
    
    print('='*60, flush=True)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('='*60, flush=True)
    if args.eval: #[need input args.model_path]
        log_file.close() 
        sys.stdout = stdout_backup
        args.eval_dir = os.path.join(args.score_dir, os.path.basename(args.model_path), os.path.dirname(args.eval_data_name))
        os.system(f'mkdir -p {args.eval_dir}')
        evaluation(eval_loader, featCal, model, classifier, args)
    else:    
        top3_DevLossindices, top3_EERindices = main_worker(trn_loader, dev_loader, featCal, model, classifier, mse_loss, criterion, optimizer, scaler, scheduler, args)
        print(f'\n top3_DevLossindices: \n {top3_DevLossindices} \n')
        print(f'\n top3_EERindices: \n {top3_EERindices} \n')
        log_file.close() 
        sys.stdout = stdout_backup

def main_worker(trn_loader, dev_loader, featCal, model, classifier, mse_loss, criterion, optimizer, scaler, scheduler, args):
    # additional args 

    label_dict = {}
    with open(f'/your own path/{args.dev_data_name}/utt2label', 'r') as file:
        for line in file:
            key, label = line.strip().split()
            label_dict[key] = (str(label))
    # Training 
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    writer = SummaryWriter(f'{args.log_dir}/{args.model_tag}')
    # dev_losses = []
    # dev_EER = []
    # EER_epoch = {}
    wait_del_indices = set()
    best_eer = float('inf')
    for epoch in range(start_epoch, num_epochs): # Training loop
        # pdb.set_trace()
        running_loss, acc = train_epoch(trn_loader, featCal, model, classifier, mse_loss, criterion, optimizer, scaler, scheduler, epoch, args)
        writer.add_scalar('train_loss', running_loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)

        if epoch % 2 == 0:
            # 计算dev loss
            EER, dev_loss = dev_epoch(dev_loader, featCal, model, classifier, mse_loss, criterion, label_dict, args)
            # 在每 2 个epoch结束时，将dev_EER和dev_loss的值写入文件
            if start_epoch == 0 and epoch == 0:
                with open(f'{args.save_dir}/results.csv', 'w', newline='') as file:
                    wr = csv.writer(file)
                    wr.writerow(['epoch', 'dev_EER', 'dev_loss'])
                    wr.writerow([epoch, EER, dev_loss])
            else:
                with open(f'{args.save_dir}/results.csv', 'a', newline='') as file:
                    wr = csv.writer(file)
                    wr.writerow([epoch, EER, dev_loss])

            # tensorboard
            writer.add_scalar('dev_EER', EER, epoch)
            writer.add_scalar('dev_loss', dev_loss, epoch)
            
            print(f'Validate\tEpoch {epoch:3d}\tTrainLoss {running_loss:2.6f}\tLR {get_lr(optimizer):f}\tDevLoss {dev_loss:.6f}\tEER {EER:.4f}\n', flush=True)
            
            
            # 在需要的时候，读取文件并获取dev_EER和dev_loss的值
            res = pd.read_csv(f'{args.save_dir}/results.csv', header=0)
            res.columns = ['epoch', 'dev_EER', 'dev_loss']
            
            # 获取dev_loss的值
            dev_losses = res.sort_values(by=['dev_loss', 'epoch'], ascending=[True, False])
            # 获取dev_EER的值
            dev_EER = res.sort_values(by=['dev_EER', 'epoch'], ascending=[True, False])
            
            # 选择dev loss,最低的一个,和dev EER最低的2个模型的checkpoint 分别进行保存
            if len(dev_EER) > 2:
                eer_sorted_indices = set(dev_EER['epoch'].values[:2])
                loss_sorted_indices = set(dev_losses['epoch'].values[:1])
                
                # 确定所有需要保留的epoch
                keep_indices = eer_sorted_indices.union(loss_sorted_indices)
                
                # 检查是否需要将当前epoch添加到等待删除的集合中
                del_eer_index = dev_EER['epoch'].values[2]
                if del_eer_index not in loss_sorted_indices:
                    wait_del_indices.add(del_eer_index)

                del_loss_index = dev_losses['epoch'].values[1]
                if del_loss_index not in eer_sorted_indices:
                    wait_del_indices.add(del_loss_index)
                
                # 需要删除的索引 = 等待删除的索引 - 保留的索引
                del_indeces = wait_del_indices - keep_indices

                for epoch_to_del in del_indeces:
                    checkpoint_path = os.path.join(args.save_dir,f"epoch_{epoch_to_del}.pkl")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                        print(f"removed the EER/loss 4th. result: epoch_{epoch_to_del}.pkl", flush=True)
                        wait_del_indices.remove(epoch_to_del) # 删除之后，从等待删除的集合中删除该索引

                # 如果当前epoch的索引在前3个中，则保存当前epoch的checkpoint，并删除之前的3个中EER最大的checkpoint
                if epoch in eer_sorted_indices:
                    # # 删除之前的3个中EER最大的checkpoint, 但不能删到loss最小的3个
                    # del_indices = dev_EER['epoch'].values[3]
                    # if del_indices in loss_sorted_indices:
                    #     print(f"epoch_{del_indices}.pkl is in top-3 DevLoss results, not removed", flush=True)
                    # else:
                    #     checkpoint_path = os.path.join(args.save_dir,f"epoch_{del_indices}.pkl")
                    #     if os.path.exists(checkpoint_path):
                    #         os.remove(checkpoint_path)
                    #         print(f"removed the EER 4th. result: epoch_{del_indices}.pkl", flush=True)
                    
                    save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
                    print(f"updated top-3 EER results with epoch_{epoch}.pkl", flush=True)

                # 如果当前epoch的索引在前3个中，则保存当前epoch的checkpoint，并删除之前的3个中loss最大的checkpoint
                elif epoch in loss_sorted_indices:
                    save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
                    print(f"updated top-3 DevLoss results with epoch_{epoch}.pkl", flush=True)
                    
                    # # 删除之前的3个中loss最大的checkpoint，但不能删到EER最小的3个
                    # del_indices = dev_losses['epoch'].values[3]
                    # if del_indices in eer_sorted_indices:
                    #     print(f"epoch_{del_indices}.pkl is in top-3 EER results, not removed", flush=True)
                    # else:
                    #     checkpoint_path = os.path.join(args.save_dir,f"epoch_{del_indices}.pkl")
                    #     if os.path.exists(checkpoint_path):
                    #         os.remove(checkpoint_path)
                    #         print(f"removed the DevLoss 4th. result: epoch_{del_indices}.pkl", flush=True)
                else:
                    pass

                
            else:
                # 保存当前epoch的checkpoint
                save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
                print(f"saved first some results epoch_{epoch}.pkl", flush=True)
            
            if args.early_stop:

                #  如果当前epoch的EER比最优的EER更好，那么更新最优的EER
                if EER < best_eer:
                    best_eer = EER
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                # 如果recent_eers列表中有early_stop个元素，并且最新的EER不小于最旧的EER，那么停止训练
                if no_improve_counter >= args.early_stop//2:
                    print(f"Early stopping due to EER not decreasing in the last {args.early_stop} epochs.")
                    # save the last epoch
                    save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
                    print(f"saved last epoch_{epoch}.pkl", flush=True)
                    break

        if epoch == args.num_epochs - 1:
            save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
            print(f"saved last epoch_{epoch}.pkl", flush=True)
        elif epoch == 99:
            save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
            print(f"saved epoch_{epoch}.pkl", flush=True)
        elif (epoch+1)%25 == 0:
            save_checkpoint(args.save_dir, epoch, model, classifier, optimizer, scheduler, scaler)
            print(f"saved epoch_{epoch}.pkl", flush=True)
        if epoch == args.start_epoch:
            save_ramdom_state(args.save_dir, random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            print(f"saved random state", flush=True)
    return dev_losses[:3], dev_EER[:3]

def train_epoch(trn_loader, featCal, model, classifier, mse_loss, criterion, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter('Trn_time', ':6.3f')
    data_time = AverageMeter('Load_time', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.mse_loss:
        unet_losses = AverageMeter('MSE_Loss', ':.6f')
        class_losses = AverageMeter('Class_Loss', ':.6f')
        progress = ProgressMeter(len(trn_loader), [batch_time, data_time, unet_losses, class_losses, losses, top1], prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(len(trn_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()
    classifier.train()

    end = time.time()
    
    # if string pair or unet NOT in model name:  -1 means not found
    for i, (_, batch_x, _, key, _) in enumerate(trn_loader):
        if epoch < args.warm_up_epoch:
            change_lr(optimizer, args.lr_lambda(len(trn_loader) * epoch + i))
        data_time.update(time.time() - end)

        feats = batch_x.to(args.device, non_blocking=True)
        feats_len = torch.LongTensor([feats.shape[1]]*feats.shape[0]).to(args.device, non_blocking=True)

        feats, feats_len = featCal(input_signal=feats, length=feats_len)
        feats_len = feats_len.to(args.device, non_blocking=True)
        key = key.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = classifier(model(feats, feats_len), key)
            loss = criterion(outputs, key)
        prec1 = accuracy(outputs, key)
        losses.update(loss.item(), feats.size(0))
        top1.update(prec1[0].item(), feats.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if epoch >= args.model_freeze_epochs:
            scheduler.step(losses.avg)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0: # print every 20 batches
            progress.display(i, 'Len %3d' % feats.size(-1), 'LR %.8f'  % get_lr(optimizer))

    loss, acc_1 = losses.avg, top1.avg

    return loss, acc_1

def dev_epoch(dev_loader, featCal, model, classifier, mse_loss, criterion, label_dict, args):
    end = time.time()
    print('Evaluating on dev set...', flush=True)
    model.eval()
    classifier.eval()
    
    dev_loss = 0.0
    num_total = 0.0
    utt_list = []
    score_list = []
    #loss_dev = AverageMeter()
    with torch.no_grad():
        for _, batch_x, _, key, utt in dev_loader:
            batch_x = batch_x.to(args.device, non_blocking=True)
            feats_len = torch.LongTensor([batch_x.shape[1]]*batch_x.shape[0]).to(args.device, non_blocking=True)
            feats, feats_len = featCal(input_signal=batch_x, length=feats_len)
            feats_len = feats_len.to(args.device, non_blocking=True)
            key = key.to(args.device, non_blocking=True)
            # if string pair or unet NOT in model name:  -1 means not found
     
            output = classifier(model(feats, feats_len))
            # if args.classifier == 'OCAngleLayer':
            #     scores = output
            # elif type(output)==tuple and len(output)==2:
            #     scores = F.log_softmax(output[1], dim=1)
            # else:
            #     scores = F.log_softmax(output, dim=1)
                
            batch_scores = output[:, 1].data.cpu().numpy().ravel()
            # add outputs
            utt_list.extend(utt)
            score_list.extend(batch_scores.tolist())

            num_total += key.size(0)
            batch_loss = criterion(output, key)
            dev_loss += batch_loss.item()
        
        dev_loss = dev_loss / num_total

    final_dict = {k:v for k,v in zip(utt_list, score_list)}

    bona_cm = []
    spoof_cm = []
    for key in final_dict:
        if key not in label_dict.keys():
            continue
        if label_dict[key] == 'bonafide' or label_dict[key] == 'genuine':
            bona_cm.append(final_dict[key])
        else:
            spoof_cm.append(final_dict[key])
    bona_cm = np.array(bona_cm).squeeze()
    spoof_cm = np.array(spoof_cm).squeeze()

    if not args.invert:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]
    print(f'Dev time cost: {(time.time() - end)}', flush=True)
    return eer_cm*100, dev_loss

import tqdm
def evaluation(eval_loader, featCal, model, classifier, args):
    print('Evaluating on eval set...', flush=True)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    classifier.eval()
    end = time.time() 
    eval_scores = os.path.join(args.eval_dir, f'epoch{args.eval_epoch}_scores.tsv')
    utt_list = []
    score_list = []
    with open(eval_scores, 'w+') as fh:
        for batch_x, utt_id in tqdm.tqdm(eval_loader):
            # pdb.set_trace()# 关于fream level 验证的代码尚未写好。
            batch_x = batch_x.to(args.device, non_blocking=True)

            feats_len = torch.LongTensor([batch_x.shape[1]]*batch_x.shape[0]).to(args.device, non_blocking=True)
            feats, feats_len = featCal(input_signal=batch_x, length=feats_len)
            feats_len = feats_len.to(args.device, non_blocking=True)
            # if string pair or unet NOT in model name:  -1 means not found
            output = classifier(model(feats, feats_len))

            batch_score = output[:, 1].data.cpu().numpy().ravel()
            utt_list.extend(utt_id)
            score_list.extend(batch_score.tolist()) 
        fh.write(f"filename\tcm-score\n")
        for f, cm in zip(utt_list, score_list):
            fh.write('{}\t{}\n'.format(f, cm))
    print(f'Scores saved to {eval_scores}', flush=True)
    print(f'Eval time cost: {(time.time() - end)}', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and development loop for Conformer ASRU2023: Conformer CM system with ASR/ASV pre-trained model')
    
    # =================general=================
    
    ## model tag and comment
    parser.add_argument('--model_cfg', type=str, help='model config')
    parser.add_argument('--track', type=str, choices=['FAD', 'ASVspoof5'], default='ASVspoof', help='track: FAD or ASVspoof')
    parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved model')
    parser.add_argument('--exp_dir', type=str, default='exp_', help='exp directory of trainning results')
    parser.add_argument('--log_dir', type=str, default='log_', help='log directory of trainning results')
    parser.add_argument('--score_dir', type=str, default='score_', help='score directory of evaluation results')
    ## seed
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    ## device
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--workers', type=int, default=16)
    ## resume
    parser.add_argument('--model_path', type=str, default=None, help='the model to be loaded')
    ## eval
    parser.add_argument('--frame_level', type=str_to_bool, default=False, help='frame level evaluation')
    parser.add_argument('--eval', type=str_to_bool, default=False, help='evaluation only')
    parser.add_argument('--eval_epoch', type=int, default=11, help='the evaluation model chekpoint num.')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size')
    # ================date setting================
    ## dataset
    parser.add_argument('--trn_data_name', type=str, default='train', help='train data name')
    parser.add_argument('--dev_data_name', type=str, default='dev', help='dev data name')
    parser.add_argument('--eval_data_name', type=str, default='eval', help='eval data name')
    parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
    parser.add_argument('--dur_range', default=[5, 5], nargs='+', type=float)
    ## feat choice [logFbankCal logSpecCal logSpecCal_low Raw_Cal]
    parser.add_argument('--feat', type=str, choices=['logFbankCal', 'logSpecCal', 'logSpecCal_low', 'Raw_Cal'], default='logFbankCal', help='feature extraction default: log mel filter bank')
    ## processing
    parser.add_argument('--preemph', type=str_to_bool, default=False, help='pre-emphasis') 
    parser.add_argument('--vad', type=str_to_bool, default=False, help='vad')
    ## augmentation
    parser.add_argument('--data_aug', type=str_to_bool, default=False, help='data augmentation')
    parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
    parser.add_argument('--aug_rate', default=0.7, type=float)
    parser.add_argument('--is_specaug', type=str_to_bool, default=False, help='specAug')
    parser.add_argument('--speed_aug', type=str_to_bool, default=False, help='speed augmentation, 0.9 or 1.1 times of original speed')
    parser.add_argument('--reverb', type=str_to_bool, default=False, help='reverb augmentation')
    parser.add_argument('--offset', type=str_to_bool, default=True, help='offset')
    parser.add_argument('--trim', type=int, default=None, help='trim')
    # ================models================
    ## model
    parser.add_argument('--model', default='ConformerMFA', type=str)
    parser.add_argument('--num_pretrain_layer', default=None, type=int)
    parser.add_argument('--model_pretrain', choices=['ASR', 'ASV', 'None'], default='None', type=str)
    parser.add_argument('--model_asr_ckp', default='ckpt/encoder_small.pkl', type=str)
    parser.add_argument('--model_asv_ckp', default='ckpt/small_model_33.pkl', type=str)
    ## hyper-parameters
    parser.add_argument('--loss', type=str, default='ce', choices=['oc', 'ce'], help='loss function')
    ## front-end
    # parser.add_argument('--embd_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    ## back-end
    parser.add_argument('--classifier', default='Linear', type=str)
    parser.add_argument('--angular_m', default=0.2, type=float)
    parser.add_argument('--angular_s', default=32, type=float)
    ## cost and optimizer
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('--max_step', type=int, default=200000)
    parser.add_argument('--mse_loss', type=str_to_bool, default=False)
    ## faster tricks
    parser.add_argument('--use_amp', type=str_to_bool, default=False, help='amp')
    
    # ================training================
    parser.add_argument('--partial_data', type=int, default=None, help='partial data int')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--warm_up_steps', type=int, default=100, help='warm up steps')
    parser.add_argument('--patience', type=int, default=100, help='patience for lr scheduler')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='min lr for lr scheduler')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epoch')
    parser.add_argument('--model_freeze_epochs', default=0, type=int)
    parser.add_argument('--early_stop', type=int, default=None, help='is so, early stop patience')
    # parser.add_argument('--continue_eval', type=str_to_bool, default=False, help='continue eval')
    ## EER
    parser.add_argument('--invert', type=str_to_bool, default=False, help='invert the score')

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=0, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    args = parser.parse_args()
    main(args)
