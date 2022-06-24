'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang, Keyu An
Apache 2.0.
This script shows how to excute CTC-CRF neural network training with PyTorch.
'''
from ctc_crf import CTC_CRF_LOSS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import timeit
import os
import sys
import argparse
import json
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from stft import Stft
from specaug import SpecAug
from beamformer_net import BeamformerNet
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from dataset_multich import pad_tensor, SpeechDataset, SpeechDatasetMem, SpeechDatasetPickle, SpeechDatasetMemPickle, PadCollate
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import ctc_crf_base
import math
from torch.utils.tensorboard import SummaryWriter
from utils import save_ckpt, optimizer_to, parse_args
from predict_net import PredictNet
from log_mel import LogMel
from utterance_mvn import UtteranceMVN
from global_mvn import GlobalMVN
import augment

TARGET_GPUS = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))]
gpus = torch.IntTensor(TARGET_GPUS)

class Model(nn.Module):
    def __init__(self, net, idim, hdim, K, n_layers, dropout, lamb, beamforming, spec_aug=True, loss='crf', ah=4, predict=False, right_context=40, wpe=False, p_multi=1, stats_file='stats.npy'):
        super(Model, self).__init__()
        self.stft = Stft(n_fft = 512, win_length = 400, hop_length = 160)
        if wpe:
            # print("use wpe!")
            self.beamformer = BeamformerNet(beamformer_type=beamforming,use_wpe=True)
        else:
            self.beamformer = BeamformerNet(beamformer_type=beamforming)
        #print(idim)
        self.logmel = LogMel(n_mels=idim)
        self.global_mvn = GlobalMVN(stats_file, norm_means=True, norm_vars=True)     
   
        self.net = eval(net)(input_size=idim, output_size=hdim,attention_heads=ah,num_blocks=n_layers,static_chunk_size=0, use_dynamic_chunk=False, use_dynamic_left_chunk=False,causal=False)      
        self.linear = nn.Linear(hdim, K)
        if loss == 'ctc':
            # print("use ctc loss")
            self.loss_fn = WARP_CTC_LOSS()
        else:
            # print("use crf loss")
            self.loss_fn = CTC_CRF_LOSS(lamb=lamb)
        if spec_aug:
            # print("use spec augment!")
            self.specaug = SpecAug(
                apply_time_warp=True,
                apply_freq_mask=True,
                apply_time_mask=True,
                )        
        self.spec_aug = spec_aug
        self.n_layers = n_layers
        self.p_multi = p_multi
        self.predict = predict
        if self.predict:
            self.predict_net = PredictNet(mel_dim=idim, out_len=right_context, hdim=256, rnn_num_layers=3, rnn_dropout=0.1, rnn_residual=True)
            self.criterion = nn.MSELoss(size_average=True)

    def forward_chunk(self, logits, chunk_size, left_context_size, right_context_size):
        if self.predict:
            simu_right_context = self.predict_net(logits.clone(), chunk_size)
        N_chunks = logits.size(1)//chunk_size
        logits = logits.view(logits.size(0)*N_chunks, chunk_size, logits.size(2))
        left_context = torch.zeros(logits.size()[0], left_context_size, logits.size()[2]).to(logits.get_device())    
        
        if left_context_size > chunk_size:
            N = left_context_size//chunk_size
            for idx in range(N):
                left_context[N-idx:, idx*chunk_size:(idx+1)*chunk_size, :] = logits[:-N+idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, :(N-idx)*chunk_size, :] = 0
        else:
            left_context[1:, :, :] = logits[:-1, -left_context_size:, :]        
            left_context[0::N_chunks, :, :] = 0
        
        if right_context_size>0:
            right_context = torch.zeros(logits.size()[0], right_context_size, logits.size()[2]).to(logits.get_device())              
            if right_context_size > chunk_size:
                right_context[:-1, :chunk_size, :] = logits[1:, :, :]
                right_context[:-2, chunk_size:, :] = logits[2:, :right_context_size-chunk_size, :]
                right_context[N_chunks-1::N_chunks, :, :] = 0
                right_context[N_chunks-2::N_chunks, chunk_size:, :] = 0
            else:
                right_context[:-1, :, :] = logits[1:, :right_context_size, :]
                right_context[N_chunks-1::N_chunks, :, :] = 0
            
            if self.predict:
                simu_loss = self.criterion(simu_right_context, right_context.detach())
                if self.training:
                    if np.random.rand() < 0.5:
                        logits_with_context = torch.cat((left_context, logits, simu_right_context), dim=1)
                    else:
                        if np.random.rand() < 0.5:
                            logits_with_context = torch.cat((left_context, logits), dim=1)
                        else:
                            logits_with_context = torch.cat((left_context, logits, right_context), dim=1)
                else:
                    logits_with_context = torch.cat((left_context, logits, simu_right_context), dim=1)
            else:
                if self.training:
                    if np.random.rand() < 0.5:
                        logits_with_context = torch.cat((left_context, logits), dim=1)
                    else:
                        logits_with_context = torch.cat((left_context, logits, right_context), dim=1)
                else:
                    logits_with_context = torch.cat((left_context, logits, right_context), dim=1)
        else:
            logits_with_context = torch.cat((left_context, logits), dim=1)     
        
        if self.training and self.spec_aug:
            logits_with_context, _ = self.specaug(logits_with_context)
        logits_with_context,  _, _, _ = self.net(logits_with_context, torch.full([logits_with_context.size(0)], logits_with_context.size(1)))
        logits = logits_with_context[:, left_context_size//4:(chunk_size + left_context_size)//4, :]
        
        logits = logits.contiguous().view(logits.size(
                    0)//N_chunks, logits.size(1)*N_chunks, -1)
        
        if self.predict:
            return logits, simu_loss
        
        return logits        
       
    def forward_utt(self, logits, input_lengths):
        logits = logits[:,:input_lengths[0],:]
        if self.training and self.spec_aug:
            logits, _ = self.specaug(logits)
        logits,  _, _, _ = self.net(logits, input_lengths)
        return logits

    def forward(self, samples, labels_padded, input_lengths, label_lengths, chunk_size, left_context_size, right_context_size):
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        # print(samples)
        samples = torch.index_select(samples, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [
            labels_padded[i, :x] for i, x in enumerate(label_lengths)
        ]
        labels = torch.cat(label_list)
        
        #print(samples.shape)
        if self.training:
            if np.random.rand() < self.p_multi:    
                # print(samples.shape)
                samples, flens = self.stft(samples, input_lengths)
                max_input_length = int(chunk_size*(math.ceil(float(samples.shape[1])/chunk_size)))    
                samples = map(lambda x: pad_tensor(x, max_input_length, 0), samples)
                samples = torch.stack(list(samples), dim=0)
                N_chunks = samples.size(1)//chunk_size
                samples = samples.view(samples.size(0)*N_chunks, chunk_size, samples.size(2), samples.size(3), samples.size(4))
                # print(samples.shape)
                samples_left_context = torch.zeros(samples.size()[0], left_context_size, samples.size()[2], samples.size(3), samples.size(4)).to(samples.get_device())
                if left_context_size > chunk_size:
                    N = left_context_size//chunk_size
                    for idx in range(N):
                        samples_left_context[N-idx:, idx*chunk_size:(idx+1)*chunk_size, :, :, :] = samples[:-N+idx, :, :, :, :]
                    for idx in range(N):
                        samples_left_context[idx::N_chunks, :(N-idx)*chunk_size, :, :, :] = 0  
                samples_with_context = torch.cat((samples_left_context, samples), dim=1)
                samples_with_context, _ , _ = self.beamformer(samples_with_context,torch.full([samples_with_context.size(0)], left_context_size+chunk_size))
                samples = samples_with_context[:, left_context_size:chunk_size + left_context_size, :]
                #print("1", samples.shape)
                samples = samples.contiguous().view(samples.size(0)//N_chunks, samples.size(1)*N_chunks, samples.size(2), samples.size(3))
                #print("2", samples.shape)
                # print(samples.shape)
            else:
                batch, length, channel = samples.shape
                random_idx = torch.randint(channel,(batch,length,1))
                samples = torch.squeeze(torch.gather(samples,2,random_idx.cuda()),-1)
                samples, flens = self.stft(samples, input_lengths)
                max_input_length = int(chunk_size*(math.ceil(float(samples.shape[1])/chunk_size)))
                samples = map(lambda x: pad_tensor(x, max_input_length, 0), samples)
                samples = torch.stack(list(samples), dim=0)
        else:
            samples, flens = self.stft(samples, input_lengths)
            max_input_length = int(chunk_size*(math.ceil(float(samples.shape[1])/chunk_size)))    
            samples = map(lambda x: pad_tensor(x, max_input_length, 0), samples)
            samples = torch.stack(list(samples), dim=0)
            N_chunks = samples.size(1)//chunk_size
            samples = samples.view(samples.size(0)*N_chunks, chunk_size, samples.size(2), samples.size(3), samples.size(4))
            samples_left_context = torch.zeros(samples.size()[0], left_context_size, samples.size()[2], samples.size(3), samples.size(4)).to(samples.get_device())
            if left_context_size > chunk_size:
                N = left_context_size//chunk_size
                for idx in range(N):
                    samples_left_context[N-idx:, idx*chunk_size:(idx+1)*chunk_size, :, :, :] = samples[:-N+idx, :, :, :, :]
                for idx in range(N):
                    samples_left_context[idx::N_chunks, :(N-idx)*chunk_size, :, :, :] = 0  
            samples_with_context = torch.cat((samples_left_context, samples), dim=1)
            samples_with_context, _ , _ = self.beamformer(samples_with_context,torch.full([samples_with_context.size(0)], left_context_size+chunk_size))
            samples = samples_with_context[:, left_context_size:chunk_size + left_context_size, :]
            #print("1", samples.shape)
            samples = samples.contiguous().view(samples.size(0)//N_chunks, samples.size(1)*N_chunks, samples.size(2), samples.size(3))
        
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        #print(input_amp.shape)
        #print(flens)
        input_feats, _ = self.logmel(input_amp, flens)
        input_feats, _ = self.global_mvn(input_feats, flens)
        
        #print(input_feats.shape)
        #print(input_feats.shape)
        #print(flens)
        if self.predict:
            logits_chunk, predict_loss = self.forward_chunk(input_feats, chunk_size, left_context_size, right_context_size)
        else:
            logits_chunk = self.forward_chunk(input_feats, chunk_size, left_context_size, right_context_size)
        
        logits_utt = self.forward_utt(input_feats, flens)
        logits_chunk = self.linear(logits_chunk)
        logits_chunk = F.log_softmax(logits_chunk, dim=2)
        
        logits_utt = self.linear(logits_utt)
        logits_utt = F.log_softmax(logits_utt, dim=2)
        olens = flens//4
        
        loss_chunk = self.loss_fn(logits_chunk, labels, olens, label_lengths)
        loss_utt = self.loss_fn(logits_utt, labels, olens, label_lengths)
        
        if self.predict:
            return loss_chunk, loss_utt, predict_loss

        return loss_chunk, loss_utt

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.start_rank + gpu
    print("rank", args.rank)
    TARGET_GPUS = [args.gpu]
    gpus = torch.IntTensor(TARGET_GPUS)
    den_path=args.den_path
    #print(den_path)
    ctc_crf_base.init_env(den_path.encode('ascii'), gpus)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    os.makedirs(args.dir + '/board', exist_ok=True)
    writer = SummaryWriter(args.dir +'/board')
    # save configuration
    with open(args.dir + '/config.json', "w") as fout:
        config = {
            "arch": args.arch,
            "output_unit": args.output_unit,
            "hdim": args.hdim,
            "layers": args.layers,
            "dropout": args.dropout,
            "feature_size": args.feature_size,
            "attention_heads": args.ah, 
        }
        json.dump(config, fout)

    epoch = 0   
    model = Model(args.arch, args.feature_size, args.hdim, args.output_unit,
                  args.layers, args.dropout, args.lamb, args.beamforming, args.spec_aug,args.loss, args.ah, args.predict, args.right_context, args.wpe, args.p_multich)
    if args.rank==0:
        print("parameters:", sum(param.numel() for param in model.parameters()))
    #exit()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    min_cv_loss = np.inf
    
    #device = torch.device("cuda:0")
    step = 1
    model.cuda(args.gpu)
    if args.resume_all:
        if args.rank==0:
            print("resume from {}".format(args.pretrained_all_model_path))
        pretrained_dict = torch.load(args.pretrained_all_model_path,map_location='cuda:{}'.format(args.rank))
        epoch = pretrained_dict['epoch']
        model.load_state_dict(pretrained_dict['model'])
        lr = pretrained_dict['lr']
        optimizer.load_state_dict(pretrained_dict['optimizer'])
        optimizer_to(optimizer, torch.device("cuda:{}".format(args.rank)))
        min_cv_loss = pretrained_dict['cv_loss']
        step = pretrained_dict['step']
    else:
        if args.resume_bf:
            if args.rank==0:
                print("resume beamformer from {}".format(args.pretrained_bf_model_path))
            model_dict =  model.state_dict()
            pretrained_dict = torch.load(args.pretrained_bf_model_path,map_location=torch.device('cpu'))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict,strict=True)
        if args.resume_am:
            if args.rank==0:
                print("resume am from {}".format(args.pretrained_am_model_path))
            model_dict =  model.state_dict()
            pretrained_dict = torch.load(args.pretrained_am_model_path,map_location=torch.device('cpu'))
            model_dict.update(pretrained_dict['model'])
            model.load_state_dict(model_dict,strict=True)
            step = pretrained_dict['step']
            if args.rank==0:
                print("step: ",step)
    # model.cuda()
    model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=True,device_ids=TARGET_GPUS)
    #model = nn.DataParallel(model)
    #model.to(device)
    
    if args.wav_aug:
        if args.rank==0:
            print("use wave augment!")
        effect_chain = augment.EffectChain()
        effect_chain.pitch("-q", augment.random_pitch_shift).rate("-q", 16_000)
        if args.reverb:
            if args.rank==0:
                print("reverb !")
            effect_chain.reverb(50, 50, augment.random_room_size).channels()

        if args.add_noise_std>0:
            if args.rank==0:
                print("add noise",args.add_noise_std)
            effect_chain.additive_noise(args.add_noise_std)
        
        effect_chain.time_dropout(max_seconds=50 / 1000)
        effect_chain_runner = augment.ChainRunner(effect_chain)

        if args.pkl:
            tr_dataset = SpeechDatasetPickle(args.data_path + "/tr.pkl",augment=effect_chain_runner)
        else:
            tr_dataset = SpeechDataset(args.data_path + "/tr.hdf5",augment=effect_chain_runner)
    else:
        if args.pkl:
            tr_dataset = SpeechDatasetPickle(args.data_path + "/tr.pkl")
        else:
            tr_dataset = SpeechDataset(args.data_path + "/tr.hdf5")    
    
    tr_sampler = DistributedSampler(tr_dataset)
    tr_sampler.set_epoch(epoch)
    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=PadCollate(),
        sampler=tr_sampler)
    
    if args.pkl:
        data_path = args.data_path + "/cv.pkl"
    else:
        data_path = args.data_path + "/cv.hdf5"    
    if args.pkl:
        cv_dataset = SpeechDatasetPickle(data_path) 
    else:
        cv_dataset = SpeechDataset(data_path)
    cv_sampler = DistributedSampler(cv_dataset)
    cv_dataloader = DataLoader(
        cv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=PadCollate(),
        sampler=cv_sampler)
    
    prev_t = 0
    model.train()
    lr_init = args.lr
    
    while True:
        # training stage
        
        epoch += 1
        tr_sampler.set_epoch(epoch)
        for i, minibatch in enumerate(tr_dataloader):
            if args.rank==0:
                print("training epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            jitter = random.randint(-args.jitter_range, args.jitter_range)*4
            chunk_size = args.chunk_size + jitter
            left_context = args.left_context + jitter*(args.left_context//args.chunk_size)
            right_context = args.right_context
            with autocast(enabled=args.amp):
                if args.predict:
                    loss_chunk, loss_utt, loss_predict = model(logits, labels_padded, input_lengths, label_lengths, chunk_size, left_context, right_context)
                    partial_chunk_loss = torch.mean(loss_chunk.cpu())
                    partial_utt_loss = torch.mean(loss_utt.cpu())
                    loss_predict = loss_predict * args.alpha
                    loss = loss_chunk + loss_utt + loss_predict
                else:
                    loss_chunk, loss_utt = model(logits, labels_padded, input_lengths, label_lengths, chunk_size, left_context, right_context)
                    partial_chunk_loss = torch.mean(loss_chunk.cpu())
                    partial_utt_loss = torch.mean(loss_utt.cpu())
                    loss = loss_chunk + loss_utt
                    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                        print("loss contains nan")
                        sys.stdout.flush()
                        model.zero_grad()
                        optimizer.zero_grad()
                        # torch.cuda.empty_cache()
                        # continue

            if args.loss == 'crf':
                weight = torch.mean(path_weights)
                real_chunk_loss = partial_chunk_loss - weight
                real_utt_loss = partial_utt_loss - weight
            else:
                real_chunk_loss = partial_chunk_loss
                real_utt_loss = partial_utt_loss
            loss = loss/args.accumulation_steps
            loss.backward(loss.new_ones(loss.shape[0]))
            for name, param in model.named_parameters():
                if param.grad is not None:
                     valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                     if not valid_gradients:
                         print("grad contains nan")
                         sys.stdout.flush()
                         model.zero_grad()
                         optimizer.zero_grad()
                         # continue
            if step%args.accumulation_steps == 0:                
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

                lr = lr_init/math.sqrt(args.hdim)*min(1./math.sqrt(step//args.accumulation_steps),(step//args.accumulation_steps)*1./math.sqrt(args.warmup_steps)/args.warmup_steps)           
                adjust_lr(optimizer, lr)
                optimizer.step()
                sys.stdout.flush()
                model.zero_grad()
                optimizer.zero_grad()
 
            t2 = timeit.default_timer()
            writer.add_scalar('training chunk loss',
                        real_chunk_loss.item(),
                        (epoch-1) * len(tr_dataloader) + i)
            writer.add_scalar('training utt loss',
                        real_utt_loss.item(),
                        (epoch-1) * len(tr_dataloader) + i)
            if args.predict:
                if args.rank==0:
                    print("time: {}, tr_real_chunk_loss: {}, tr_real_utt_loss: {}, predict_loss:{} , lr: {}".format(t2 - prev_t, real_chunk_loss.item(), real_utt_loss.item(), loss_predict.mean().item(), optimizer.param_groups[0]['lr']))
            else:
                if args.rank==0:
                    print("time: {}, tr_real_chunk_loss: {}, tr_real_utt_loss: {}, lr: {}".format(t2 - prev_t, real_chunk_loss.item(), real_utt_loss.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
            step += 1
            # exit()
        
        # cv stage
        model.eval()
        cv_losses_sum = []
        predict_losses_sum = []
        count = 0
        for i, minibatch in enumerate(cv_dataloader):
            if args.rank==0:
                print("cv epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            with autocast(enabled=args.amp):
                if args.predict:
                    loss_chunk, loss_utt,  loss_predict = model(logits, labels_padded, input_lengths, label_lengths, args.chunk_size, args.left_context, args.right_context)
                    partial_chunk_loss = torch.mean(loss_chunk.cpu())
                    partial_utt_loss = torch.mean(loss_utt.cpu())
                    loss_predict = loss_predict * args.alpha
                else:
                    loss_chunk, loss_utt = model(logits, labels_padded, input_lengths, label_lengths, args.chunk_size, args.left_context, args.right_context)
                    partial_chunk_loss = torch.mean(loss_chunk.cpu())
                    partial_utt_loss = torch.mean(loss_utt.cpu())
                
            loss_size = logits.size(0)
            count = count + loss_size
            if args.loss == 'crf':
                weight = torch.mean(path_weights)
                real_chunk_loss = partial_chunk_loss - weight
                real_utt_loss = partial_utt_loss - weight
            else:
                real_chunk_loss = partial_chunk_loss
                real_utt_loss = partial_utt_loss
            real_loss_sum = real_chunk_loss * loss_size + real_utt_loss * loss_size 
            cv_losses_sum.append(real_loss_sum.item())
            if args.predict:
                loss_predict = loss_predict.mean().item()
                predict_loss_sum = loss_predict * loss_size
                predict_losses_sum.append(predict_loss_sum)
                if args.rank==0:
                    print("cv_real_chunk_loss: {}, cv_real_utt_loss: {}, loss_predict:{}".format(real_chunk_loss.item(),real_utt_loss.item(), loss_predict))
            else:
                if args.rank==0:
                    print("cv_real_chunk_loss: {}, cv_real_utt_loss: {}".format(real_chunk_loss.item(), real_utt_loss.item()))
            # exit()
        cv_loss = np.sum(np.asarray(cv_losses_sum)) / count
        if args.rank==0:
            print("mean_cv_loss: {}".format(cv_loss))
        if args.predict:
            cv_predict_loss = np.sum(np.asarray(predict_losses_sum)) / count
            if args.rank==0:
                print("mean_predict_loss: {}".format(cv_predict_loss))
        #exit()
        if args.rank==0:
            writer.add_scalar('mean_cv_loss',cv_loss,epoch)
          
            # save model
            save_ckpt({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'cv_loss': cv_loss,
                'step': step
                }, epoch < args.min_epoch or cv_loss <= min_cv_loss, args.dir, "model.epoch.{}".format(epoch))
        
            if epoch < args.min_epoch or cv_loss <= min_cv_loss:
                min_cv_loss = cv_loss                
            else:
                print(
                    "cv loss does not improve, decay the initial learning rate from {} to {}"
                    .format(lr_init, lr_init / 10.0))
                #adjust_lr(optimizer, lr / 10.0)
                lr_init = lr_init / 10.0
                if (lr_init < args.stop_lr):
                    print("learning rate is too small, finish training")
                    break
        # exit()
        model.train()

    ctc_crf_base.release_env(gpus)

def main():
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()

