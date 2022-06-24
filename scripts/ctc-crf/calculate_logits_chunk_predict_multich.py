import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import kaldi_io
import argparse
import json
import math
from wenet.transformer.encoder import ConformerEncoder
from train_chunk_predict_unified_multich import str2bool
from predict_net import PredictNet
from stft import Stft
from beamformer_net import BeamformerNet
from log_mel import LogMel
from utterance_mvn import UtteranceMVN
from global_mvn import GlobalMVN
from pathlib import Path


class Model(nn.Module):
    def __init__(self, net, idim, hdim, K, n_layers, dropout, chunk_size, ah, beamforming, wpe=False, predict=False, right_context=40,  stats_file='stats.npy', hop_length = 160):
        super(Model, self).__init__()
        self.stft = Stft(n_fft = 512, win_length = 400, hop_length = hop_length)
        if wpe:
            print("use wpe!")
            self.beamformer = BeamformerNet(beamformer_type=beamforming,use_wpe=True)
        else:
            self.beamformer = BeamformerNet(beamformer_type=beamforming)
        self.logmel = LogMel(n_mels=idim)
        self.global_mvn = GlobalMVN(stats_file,norm_means=True, norm_vars=True)  
        self.net = eval(net)(input_size=idim, output_size=hdim,attention_heads=ah,num_blocks=n_layers,static_chunk_size=0, use_dynamic_chunk=False, use_dynamic_left_chunk=False,causal=False)
        self.linear = nn.Linear(hdim, K)
        self.predict = predict
        if self.predict:
            self.predict_net = PredictNet(mel_dim=idim, out_len=right_context, hdim=256, rnn_num_layers=3, rnn_dropout=0.1, rnn_residual=True)
        self.hl = hop_length       

    def forward_utt(self, samples, input_lengths, chunk_size, left_context_size, right_context_size):
        samples, flens = self.stft(samples, input_lengths)
        samples, _ , _ = self.beamformer(samples,flens)
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, flens)
        input_feats, _ = self.global_mvn(input_feats, flens)        

        input_feats = input_feats[:,:flens[0],:]
        logits,  _,_,_ = self.net(input_feats, flens)
        netout = self.linear(logits)
        netout = F.log_softmax(netout, dim=2)

        return netout

    def forward_chunk_pcm_with_left_context(self, samples, input_lengths, chunk_size, left_context_size, right_context_size):
        # STFT transform
        max_input_length = int(self.hl*chunk_size*(math.ceil(float(samples.shape[1])/(chunk_size*self.hl))))    
        samples = map(lambda x: pad_tensor(x, max_input_length, 0), samples)
        samples = torch.stack(list(samples), dim=0)
        N_chunks = samples.size(1)//(chunk_size*self.hl)
        samples = samples.view(samples.size(0)*N_chunks, chunk_size*self.hl, samples.size(2))
        # add left context for each chunk. Here we simulate the streaming setting.  In real streming settings, the current chunk should be cached and used as the left context for the next chunk.
        left_context = torch.zeros(samples.size()[0], left_context_size*self.hl, samples.size(2)).to(samples.get_device())
        if left_context_size > chunk_size:
            N = left_context_size//chunk_size
            for idx in range(N):
                left_context[N-idx:, idx*chunk_size*self.hl:(idx+1)*chunk_size*self.hl, :] = samples[:-N+idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, :(N-idx)*chunk_size*self.hl, :] = 0
        else:
            left_context[1:, :, :] = samples[:-1, -left_context_size:, :]
            left_context[0::N_chunks, :, :] = 0
        
        samples_with_context = torch.cat((left_context, samples), dim=1)
        
        samples_with_context, flens = self.stft(samples_with_context[:,:-1,:], torch.full([samples_with_context.size(0)], (left_context_size+chunk_size)*self.hl-1)) #[1, 34543, 8] 16k
        
        # beamforming and feature extraction
        samples_with_context, _ , _ = self.beamformer(samples_with_context,torch.full([samples_with_context.size(0)], left_context_size+chunk_size))
        samples = samples_with_context[:, left_context_size:chunk_size + left_context_size, :]
        # samples = samples.contiguous().view(samples.size(0)//N_chunks, samples.size(1)*N_chunks, samples.size(2), samples.size(3))        
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, torch.full([samples.size(0)], chunk_size))
        input_feats, _ = self.global_mvn(input_feats, torch.full([samples.size(0)], chunk_size))        
              
        # NN forward
        left_context = torch.zeros(input_feats.size()[0], left_context_size, input_feats.size()[2]).to(input_feats.get_device())
        if left_context_size > chunk_size:
            N = left_context_size//chunk_size
            for idx in range(N):
                left_context[N-idx:, idx*chunk_size:(idx+1)*chunk_size, :] = input_feats[:-N+idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, :(N-idx)*chunk_size, :] = 0
        else:
            left_context[1:, :, :] = input_feats[:-1, -left_context_size:, :]
            left_context[0::N_chunks, :, :] = 0
        
        # add right context for each chunk. Here we simulate the streaming setting. 
        # In real streming settings, we 
        # 1) wait for the future frames if we use real future frames as right context. 
        # 2) do not use future frames. 
        # 3) used simulated future frames. In this case, we should first run the predict_net chunk by chunk to obtain the simulated frames. 
        if right_context_size>0:
            if self.predict:        
                predict_right_context = self.predict_net(input_feats.reshape(input_feats.shape[0]//N_chunks,input_feats.shape[1]*N_chunks,-1).clone(), chunk_size)
                logits_with_context = torch.cat((left_context, input_feats, predict_right_context), dim=1)
            else:
                right_context = torch.zeros(input_feats.size()[0], right_context_size, input_feats.size()[2]).to(input_feats.get_device())
                right_context[:-1, :, :] = input_feats[1:, :right_context_size, :]
                logits_with_context = torch.cat((left_context, input_feats, right_context), dim=1)
        else:
            logits_with_context = torch.cat((left_context, input_feats), dim=1)
       
        # NN forward
        logits_with_context,_,_,_ = self.net(logits_with_context, torch.full([logits_with_context.size(0)], logits_with_context.size(1)))
        logits = logits_with_context[:, left_context_size//4:(chunk_size + left_context_size)//4, :]
        
        # reshape the chunk output into utterance again.
        logits = logits.contiguous().view(logits.size(
                 0)//N_chunks, logits.size(1)*N_chunks, -1)
        netout = self.linear(logits)
        netout = F.log_softmax(netout, dim=2)
       
        return netout
    
    def forward_chunk_pcm_without_context(self, samples, input_lengths, chunk_size, left_context_size, right_context_size):
        # STFT transform
        max_input_length = int(self.hl*chunk_size*(math.ceil(float(samples.shape[1])/(chunk_size*self.hl))))    
        samples = map(lambda x: pad_tensor(x, max_input_length, 0), samples)
        samples = torch.stack(list(samples), dim=0)
        N_chunks = samples.size(1)//(chunk_size*self.hl)
        samples = samples.view(samples.size(0)*N_chunks, chunk_size*self.hl, samples.size(2))
       
        samples, flens = self.stft(samples[:,:-1,:], torch.full([samples.size(0)], chunk_size*self.hl-1)) #[1, 34543, 8] 16k
        
        # beamforming and feature extraction
        samples, _ , _ = self.beamformer(samples,torch.full([samples.size(0)], chunk_size))
   
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, torch.full([samples.size(0)], chunk_size))
        input_feats, _ = self.global_mvn(input_feats, torch.full([samples.size(0)], chunk_size))        
              
        # NN forward
        left_context = torch.zeros(input_feats.size()[0], left_context_size, input_feats.size()[2]).to(input_feats.get_device())
        if left_context_size > chunk_size:
            N = left_context_size//chunk_size
            for idx in range(N):
                left_context[N-idx:, idx*chunk_size:(idx+1)*chunk_size, :] = input_feats[:-N+idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, :(N-idx)*chunk_size, :] = 0
        else:
            left_context[1:, :, :] = input_feats[:-1, -left_context_size:, :]
            left_context[0::N_chunks, :, :] = 0
        
        # add right context for each chunk. Here we simulate the streaming setting. 
        # In real streming settings, we 
        # 1) wait for the future frames if we use real future frames as right context. 
        # 2) do not use future frames. 
        # 3) used simulated future frames. In this case, we should first run the predict_net chunk by chunk to obtain the simulated frames. 
        if right_context_size>0:
            if self.predict:        
                predict_right_context = self.predict_net(input_feats.reshape(input_feats.shape[0]//N_chunks,input_feats.shape[1]*N_chunks,-1).clone(), chunk_size)
                logits_with_context = torch.cat((left_context, input_feats, predict_right_context), dim=1)
            else:
                right_context = torch.zeros(input_feats.size()[0], right_context_size, input_feats.size()[2]).to(input_feats.get_device())
                right_context[:-1, :, :] = input_feats[1:, :right_context_size, :]
                logits_with_context = torch.cat((left_context, input_feats, right_context), dim=1)
        else:
            logits_with_context = torch.cat((left_context, input_feats), dim=1)
       
        # NN forward
        logits_with_context,_,_,_ = self.net(logits_with_context, torch.full([logits_with_context.size(0)], logits_with_context.size(1)))
        logits = logits_with_context[:, left_context_size//4:(chunk_size + left_context_size)//4, :]
        
        # reshape the chunk output into utterance again.
        logits = logits.contiguous().view(logits.size(
                 0)//N_chunks, logits.size(1)*N_chunks, -1)
        netout = self.linear(logits)
        netout = F.log_softmax(netout, dim=2)
       
        return netout

def pad_tensor(t, pad_to_length, dim):
    pad_size = list(t.shape)
    pad_size[dim] = pad_to_length - t.size(dim)
    return torch.cat([t, torch.zeros(*pad_size).type_as(t)], dim=dim)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument(
        "--arch",
        choices=[
            'BLSTM', 'LSTM', 'VGGBLSTM', 'VGGLSTM', 'LSTMrowCONV', 'TDNN_LSTM',
            'BLSTMN'
        ],
        default='BLSTM')
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--stats_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=40)
    parser.add_argument("--left_context", type=int, default=40)
    parser.add_argument("--right_context", type=int, default=40)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--feature_size", type=int, default=80)
    parser.add_argument("--ah", type=int, default=4)
    parser.add_argument("--model", type=str)
    parser.add_argument("--causal", type=str2bool, default=False)
    parser.add_argument("--predict", type=str2bool,default=False)
    parser.add_argument("--beamforming", type=str, default="mvdr")
    parser.add_argument("--wpe", type=str2bool,default=False)
    parser.add_argument("--nj", type=int)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as fin:
            config = json.load(fin)
            args.arch = config['arch']
            args.feature_size = config['feature_size']
            args.hdim = config['hdim']
            args.output_unit = config['output_unit']
            args.layers = config['layers']
            args.dropout = config['dropout']
    model = Model(args.arch, args.feature_size, args.hdim, args.output_unit,
                  args.layers, args.dropout, args.chunk_size, args.ah, args.beamforming, args.wpe, args.predict, args.right_context, args.stats_file)
    
    model_dict = torch.load(args.model)
    model.load_state_dict(model_dict['model'])
    '''
    model_dict =  model.state_dict()
    pretrained_bf_model_path='estimator_0.5814.pkl'
    print("resume beamformer from {}".format(pretrained_bf_model_path))
    pretrained_dict = torch.load(pretrained_bf_model_path,map_location=torch.device('cpu'))
    model_dict.update(pretrained_dict)    
    model.load_state_dict(model_dict,strict=True)
    pretrained_am_model_path='best_model_aishell_unified_global_cmvn_predict'
    print("resume am from {}".format(pretrained_am_model_path))
    pretrained_dict = torch.load(pretrained_am_model_path,map_location=torch.device('cpu'))['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict,strict=True)
    '''
    model.eval()
    model.cuda()
    n_jobs = args.nj
    writers = []
    write_mode = 'w'
    if sys.version > '3':
        write_mode = 'wb'

    for i in range(n_jobs):
        writers.append(
            open('{}/decode.{}.ark'.format(args.output_dir, i + 1),
                 write_mode))

    with open(args.input_scp) as f:
        lines = f.readlines()
   
    for i, line in enumerate(lines):
        utt, feature_path = line.split()
        feature = kaldi_io.read_mat(feature_path)
        input_lengths = torch.IntTensor([feature.shape[0]])
        feature = torch.from_numpy(feature[None])
        N_chunks = math.ceil(input_lengths/args.chunk_size)
        input_length = args.chunk_size*N_chunks
        feature = pad_tensor(feature, input_length,1)
        feature = feature.cuda()
        with torch.no_grad():
            #netout = model.forward_chunk_pcm_with_left_context(feature, input_lengths, args.chunk_size, args.left_context, args.right_context)
            netout = model.forward_chunk_pcm_without_context(feature, input_lengths, args.chunk_size, args.left_context, args.right_context)
        r = netout.cpu().data.numpy()
        r[r == -np.inf] = -1e16
        r = r[0]
        kaldi_io.write_mat(writers[i % n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
