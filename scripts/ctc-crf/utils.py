import sys
import argparse
import torch
import shutil
import math
import random
import timeit
base_line_batch_size = 256

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_ckpt(state, is_best, ckpt_path, filename):
    torch.save(state, '{}/{}'.format(ckpt_path, filename))
    if is_best:
        #torch.save(state, '{}/best_model'.format(ckpt_path))
        shutil.copyfile('{}/{}'.format(ckpt_path, filename),
                       '{}/best_model'.format(ckpt_path))


def train(model, tr_dataloader, optimizer, epoch, args, logger):
    model.train()
    prev_t = timeit.default_timer()
    for i, minibatch in enumerate(tr_dataloader):
        try:
            features, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            if (features.size(0) < args.gpu_batch_size):
                continue
            model.zero_grad()
            optimizer.zero_grad()

            loss = model(features, labels_padded, input_lengths, label_lengths)
            if args.ctc_crf:
                partial_loss = torch.mean(loss.cpu())
                weight = torch.mean(path_weights)
                real_loss = partial_loss - weight
            else:
                real_loss = torch.mean(loss.cpu())
            loss.backward()
            optimizer.step()

            t2 = timeit.default_timer()
            if i % 200 == 0 and args.rank == 0:
                logger.debug("epoch: {}, step: {}, time: {}, tr_real_loss: {}, lr: {}".format(
                    epoch, i, t2 - prev_t, real_loss.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
        except Exception as ex:
            print("rank {} train exception ".format(args.rank), ex)


def validate(model, cv_dataloader, epoch, args, logger):
    # cv stage
    model.eval()
    cv_total_loss = 0
    cv_total_sample = 0
    with torch.no_grad():
        for i, minibatch in enumerate(cv_dataloader):
            try:
                features, input_lengths, labels_padded, label_lengths, path_weights = minibatch
                loss = model(features, labels_padded,
                             input_lengths, label_lengths)
                if args.ctc_crf:
                    partial_loss = torch.mean(loss.cpu())
                    weight = torch.mean(path_weights)
                    real_loss = partial_loss - weight
                else:
                    real_loss = torch.mean(loss.cpu())
                cv_total_loss += real_loss.item() * features.size(0)
                cv_total_sample += features.size(0)
            except Exception as ex:
                print("rank {} cv exception ".format(args.rank), ex)
        cv_loss = cv_total_loss / cv_total_sample
        if args.rank == 0:
            logger.info("epoch: {}, mean_cv_loss: {}".format(epoch, cv_loss))
    return cv_loss


def train_chunk_model(model, reg_model, tr_dataloader, optimizer, epoch, chunk_size, TARGET_GPUS, args, logger):
    prev_t = 0
    for i, minibatch in enumerate(tr_dataloader):
        try:
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            model.zero_grad()
            optimizer.zero_grad()
            input_lengths = map(lambda x: x.size()[0], logits)
            if sys.version > '3':
                 input_lengths = list(input_lengths)
            
            input_lengths = torch.IntTensor(input_lengths)
            out1_reg, out2_reg, out3_reg = reg_model(
                logits, labels_padded, input_lengths, label_lengths)
            loss, loss_cls, loss_reg = model(logits, labels_padded, input_lengths, label_lengths,
                                             chunk_size, out1_reg.detach(), out2_reg.detach(), out3_reg.detach())

            partial_loss = torch.mean(loss.cpu())
            loss_cls = torch.mean(loss_cls.cpu())
            loss_reg = torch.mean(loss_reg.cpu())

            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight
            loss_cls = loss_cls - weight

            count = min(loss.size(0), len(TARGET_GPUS))
            loss.backward(loss.new_ones(count))

            optimizer.step()
            t2 = timeit.default_timer()
            if i % 200 == 0 and args.rank == 0:
                logger.debug("rank {} epoch:{} step:{} time: {}, tr_real_loss: {},loss_cls: {},loss_reg:{}, lr: {}".format(
                    args.rank, epoch, i, t2 - prev_t, real_loss.item(), loss_cls.item(), loss_reg.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
            torch.cuda.empty_cache()
        except Exception as ex:
            print("rank {} train exception ".format(args.rank), ex)


def validate_chunk_model(model, reg_model, cv_dataloader, epoch, cv_losses_sum, cv_cls_losses_sum, args, logger):
    count = 0
    for i, minibatch in enumerate(cv_dataloader):
        try:
            #print("cv epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch

            input_lengths = map(lambda x: x.size()[0], logits)
            if sys.version > '3':
                 input_lengths = list(input_lengths)
            input_lengths = torch.IntTensor(input_lengths)

            reg_out1, reg_out2, reg_out3 = reg_model(
                logits, labels_padded, input_lengths, label_lengths)
            loss, loss_cls, loss_reg = model(logits, labels_padded, input_lengths, label_lengths,
                                             args.default_chunk_size, reg_out1.detach(), reg_out2.detach(), reg_out3.detach())

            loss_size = loss.size(0)
            count += loss_size

            partial_loss = torch.mean(loss.cpu())
            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight

            loss_cls = torch.mean(loss_cls.cpu())
            loss_cls = loss_cls - weight
            loss_reg = torch.mean(loss_reg.cpu())

            real_loss_sum = real_loss * loss_size
            loss_cls_sum = loss_cls * loss_size

            cv_losses_sum.append(real_loss_sum.item())
            cv_cls_losses_sum.append(loss_cls_sum.item())
        except Exception as ex:
            print("rank {} cv exception ".format(args.rank), ex)
    return count


def adjust_lr(optimizer, origin_lr, lr, cv_loss, prev_cv_loss, epoch, min_epoch):
    if epoch < min_epoch or cv_loss <= prev_cv_loss:
        pass
    else:
        lr = lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_distribute(optimizer, origin_lr, lr, cv_loss, prev_cv_loss, epoch, annealing_epoch, gpu_batch_size, world_size):
    '''
    The hyperparameter setup for the batch size 256
    configuration is the learning rate is set to be 0.1, momentum is
    set as 0.9, and learning rate anneals by p12 every epoch from the
    11th epoch. The training finishes in 16 epochs. Inspired by the
    work proposed in[4], we are able to increase the batch size from
    256 to 2560 without decreasing model accuracy by (1) linearly
    warming up the base learning rate from 0.1 to 1 in the first 10
    epochs and (2) annealing the learning rate by p12 from the 11th
    Epoch. refer Wei Zhang, Xiaodong Cui, Ulrich Finkler, Brian Kingsbury,
    George Saon, David Kung, Michael Picheny "Distributed Deep Learning 
    Strategies For Automatic Speech Recognition"
    '''

    new_lr = lr
    batch_size = gpu_batch_size * world_size
    if epoch < annealing_epoch:
        if batch_size > base_line_batch_size:
            max_lr = (batch_size/base_line_batch_size) * origin_lr
            new_lr = lr + round(max_lr/annealing_epoch, 6)
        else:
            new_lr = origin_lr
    else:
        new_lr = lr / math.sqrt(2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

                        
def parse_args():
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("dir", default="models")
    parser.add_argument(
        "--arch",
        choices=[
            'ConformerEncoder','TransformerEncoder'
        ],
        default='BLSTM')
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--min_epoch", type=int, default=5)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--merge_ratio", type=float, default=1)
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_size", type=int, default=83)
    parser.add_argument("--data_path")
    parser.add_argument("--den_path")
    parser.add_argument("--lr", type=float,default=1)
    parser.add_argument("--stop_lr", type=float,default=0.1)
    parser.add_argument("--ah", type=int,default=4)
    parser.add_argument("--spec_aug", type=str2bool,default=True)
    parser.add_argument("--wav_aug", type=str2bool,default=True)
    parser.add_argument("--predict", type=str2bool,default=False)
    parser.add_argument("--chunk_size", type=int, default=40)
    parser.add_argument("--left_context", type=int, default=40)
    parser.add_argument("--right_context", type=int, default=40)
    parser.add_argument("--loss", choices=['ctc','crf'],default='crf')
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--pkl", action="store_true")
    parser.add_argument("--amp", action="store_true",
        help="Enable auto mixed precision training.")
    parser.add_argument("--alpha", type=float,default=10)
    parser.add_argument("--accumulation_steps", type=int, default=5)
    parser.add_argument("--jitter_range", type=int, default=5)
    parser.add_argument("--dynamic_batch", type=str2bool, default=True)
    parser.add_argument("--resume_all", type=str2bool, default=False)
    parser.add_argument("--resume_bf", type=str2bool, default=False)
    parser.add_argument("--resume_am", type=str2bool, default=False)
    parser.add_argument("--pretrained_bf_model_path")
    parser.add_argument("--pretrained_am_model_path")
    parser.add_argument("--pretrained_all_model_path")
    parser.add_argument("--beamforming", type=str, default="mvdr")
    parser.add_argument("--wpe", type=str2bool,default=False)
    parser.add_argument("--reverb", type=str2bool,default=False)
    parser.add_argument("--add_noise_std", type=float,default=1)
    parser.add_argument("--p_multich", type=float,default=0.5)
    parser.add_argument('--dist_url', type=str, default=None,
                        help='Url used to set up distributed training. It must be the IP of device running the rank 0, \
                        and a free port can be opened and listened by pytorch distribute framework.')
    parser.add_argument('--world_size', type=int, default=0, help='world size must be set to indicate the number of ranks you \
                        will  start to train your model. If the real rank number is less then the value of \
                        --world_size it will be blocked on function init_process_group. Rather than that if a rank \
                        number grater then the value of world_size was given the function init_process_group will \
                        throw a exception to terminate the training. So you need make sure the world size is equal \
                        to the ranks number.')
    parser.add_argument("--start_rank", type=int, default=0, help='This value was used to specify the start rank on the device. \
                        For example, if you have 3 gpu and 3 ranks on device 1 and the 4 gpu and 4 ranks on \
                        device 2. The device 1 has rank 0, then the device 2 must start from rank 3.So you must \
                        set --start_rank 3 on device 2. ')

    parser.add_argument('--gpu_batch_size', default=32, type=int,
                        help='This value was depended on the memory size of your gpu, the biger the value was set, \
                        the higher training speed you can get. The total batch size of your training was \
                        (world size) * (batch size per gpu). The max batch size must not grater then 2560, \
                        or rather the training will not be convergent.')
    args = parser.parse_args()

    return args
