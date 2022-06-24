#!/bin/bash

# Copyright 2020-2021 Tsinghua University, Author: Keyu An
# Apache 2.0.

# This script implements CTC-CRF training on Aishell4 dataset.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh
# Begin configuration section.
stage=8
stop_stage=8
data=$(readlink -f data)
# End configuration section
. utils/parse_options.sh

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  python data_prep.py || exit 1;
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  local/prepare_data.sh data/local/dict_phn data/local/train data/train || exit 1;
  local/prepare_data.sh data/local/dict_phn data/local/dev   data/dev   || exit 1;
  local/prepare_data.sh data/local/dict_phn data/local/eval  data/eval  || exit 1;
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for setname in train dev eval; do
     local/dump_pcm.sh --nj 32 --cmd "${train_cmd}" --format flac data/${setname} exp/dump_pcm/$setname /mnt/nas_data/AISHELL-4/pcm/$setname
  done
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  local/aishell4_prepare_phn_dict.sh || exit 1;
  ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
       data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
  local/aishell4_train_lms.sh data/local/dict_phn/lexicon.txt data/train/text  data/local/lm_phn || exit 1;
  local/aishell4_decode_graph.sh data/local/lm_phn data/lang_phn data/lang_phn_test || exit 1;
fi

data_tr=data/train
data_cv=data/dev

if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
  echo "convert text_number finished"

  # prepare denominator
  cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number

  mkdir -p data/den_meta
  chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst || exit 1 
  ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"

  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
  echo "prepare weight finished"
fi

if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  mkdir -p data/pickle_filtered
  python ctc-crf/convert_to_pickle.py $data_cv/feats.scp $data_cv/text_number $data_cv/weight data/pickle_filtered/cv.pkl 160000 || exit 1
  python ctc-crf/convert_to_pickle.py $data_tr/feats.scp $data_tr/text_number $data_tr/weight data/pickle_filtered/tr.pkl 160000 || exit 1
fi

if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  python collect_stats.py
fi

arch=ConformerEncoder
beamforming=mvdr
batch_size=1
loss=crf
chunk_size=40
left_context=80
right_context=40
lr=1
stop_lr=0.01
min_epoch=50
warm_up=25000
hdim=256
ah=4
layers=12
predict=true
alpha=100
spec_aug=true
wav_aug=true
accumulation_steps=1
jitter_range=5
dynamic_batch=false
resume_am=true
resume_bf=true
p_multi=0.5
dir=exp/${arch}_b${batch_size}_accum_${accumulation_steps}_${loss}_lr${lr}_stop_lr_${stop_lr}_chunk_size_${chunk_size}_jitter_range_${jitter_range}_left_context${left_context}_right_context${right_context}_min_epoch${min_epoch}_predict_${predict}_alpha${alpha}_p_multi${p_multi}_resume_am_${resume_am}_resume_bf_${resume_bf}
output_unit=$(awk '{if ($1 == "#0")print $2 - 1 ;}' data/lang_phn/tokens.txt)

if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "nn training."
    python3 ctc-crf/train_chunk_predict_unified_multich_dist.py \
        --feature_size=80 \
        --warmup_steps=$warm_up \
        --arch=$arch \
        --output_unit=$output_unit \
        --lamb=0.01 \
        --data_path=data/pickle_filtered \
        --batch_size=$batch_size \
        --accumulation_steps=$accumulation_steps \
        --loss=$loss \
        --lr=$lr \
        --stop_lr=$stop_lr \
        --clip_grad \
        --hdim=$hdim \
        --ah=$ah \
        --layers=$layers \
        --min_epoch=$min_epoch \
        --chunk_size=$chunk_size \
        --jitter_range=$jitter_range \
        --left_context=$left_context \
        --right_context=$right_context \
        --predict=$predict \
        --alpha=$alpha \
        --spec_aug=$spec_aug \
        --dynamic_batch=$dynamic_batch \
        --resume_am=$resume_am \
        --pretrained_am_model_path=best_model_aishell_unified_global_cmvn_predict_new_feat \
        --resume_bf=$resume_bf \
        --pretrained_bf_model_path=estimator_0.5814.pkl \
        --beamforming=$beamforming \
        --den_path=data/den_meta/den_lm.fst \
        --wav_aug=$wav_aug \
        --p_multich=$p_multi \
        --pkl \
        --world_size=10 \
        --start_rank=0 \
        --gpu_batch_size=16 \
        $dir
fi

stats_file=stats.npy
if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  for set in eval; do
    CUDA_VISIBLE_DEVICES=0 \
    ctc-crf/decode_chunk_predict_multich.sh --stage 0 --cmd "$decode_cmd" --nj 20 --acwt 1.0 \
      data/lang_phn_test data/$set data/${set}/feats.scp $dir/decode_${chunk_size}_${left_context}_${right_context} $chunk_size $left_context $right_context $hdim $ah $layers $predict $stats_file
  done
fi
