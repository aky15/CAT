#!/bin/bash
# Copyright 2018 Keyu An
# Apache 2.0

# transform raw AISHELL-2 data to kaldi format

. ./path.sh || exit 1;

tmp=
dir=

if [ $# != 3 ]; then
  echo "Usage: $0 <corpus-data-dir> <dict-dir> <tmp-dir> <output-dir>"
  echo " $0 /export/AISHELL-2/iOS/train data/local/dict data/local/train data/train"
  exit 1;
fi

corpus=$3
dict_dir=$1
tmp=$2
dir=$3

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $tmp
mkdir -p $dir

# corpus check
if [ ! -d $corpus ] || [ ! -f $corpus/wav.scp ] || [ ! -f $corpus/text ]; then
  echo "Error: $0 requires wav.scp and trans.txt under $corpus directory."
  exit 1;
fi

# validate utt-key list
awk '{print $1}' $corpus/wav.scp   > $tmp/wav_utt.list
awk '{print $1}' $corpus/text > $tmp/text_utt.list
utils/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/text_utt.list > $tmp/utt.list

# wav.scp
awk -F'\t' -v path_prefix= '{printf("%s\t%s%s\n",$1,path_prefix,$2)}' $corpus/wav.scp > $tmp/tmp_wav.scp
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

#exit 0
# text
python -c "import jieba" 2>/dev/null || \
  (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
utils/filter_scp.pl -f 1 $tmp/utt.list $corpus/text | sort -k 1 | uniq > $tmp/trans.txt
# jieba's vocab format requires word count(frequency), set to 99
awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $tmp/word_seg_vocab.txt
#exit 0
python local/word_segmentation.py $tmp/word_seg_vocab.txt $tmp/trans.txt > $tmp/text
#exit 0
sed -i 's/< sil >/<sil>/g' $tmp/text
sed -i 's/< - >/<->/g' $tmp/text
sed -i 's/< $ >/<$>/g' $tmp/text
sed -i 's/< _ >/<_>/g' $tmp/text
sed -i 's/< % >/<%>/g' $tmp/text
sed -i 's/< # >/<#>/g' $tmp/text

#exit 0
# utt2spk & spk2utt
awk -F'\t' '{print $2}' $tmp/wav.scp > $tmp/wav.list
cp $dir/utt2spk  $tmp/tmp_utt2spk
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
utils/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text utt2spk spk2utt; do
  cp $tmp/$f $dir/$f || exit 1;
done

echo "local/prepare_data.sh succeeded"
exit 0;

