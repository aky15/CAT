#!/usr/bin/env python
# encoding=utf-8
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

from __future__ import print_function
import sys
import jieba
#reload(sys)
#sys.setdefaultencoding('utf-8')

if len(sys.argv) < 3:
  sys.stderr.write("word_segmentation.py <vocab> <trans> > <word-segmented-trans>\n")
  exit(1)

vocab_file=sys.argv[1]
trans_file=sys.argv[2]

jieba.set_dictionary(vocab_file)
#jieba.load_userdict(vocab_file) 
#jieba.add_word('<sil>')
#jieba.suggest_freq('<sil>', True)
#jieba.add_word('<->')
#jieba.add_word('<_>')
#jieba.add_word('<$>')
#jieba.add_word('<%>')
for line in open(trans_file):
  #print(line)
  key,trans = line.strip().split(' ',1)
  words = jieba.cut(trans, HMM=False) # turn off new word discovery (HMM-based)
  new_line = key + ' ' + " ".join(words)
  print(new_line)

