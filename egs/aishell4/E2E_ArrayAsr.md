以aishell-4数据集为例，讲解多通道端到端模型的训练和测试流程。
构建分母图、构建解码WFST、流式/非流式一体化训练等步骤与单通道模型训练类似，此处不再赘述。

### 数据准备
+ data_prep.py 根据标注，将句子划分为无说话人混叠的片段。
+ local/dump_pcm.sh 对每一通道的语音，提取pcm特征。
  

### 特征提取与预处理

+ stft.py中定义了短时傅里叶变换。
+ log_mel.py中定义了对数mel谱变换。
+ global_mvn.py和utterance_mvn.py定义了两种不同的mvn方式。前者预先提取全局的均值和方差统计特征，用于归一化(均值和方差统计特征提取方式可见collect_stats.py)，一般用于流式识别；后者对每句计算各自的均值和方差统计特征，一般用于离线识别。

### 数据增广

训练中我们采用了两种数据增广方式：
+ 波形增广(wavaug):对PCM特征进行增广，包括pitch扰动，添加噪音等。
+ 频谱增广(specaug):对log mel谱进行增广，包括添加mask，time warp等。


### 加载预训练模型 
+ [optional] 加载预训练的前后端一体化模型，设置resume_all为true，并指定前后端一体化模型的位置 pretrained_all_model_path。
+ [optional] 如需使用预训练的单通道模型对后端作初始化，则需先使用local/dump_fbank.sh提取单通道fbank特征，再使用单通道训练的脚本，训练单通道模型。加载时设置resume_am为true，并指定预训练的后端的位置 pretrained_am_model_path。
+ [optional] 如需使用预训练的波束成形器对前端作初始化，则需先训练前端模型 [https://github.com/funcwj/chime4-nn-mask](https://github.com/funcwj/chime4-nn-mask), 加载时设置resume_bf为true，并指定预训练的前端的位置 pretrained_bf_model_path。

### 训练多通道端到端模型 
+ 默认的前端是一个neural beamformer。beamformer_net.py中定义了基于TF mask估计的波束成形器。默认的波束成形器是MVDR beamformer，此处参考了espnet的实现。
+ 默认的后端是conformer，此处参考了wenet的实现。
+ train_chunk_predict_unified_multich_dist.py是多通道端到端模型的训练流程。训练中会以p_multi的概率进行前后端一体化训练，以1-p_multi的概率绕过前端，随机选择某一通道语音送入后端，仅优化后端。其他训练设置与单通道训练类似。
