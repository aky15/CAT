以aishell-4数据集为例，讲解多通道端到端模型的训练和测试流程。
构建分母图、构建解码WFST等步骤与单通道模型训练类似，此处不再赘述。

### 数据准备
+ data_prep.py

  根据标注，将句子划分为无说话人混叠的片段。
  
+ local/dump_pcm.sh

  对每一通道的语音，提取pcm特征。

### 训练多通道端到端模型 

+ [optional] 如需使用预训练的单通道模型作初始化，则需先使用local/dump_fbank.sh提取单通道fbank特征，再使用单通道训练的脚本，训练单通道模型。

+ beamformer_net.py中定义了基于TF mask估计的波束成形器。默认的波束成形器是MVDR beamformer。

+ log_mel.py中定义了对数mel谱变换。

+ global_mvn.py(预先提取全局的均值和方差统计特征，用于归一化。均值和方差统计特征提取方式可见collect_stats.py)和utterance_mvn.py(对每句计算各自的均值和方差统计特征)定义了两种不同的mvn方式。前者一般用于流式识别，后者用于离线识别。

+ train_chunk_predict_unified_multich_dist.py是多通道端到端模型的训练流程。
