# BERT4Rec_AC
飞浆论文复现挑战赛第四期 BERT4Rec

原论文地址 https://dl.acm.org/doi/abs/10.1145/3357384.3357895

原论文代码 https://github.com/FeiSun/BERT4Rec

参考实现 https://github.com/Qdriving/Bert4Rec_Paddle2.0

# 模型描述
BERT4Rec将NLP中的mask language任务迁移到序列推荐问题来，给予了序列推荐一种不同于item2item，left2right的训练范式。

具体来说，对于一条物品序列，以一定的概率p随机mask掉序列中的物品，使用transformer的encoder结构对mask item进行预测。

通过数据增强，完形填空任务的方式使得训练更加充分。

# 复现精度

BERT4Rec论文的一个创新点是将nlp领域完形填空式的任务引入序列推荐 具体就体现在对序列数据的增强上

我们根据原论文和作者开源代码的实现 对不同数据集设置数据增强的参数：mask proportion 0.6 for beauty, 0.2 for ML-1m. Dual factor = 10 

模型参数设置上根据论文作者提供的json文件进行设置，最终复现效果如下：



| Data | Hit@10 | NDCG@10 | MRR|
|:-------:|:-----|--------:|--------:|
|Beauty| 0.312301 | 0.190869 |0.172197 |
|ML-1m| 0.692595| 0.486158 | 0.432563|


# 环境依赖
- 硬件：CPU、GPU
- 框架： 
   - PaddlePaddle >= 2.0.0 
   - Python >= 3.7
        


# 数据生成与数据增强
下载数据放置到 ./dataset/modelnet40_normal_resampled/

```
|--models
  |--rank
    |--dlrm                   # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类            
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # dlrm 核心算法代码，包括 dlrm 组网等
|--tools                      # PaddleRec 工具类
|--LICENSE                    # 项目 LICENSE
|--README.md                  # readme
|--README-old.md              # 原始 readme
|--README_CN.md               # PaddleRec 中文 readme
|--README_EN.md               # PaddleRec 英文 readme
|--run.sh                     # 项目执行脚本(需在 aistudio notebook 中运行)
```
