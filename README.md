# BERT4Rec_AC
飞浆论文复现挑战赛第四期 BERT4Rec

原论文地址 https://dl.acm.org/doi/abs/10.1145/3357384.3357895

原论文代码 https://github.com/FeiSun/BERT4Rec

# 模型描述
BERT4Rec将NLP中的mask langua任务迁移到序列推荐问题来，给予了序列推荐一种不同于item2item，left2right的训练范式。

具体来说，对于一条物品序列，以一定的概率p随机mask掉序列中的物品，使用transformer的encoder结构对mask item进行预测。

通过数据增强，完形填空任务的方式使得训练更加充分。
