## 解冻codebook,文本侧前面与图像侧做cross-attention,后面与文本侧编码器的结果做cross-attention
## 生成结果出奇的好，但是量化后图像与文本的组成成分完全一致
## 失败的版本，cross-attention的引入导致信息泄漏，模型没有学习到任何东西
## 下一个版本将进行大量改动

## 改动：
## 1.去掉cross-attention
## 2.文本侧解码器取消跳层连接
## 3.去掉t2t_loss
## 4.