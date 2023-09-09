## 解冻codebook,文本侧前面与图像侧做cross-attention,后面与文本侧编码器的结果做cross-attention
## test1: 交换 Q, K, V, 图像做 Q，文本做 K，V,其余不变