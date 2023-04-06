## 这个版本加入了文本侧的代码，对文本侧进行梯度反向传播，无交叉量化，codebook向文本侧靠拢，图像侧使用预训练权重，与完全放开只差交叉量化
```
conda env create -f environment.yaml
conda activate taming
```
#### freeze.yml是我导出来的我所用到的依赖包
## 运行指令
```
bash train.sh
```
#### 使用单卡运行时注意不要丢掉 --gpus 0, 这里的逗号