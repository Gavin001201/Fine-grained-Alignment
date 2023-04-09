## 全部放开，图像与文本侧都使用预训练模型权重，文本损失只考虑非pad的token
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