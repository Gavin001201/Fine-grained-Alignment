## 下面是官方的环境配置，但是版本较老，cuda版本大概率会冲突
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