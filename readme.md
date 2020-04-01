Negative Correlated Natural Evolution Strategy
===========================
#### Describution
This repository contains PyTorch implementations of Deep Reinforcement Learning with 'Negative Correlated Natural Evolution Strategy'(NCNES) algorithms.

#### Dependency
- gym0.12.1
- pytorch1.0.1
- python3.6.5

All Dependency can be imported by anaconda environment with environment.yml

#### Usage
```
python main.py 
-game     [Freeway,Enduro,Qbert,Alien...]
-ncpu     default = 40 
-lr_mean  default= 0.2 
-lr_sigma default = 0.1 
-phi      default= 0.0001 
-sigma_init default= 2  
-eva      defalut = 3  
-lam      default = 5 
-mu       default = 15
```

#### File Tree
├── readme.md                   // help  
├── log                         // log  
│   ├── namemark-time-phi-ft    // log  
│   ├── namemark-game-seed.pt   // pytorch saved model 
│   ├── state.txt               // state log 
│   └── train_curve.txt    // log for train curve
├── __init__.py                 // init file  
├── environment.yml             // dependenct Installation file 
├── main.py                     // run  
├── model.py                    // class of neural network (model)  
├── noisetable.py               // class of shared noise table 
├── optimizer.py                // class of gaussian distribution optimizer  
├── preprocess.py               // class of preprocess transform 
├── train.py                    // train and test function  
├── util.py                     // other function  
└── vbn                         // class and function about vitural batch   normalization  

#### Update log
1. 指定随机帧           增加了可修改和查看每次 reward 对应的随机帧
2. 修改权重方式         pytorch 修改权重方式tmp取值改为named_parameters, params.data
3. 修改随机数种子       pytorch随机数种子改为每次修改模型时当前时间取样，env seed, np.random, torch seed改为 time.time()
4. 删除SGD             未采用SGD方式
5. 新增 build_mean     用于建立高斯分布字典，其初始化过程为mean= L + (H-L) *rand
6. 删除 mirror sample   不使用mirror sample noise机制
7. 修改ARGS            folderpath, checkpointname 整合到 ARGS 参数
8. 修改noise获取        get reward atari 改成noise 临时取样再保存,train simulate 也是直接保存 model，noisetable删除
9. 修改optimizer        d,f,fisher 整合到optimizer中
10. 修改logger          logging 修改为同时输出到屏幕和文件
11. 修改

#### Reference
\[1\] Peng Yang, Ke Tang, Xin Yao, "Negatively Correlated Search as a Parallel Exploration Search Strategy", arXiv-https://arxiv.org/abs/1910.07151, 2019

#### Contributing
#### Copyright/Licence
