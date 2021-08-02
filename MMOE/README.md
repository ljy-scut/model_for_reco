# MMOE模型
- 传统的Share-Bottom model不同任务之间是共享参数的，而One-gate MoE 不同的Tower也是共享输入的,
  所以模型对于任务之间的相关性是非常敏感的，任务的相关性越小，差异性越大，模型的性能表现越差。
- MMOE输入到不同的expert，不共享参数，使用Gate给不同的expert不同的权重（根据softmax函数输出的权重），
  Gate的作用类似于attention，然后进行加权输入到不同的塔进行输出，不同的任务使用的不同的Gate。

![Image text](https://github.com/ljy-scut/model_for_reco/blob/master/MMOE/image/model.png)
![Image text](https://github.com/ljy-scut/model_for_reco/blob/master/MMOE/image/formula_1.png)
![Image text](https://github.com/ljy-scut/model_for_reco/blob/master/MMOE/image/formula_2.png)
![Image text](https://github.com/ljy-scut/model_for_reco/blob/master/MMOE/image/result.png)

# MMOE实验
- 数据集：Cenus-Income数据集
- 进行marital和income两个二分类任务
- 改动：
  - 对类别特征进行embedding
  - 将embedding特征和dense特征拼接在一起，过一个两层的DNN,最后输入到MMOE层

# Reference
- MMOE论文：https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-
- https://github.com/drawbridge/keras-mmoe
- https://github.com/zanshuxun/WeChat_Big_Data_Challenge_DeepCTR_baseline
- https://github.com/shenweichen/DeepCTR


