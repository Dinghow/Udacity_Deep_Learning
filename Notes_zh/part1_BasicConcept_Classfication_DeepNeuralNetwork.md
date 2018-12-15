因为之前已经有过Google那套ML的笔记，所以很多重复的部分就没有再写了，算是做了一些DL上的补充吧，其实严格意义上来说，Google那个课程应该算DL的入门课程，机器学习的很多非神经网络的算法：K-means、决策树这些都没有讲。Whatever，希望对你也有一些帮助～

# 0. 杂记

### 关于吸引盆(basin of attraction)

多层神经网络的初始化隐层不能简单置0的原因，因为0很容易陷进一个非常浅的吸引盆，意味着局部最小值非常大：

因而，如何避免一开始就吸到一个倒霉的超浅的盆中呢，答案是权值初始化。为了统一初始化方案，通常将输入缩放到[−1,1][−1,1]

经验规则给出，$W\sim Uniform(-\frac{1}{\sqrt{N}},\frac{1}{\sqrt{N}})$ ，Uniform为均匀分布。

Bengio组的Xavier在2010年推出了一个更合适的范围，能够使得隐层Sigmoid系函数获得最好的激活范围。[[Glorot10\]](http://deeplearning.cs.cmu.edu/pdfs/1111/AISTATS2010_Glorot.pdf)

对于Log-Sigmoid：   

$$[-4*\frac{\sqrt{6}}{\sqrt{LayerInput+LayerOut}},4*\frac{\sqrt{6}}{\sqrt{LayerInput+LayerOut}}]$$



对于Tanh-Sigmoid：

$[\frac{\sqrt{6}}{\sqrt{LayerInput+LayerOut}},\frac{\sqrt{6}}{\sqrt{LayerInput+LayerOut}}]$

### 关于预训练

目前关于Pre-Training的最好的理解是，它可以让模型分配到一个很好的初始搜索空间，按照[[Erhan09, Sec 4.2\]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_ErhanMBBV.pdf)中说法：

The advantage of pre-training could be that it puts us in a region of parameter space

where basins of attraction run **deeper** than when picking starting parameters

at random. The advantage would be due to a better optimization.

预训练的做法：

**逐层贪婪训练。无监督预训练（unsupervised pre-training）**即训练网络的第一个隐藏层，再训练第二个，最后用这些训练好的网络参数值作为整个网络参数的初始值。  **无监督学习--->参数初始值；监督学习--->fine-tuning，即训练有标注样本**。经过预训练最终能得到比较好的局部最优解

### 关于全连接层

全连接层的每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的。

全连接层(fully connected layers,FC),在整个卷积神经网络中起到”分类器”作用。如果说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空间的话，全连接层则起到”将学到的分布式特征表示”映射到样本标记空间的作用。 

### 关于池化

为了描述大的图像，一个很自然的想法就是对不同位置的特征进行聚合统计，例如，人们可以计算图像一个区域上的某个特定特征的平均值 (或最大值)。这些概要统计特征不仅具有低得多的维度 (相比使用所有提取得到的特征)，同时还会改善结果(不容易过拟合)。这种聚合的操作就叫做池化 (pooling)，有时也称为平均池化或者最大池化 (取决于计算池化的方法)。

![](http://ufldl.stanford.edu/wiki/images/0/08/Pooling_schematic.gif)

# 1. 深度学习基础概念

### 1.1  杂谈

- 神经网络的许多工作是在八九十年代完成的
- 二十一世纪头十年，机器学习领域几乎不涉及到神经网络
- 2009年，神经网络被用于语音识别；2012年被用于CV
- 神经网络复兴的原因：
  - 大量的数据
  - GPU的算力提升



### 1.2 分类

- 线性模型参数的计算

  ![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-8.png)

  线性模型 $Y = WX + B$

  Y的维度位10\*1，故B的维度也是10\*1，而输入的X是28\*28的矩阵，通过WX要转换为10\*1的维度，故W的维度位28\*28\*10

- softmax

  - 将输入数据同时放大，softmax的输出将更靠近0或1
  - 将输入数据同时缩小，输出的分布更均匀

- 多项式逻辑回归分类法

  ![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-1.png)

- 独热编码

交叉熵(Cross-Entropy)：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-2.png)

- 模型性能评估：

  - 训练集 ： 用于训练模型的子集。
  - 验证集：数据集的一个子集，从训练集分离而来，用于调整[**超参数**](https://developers.google.cn/machine-learning/crash-course/glossary#hyperparameter)

  > 从狭义来讲，验证集没有参与梯度下降的过程，也就是说是没有经过训练的；但从广义上来看，验证集却参与了一个“人工调参”的过程，我们根据验证集的结果调节了迭代数、调节了学习率等等，使得结果在验证集上最优。

  - 测试集 ： 用于测试训练后模型的子集。

- 验证集的大小：通常情况越大越好，模型泛化的准确率也会越高，验证集越大其accuracy也越有实际反应模型泛化能力的意义

  - 针对较小数据集采用交叉验证(Cross Validation)是个不错的解决方法，但是最好的方法仍然是获取更多的数据

- 随机梯度下降：

  - 动量(momentum): 

    ```
    v = momentum * v - learning_rate * dx # integrate velocity
    x += v # integrate position
    ```

    代码中v指代速度，其计算过程中有一个超参数momentum，称为动量（momentum）。虽然名字为动量，其物理意义更接近于摩擦，其可以降低速度值，降低了系统的动能，防止石头在山谷的最底部不能停止情况的发生。动量的取值范围通常为[0.5, 0.9, 0.95, 0.99]，一种常见的做法是在迭代开始时将其设为0.5，在一定的迭代次数（epoch）后，将其值更新为0.99

    **在实践中，一般采用SGD+momentum的配置**，相比普通的SGD方法，这种配置通常能极大地加快收敛速度

    AdaGrad就是SGD的优化版，采用了momentum防止过拟合，且自动设置了学习速率下降，通常其准确率比SGD低

  - 学习速率下降: 较小的学习速率更利于收敛

  - SGD中的超参数：

  ![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-3.png)



# 2. 深度神经网络

#### 隐藏层

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-4.png)

**注意**： 以上描述的是一个“两层”神经网络

1. 第一层由一组 X 的权重和偏差组成并通过 ReLU 函数激活。 这一层的输出会提供给下一层，但是在神经网络的外部不可见，因此被称为*隐藏层*。
2. 第二层由隐藏层的权重和偏差组成，隐藏层的输入即为第一层的输出，然后由 softmax 函数来生成概率。

#### 反向传播

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-5.png)

#### 正则化

> **正则化**方法是在训练数据不够多时，或者over training时，常常会导致过拟合（overfitting）。这时向原始模型引入额外信息，以便防止过拟合和提高模型泛**化**性能的一类方法的统称。

#### Dropout

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-6.png)

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-7.png)

通过随机选择舍弃掉相同量的激活值（并对保留值进行放大），使得神经网络不再能依赖于任何给定的激活值，因而能更好地防止overfitting