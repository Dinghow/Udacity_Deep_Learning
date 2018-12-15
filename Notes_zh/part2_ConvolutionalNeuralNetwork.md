## 3. 卷积神经网络

### 灰度图

使用灰度图的原因是避免颜色对分类造成影响

### 统计不变性(Statistical Invariants)

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-9.png)

当两种输入可以获得同样的信息（如猫的位置与分类并没有关系），则应该共享权重，并利用这些输入来训练对应的同一权重

> **参数共享(Parameter Sharing)**
>
> 应用参数共享可以大量减少参数数量，参数共享基于一个假设：如果图像中的一点（x1, y1）包含的特征很重要，那么它应该和图像中的另一点（x2, y2）一样重要。换种说法，我们把同一深度的平面叫做**深度切片(depth slice)**（(e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55])），那么同一个切片应该共享同一组权重和偏置。我们仍然可以使用梯度下降的方法来学习这些权值，只需要对原始算法做一些小的改动， 这里**共享权值的梯度是所有共享参数的梯度的总和**。
>
> 我们不禁会问为什么要权重共享呢？一方面，重复单元能够对特征进行识别，而不考虑它在可视域中的位置。另一方面，权值共享使得我们能更有效的进行特征抽取，因为它极大的减少了需要学习的自由变量的个数。通过控制模型的规模，卷积网络对视觉问题可以具有很好的泛化能力。

### 卷积网络基本概念

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-10.png)

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-11.png)

通过卷积操作逐步挤压空间维度，增大深度，通过深度信息来表达复杂的语义，如下图将3个特征图映射到了K个输出深度。

使用k个filter来进行卷积，输出深度就为k，fliter的通道数必须和特征图的通道数相等，下列图(From CS231n)就解释了这个过程：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-14.png)

> **DIP（数字图像处理）相关：**
>
> 时域上的卷积对应频域相乘。图像与卷积核进行卷积，相当于频域信息进行筛选。（图像中的边缘和轮廓属于是高频信息，图像中某区域强度的综合考量属于低频信息）



![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-12.png)

> **空间排列（Spatial arrangement）**
>
> 一个输出单元的大小有以下三个量控制：**depth**, **stride** 和 **zero-padding**。
>
> - **深度(depth)** : 顾名思义，它控制输出单元的深度，也就是filter的个数，连接同一块区域的神经元个数。又名：**depth column**
> - **步幅(stride)**：它控制在同一深度的相邻两个隐含单元，与他们相连接的输入区域的距离。如果步幅很小（比如 stride = 1）的话，相邻隐含单元的输入区域的重叠部分会很多; 步幅很大则重叠区域变少。
> - **补零(zero-padding)** ： 我们可以通过在输入单元周围补零来改变输入单元整体大小，从而控制输出单元的空间大小。

### 高阶的卷积网络设计

#### 1. 池化(Pooling)

池化的方法：

- Max Pooling
- Average Pooling



池化的意义：

- 减少参数，通过对 Feature Map 降维，有效减少后续层需要的参数
- 增强网络的抗扰动作用



池化的缺点：

- 引入了另外更多的超参数，如池化区域尺寸(pooling size)、池化步幅(pooling stride)



常见的卷积网络设计：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-13.png)

#### 2. 1*1卷积

使用1*1卷积主要有两个作用：

- 降/升维：通过k个1*1的filter去做卷积，可将图片转化为k维（均只指最高维）
- 加入非线性



#### 3. Inception结构

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/note/img/dl-15.png)

使用多种结构的卷积核并将卷积结果进行复合