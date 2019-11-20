## 4. 文本和序列的深度模型

### 嵌入

- 文本嵌入模型的一大问题在于很多可以用来判断语句主题的关键词汇（如retinopathy）出现频率很低，而 Ah, Oh之类的语气词出现频率却较高
- 第二个问题在于对于同义词之间关系的学习，以方便共享权重

通过上下文训练一个embedding模型来将word映射到低纬向量，既可以解决稀疏性问题，又可以通过计算距离来判断词之间的关系

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-16.png)

### Word2Vec

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-17.png)

中间词的Vec ($V_{Fox}$)来预测Window内的上下文，使用对数回归模型，训练的目的就是通过对一个预料库进行训练后，能够对词进行有效的embedding

而计算Vec之间的相似度是使用cos相似而非L2距离，这是因为Vec的长度与意思并无明显关系，所以采用角度来避免长度的影响

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-18.png)

完整的Word2Vec模型:

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-19.png)

Sampled softmax: 如果所有词都进行参加softmax分类，将会是非常复杂的，因此对于非目标词进行采样，来降低计算复杂度，同时不会损失性能

> 在论文中，作者指出指出对于小规模数据集，选择5-20个negative words会比较好，对于大规模数据集可以仅选择2-5个negative words。

在negative words的选取中，采用一元分布模型 (unigram distribution)来选择:
$$
P(w_i) = \frac{f(w_i)^\frac{3}{4}}{\sum^n_{j=0}(f(w_j)^\frac{3}{4})}
$$
$P(w_i)$为选择该单词来BP的概率，$f(w_i)$为单词出现的频次，3/4根据经验设置

word2vec模型有两类：

- Skip-gram: 单个词汇为input，上下文为label
- [CBOW](http://arxiv.org/abs/1301.3781) (Continuous Bag of Words): 上下文词汇为input，单个词汇为label



### RNN & LSTM

图像信号用卷积来建立领域关系，而文本等数据的领域关系建立则用到RNN，结构如下图所示：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-20.png)

然而RNN过长的连接会使得反向传播效果变差，出现梯度消失或者梯度爆炸问题（在CNN中使用残差连接解决了深层网络反向传播的问题），对于该问题，常见的做法是：

- 梯度爆炸：采用梯度剪裁 (gradient clipping)，当梯度增大到某个阈值时调低步长
- 梯度消失：在RNN结构下，使用长短时记忆结构来解决这一问题



LSTM主要包含三个门控结构，输入门 (write)，输出门 (forget)，遗忘门 (forget)

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-21.png)

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-22.png)

门控函数的实现是利用sigmoid激活使得变量值在[0, 1]之间，同时保证连续可导，而对于输入变量的处理还使用了tanh进行激活，范围在[-1, 1]之间，关于0对称，且在0附近梯度大，方便梯度下降，下图为LSTM详细结构，左侧为输入们，中间为遗忘门，右侧为输出门

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-23.png)

具体的公式推导见下图：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-24.png)

在训练的过程中，L2正则化和dropout同样适用于LSTM，注意是用在输入或输出层，而非递归链接层

**LSTM的应用**

- 多对一：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-25.png)

将可变长度的序列输入为固定长度的向量

- 一对多：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-26.png)

采用Beam搜索的方法可以进行连续的预测，将固定长度的序列输入为可变长度的向量

Beam搜索利用input给出t时刻预测，按照树状结构，将可能的t时刻输出作为输入预测t+1时刻，以此类推，可以利用两层树结构的条件概率来避免只用一个预测输入带来的较大误差，在树拓展的过程中可对低概率分支进行剪枝来降低计算量

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-28.png)

- 多对多：

![](https://github.com/Dinghow/Udacity_Deep_Learning/raw/master/Notes_zh/img/dl-27.png)

结合前两种即可实现不定长序列之间的变换，如文字翻译，语音识别等场景，结合CNN将image处理为vector，还可以实现image caption