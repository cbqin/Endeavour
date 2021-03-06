# Word2Vec

Word2vec 包括两个模型：CBOW 和 Skip-Gram 以及两个优化算法：Hierarchical Softmax 和 Negative Sampling。

## CBOW

上下文预测中间词。

<center><img src="./images/cbow.png" width = "200" height = "200"/></center>

### Notations

- $w_i$: 词表 $V$ 中第 $i$ 个单词。
- $\mathcal{V}\in R^{n\times \lvert V \rvert}$: 输入单词矩阵。
- $\mathcal{v_i}$: $\mathcal{V}$ 的第 $i$ 列，$w_i$ 的输入向量表示。
- $\mathcal{U}\in R^{\lvert V \rvert \times n}$: 输出单词矩阵。
- $\mathcal{u_i}$: $\mathcal{U}$ 的第 $i$ 行，$w_i$ 的输出向量表示。

### Steps

1. 上下文用独热编码表示，窗口大小为 $m$: $(x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x^{(c+m)} \in R^{\lvert V \rvert})$
2. 输入向量转换: ($\mathcal{v_{c-m}}=\mathcal{V}x^{(c-m)},...,\mathcal{v_{c-1}}=\mathcal{V}x^{(c-1)},\mathcal{v_{c+1}}=\mathcal{V}x^{(c+1)},...,\mathcal{v_{c+m}}=\mathcal{V}x^{(c+m)}$)
3. 平均: $\hat{v}=\frac{v_{c-m}+...+v_{c-1}+v_{c+1}+...+v_{c+m}}{2m}\in{R^n}$
4. 得分: $z=\mathcal{U}\hat{v}\in{R^{\lvert V \rvert}}$
5. 概率: $\hat{y}=softmax(z)\in{R^{\lvert V \rvert}}$
6. 比较: $\hat{y}\in{R^{\lvert V \rvert}}$ 与 $y\in{R^{\lvert V \rvert}}$

### Object Function

一般选择交叉熵来衡量分布 $\hat{y}$ 和 $y$ 的差异。


$$
\begin{aligned}
H(y,\hat{y}) &= -\sum_{j=1}^{\lvert V \rvert}y_j\log{\hat{y}}
 \\
&=-y_i\log{\hat{y_i}} \\
&= -\log{\hat{y_i}}
\end{aligned}
$$

所以有：

$$
\begin{aligned}
minimize \ J &= -\log{P(w_c|w_{c-m},...,w_{c-1},w_{c+1},w_{c+m})} \\
&= -\log{P(u_c|\hat{v})} \\
&= -\log{\frac{exp(u_c^T\hat{v})}{\sum_{j=1}^{\lvert V \rvert}exp(u_j^T\hat{v})}} \\
&= -u_c^T\hat{v} + \log{\sum_{j=1}^{\lvert V \rvert}exp(u_j^T\hat{v})}
\end{aligned}
$$


## Skip-Gram 

中间词预测上下文中的词。

<center><img src="./images/skip-gram.png" width = "250" height = "200"/></center>


### Notations

- $w_i$: 词表 $V$ 中第 $i$ 个单词。
- $\mathcal{V}\in R^{n \times \lvert V \rvert}$: 输入单词矩阵。
- $\mathcal{v_i}$: $\mathcal{V}$ 的第 $i$ 列，$w_i$ 的输入向量表示。
- $\mathcal{U}\in R^{\lvert V \rvert \times n}$: 输出单词矩阵。
- $\mathcal{u_i}$: $\mathcal{U}$ 的第 $i$ 行，$w_i$ 的输出向量表示。

### Steps

1. 中心词用独热编码表示: $x\in{R^{\lvert V \rvert}}$
2. 输入向量转换: $v_c=\mathcal{V}x\in{R^n}$
3. 得分: $z=\mathcal{U}v_c\in{R^{\lvert V \rvert}}$
4. 概率: $\hat{y}=softmax(z)\in{R^{\lvert V \rvert}}$，$\hat{y}^{c-m},...,\hat{y}^{c-1},\hat{y}^{c+1},...,\hat{y}^{c+m}$ 是与上下文中词对应的概率
5. 比较: 独热编码表示 $y^{c-m},...,y^{c-1},y^{c+1},...,y^{c+m}$ 和对应的概率

### Object Function

$$
\begin{aligned}
minimize\ J &= -\log{P(w_{c-m},...,w_{c-1},w_{c+1},w_{c+m}|w_c)} \\
&= -\log{\prod_{j=0,j \not = m}^{2m}P(w_{c-m+j}|w_c)} \\
&= -\log{\prod_{j=0,j \not = m}^{2m}P(u_{c-m+j}|v_c)}   \\
&= -\log{\prod_{j=0,j \not = m}^{2m} \frac{exp(u_{c-m+j}^Tv_c)}{\sum_{k=1}^{\lvert V \rvert}exp(u_k^Tv_c)}} \\
&= -\sum_{j=0,j \not = m}u_{c-m+j}^Tv_c + 2m\log{\sum_{k=1}^{\lvert V \rvert}exp(u_k^Tv_c)}
\end{aligned}
$$

## Negative Sampling

以上目标函数的每一次更新都要遍历整个词表，当词表很大的时候，计算量就非常大，所以可以负采样少许样本，降低计算量。

此时的目标函数不一样了。考虑一组单词 $(w,c)$ ,分别为是中心词和上下文中的词，用 $P(D=1 \lvert w,c)$ 表示 $(w,c)$ 来自于语料库的概率，而 $P(D=0 \lvert w,c)$ 则表示 $(w,c)$ 来自非语料库的概率，目标函数就是让这两者的概率都尽可能的大。

$$
\begin{aligned}

\theta &= \underset{\theta}{argmax}\prod_{(w,c) \in D}P(D=1|w,c,\theta)\prod_{(w,c) \in \tilde{D}}P(D=0|w,c,\theta) \\
&= \underset{\theta}{argmax}\prod_{(w,c) \in D}P(D=1|w,c,\theta)\prod_{(w,c) \in \tilde{D}}(1-P(D=0|w,c,\theta)) \\
&= \underset{\theta}{argmax}\sum_{(w,c) \in D} \log P(D=1|w,c,\theta) + \sum_{(w,c) \in \tilde{D}} \log (1-P(D=0|w,c,\theta)) \\
&= \underset{\theta}{argmax} \sum_{(w,c) \in D} \log \frac{1}{1+exp(-u_w^Tv_c)} + \sum_{(w,c) \in \tilde{D}} \log (1-\frac{1}{1+exp(-u_w^Tv_c)}) \\
&= \underset{\theta}{argmax} \sum_{(w,c) \in D} \log \frac{1}{1+exp(-u_w^Tv_c)} + \sum_{(w,c) \in \tilde{D}} \log \frac{1}{1+exp(u_w^Tv_c)}
\end{aligned}
$$

则：

$$
J = -\sum_{(w,c) \in D} \log \frac{1}{1+exp(-u_w^Tv_c)} - \sum_{(w,c) \in \tilde{D}} \log \frac{1}{1+exp(u_w^Tv_c)}
$$

在 skip-gram 中，给定中心词 $c$，对于上下文的词 $c-m+j$ 的目标函数是：

$$
-\log \sigma(u_{c-m+j}^T v_c) - \sum_{k=1}^K \log \sigma(-\tilde{u}_k^T v_c)
$$

在 cbow 中，给定上下文向量 $\hat{v}=\frac{v_{c-m}+...+v_{c-1}+v_{c+1}+...+v_{c+m}}{2m}$，对于中心词 $c$ 的目标函数是：

$$
-\log \sigma(u_c^T \hat{v}) - \sum_{k=1}^K \log \sigma(-\tilde{u}_k^T \hat{v})
$$


以上公式中，$\{\hat{u_k} \lvert k=1,2,..,K\}$ 依概率 $P_n(w)$ 采样，一般为一元模型的 $\frac{3}{4}$ 次方：

$$
P_n(w)={P_{unigram}(w)}^{\frac{3}{4}}
$$

## Hierarchical Softmax

Hierarchical softmax 利用二叉树来表示词表和计算概率。每一个叶子节点代表一个词，从根结点到叶子节点有唯一的一条路径。没有输出向量，而除了根结点和叶子节点，树中的节点都对应一个需要学习的向量。输出某一叶子节点的概率为 从根结点到此叶子节点路径出现的概率，所以复杂度由 O($\vert$V$\vert$) 变为 O(log$\vert$V$\vert$)。

<center><img src="./images/binary-tree.png" width = "300" height = "200"/></center>

### Notations

- $L(w)$: 从根结点到叶子节点 $w$ 的节点数
- $n(w,i)$: 路径中第 $i$ 个节点，所以 $n(w,1)$ 是根结点，$n(w,L(w))$ 是叶子节点 $w$ 的父节点
- $v_{n(w,i)}$: 路径中第 $i$ 个节点对应的向量
- $ch(n)$: 对与内部节点 $n$，每次都选择其左孩子（或者右孩子）
- $w_i$: 输入单词

### Object Function

对于输入单词 $w_i$，输出单词为 $w$ 的概率为：

$$
P(w \vert w_i) = \prod_{j=1}^{L(w)-1} \sigma([n(w,j+1)=ch(n(w,j))] \cdot v_{n(w,j)}^T v_{w_i})
$$

并且

$$
[x] = \begin{cases}
    1 &\text{if x is true}  \\
    -1 &\text{otherwise} 
\end{cases}
$$

上式保证了在节点 $n$ 处，有：

$$
\sigma(v_n^T v_{w_i}) + \sigma(-v_n^T v_{w_i}) = 1
$$

而且像原始的 softmax 一样，有：

$$
\sum_{w=1}^{\lvert V \rvert} P(w \vert w_i) = 1
$$

在 cbow 中，$v_{w_i}=\frac{v_{c-m}+...+v_{c-1}+v_{c+1}+...+v_{c+m}}{2m}$；在 skip-gram 中，$v_{w_i}=v_c$，即中心词的输入向量。目标函数为：

$$
\begin{aligned}
minimize \thickspace J &= -\log P(w \vert w_i) \\
&= \prod_{j=1}^{L(w)-1} \sigma([n(w,j+1)=ch(n(w,j))] \cdot v_{n(w,j)}^T v_{w_i})
\end{aligned}
$$

训练时，只用更新从根结点到对应叶子节点路径上的点的向量即可。二叉树一般用 Huffman 树构建，频次高的词用有较小的路径长度，可以加快训练速度。在实际中，hierarchical softmax 对于频次较低的词比较友好，因为不涉及依据频次负采样的问题；negative sampling 对于频次较高的词和低维度向量比较友好。


## SGNS (Skip-Gram with Negative Sampling)

在这一节，作为练习，我们要实现 SGNS 。

### Naive Softmax Loss

$$
V \in R^{V \times d} \\
U \in R^{V \times d}
$$

$$
P(O=o \vert C=c) = \frac{exp(u_o^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)} 
$$

$$
\begin{aligned}
    J_{naive-softmax(v_c,o,U)} &= -\sum_{w \in V} y_w \log(\hat{y}_w) \\
    &= - 1 \times \log P(O=o \vert C=c) \\
    &= -\log P(O=o \vert C=c) \\
    &= -\log \frac{exp(u_o^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)} \\
    &= -\log (\hat{y}_o)
\end{aligned}
$$

目标函数对 $v_c$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{naive-softmax(v_c,o,U)}}{\partial v_c} &= -\frac{\partial (u_o^T v_c)}{\partial v_c} + \frac{\partial (\log \sum_{w \in V} exp(u_w^T v_c))}{\partial v_c} \\
    &= -u_o + \frac{1}{\sum_{w \in V} exp(u_w^T v_c)} \frac{\partial (\sum_{w \in V} exp(u_w^T v_c))}{\partial v_c} \\
    &= -u_o + \sum_{w \in V} \frac{exp(u_w^T v_c) u_w}{\sum_{w \in V} exp(u_w^T v_c)} \\
    &= -u_o + \sum_{w \in V} P(O=w \vert C=c) u_w \\
    &= -u_o + \sum_{w \in V} \hat{y}_w u_w \\
    &= -U^Ty + U^T \hat{y} \\
    &= U^T(\hat{y}-y)
\end{aligned}
$$

目标函数对 $u_w$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{naive-softmax(v_c,o,U)}}{\partial u_w} &= -\frac{\partial (u_o^T v_c)}{\partial u_w} + \frac{\partial (\log \sum_{w \in V} exp(u_w^T v_c))}{\partial u_w} 
\end{aligned}
$$

当 $w \not = o$ 时：

$$
\begin{aligned}
    \frac{\partial J_{naive-softmax(v_c,o,U)}}{\partial u_w} &= -0 + \frac{\partial (\log \sum_{w \in V} exp(u_w^T v_c))}{\partial u_w} \\
    &= \frac{1}{\sum_{w \in V} exp(u_w^T v_c)} \frac{\partial (\sum_{w \in V} exp(u_w^T v_c))}{\partial u_w} \\
    &= \frac{exp(u_w^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)} v_c \\
    &= \hat{y}_w v_c \\
    &= (\hat{y}_w-0)v_c
\end{aligned}
$$

当 $w = o$ 时：

$$
\begin{aligned}
    \frac{\partial J_{naive-softmax(v_c,o,U)}}{\partial u_w} &= -v_c + \frac{\partial (\log \sum_{w \in V} exp(u_w^T v_c))}{\partial u_w} \\
    &= -v_c + \frac{1}{\sum_{w \in V} exp(u_w^T v_c)} \frac{\partial (\sum_{w \in V} exp(u_w^T v_c))}{\partial u_w} \\
    &= -v_c + \frac{exp(u_w^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)} v_c \\
    &= -v_c + \hat{y}_w v_c \\
    &= (\hat{y}_w-1)v_c
\end{aligned}
$$

所以：

$$
\begin{aligned}
    \frac{\partial J_{naive-softmax(v_c,o,U)}}{\partial U} = (\hat{y}-y)v_c^T
\end{aligned}
$$

### Sigmoid

$$
\begin{aligned}
    \sigma(x) &= \frac{1}{1+e^{-x}} \\
    &= \frac{e^x}{1+e^x}
\end{aligned}
$$

$$
\begin{aligned}
    \frac{\partial \sigma(x_i)}{\partial x_i} &= \frac{e^{x_i}(1+e^{x_i})-e^{2x_i}}{(1+e^{x_i})^2} \\
    &= \frac{e^{x_i}}{1+e^{x_i}} \frac{1}{1+e^{x_i}} \\
    &= \sigma(x_i)(1-\sigma(x_i))
\end{aligned}
$$

$$
\begin{aligned}
    \frac{\partial \sigma(x)}{\partial x} &= \left[\frac{\partial \sigma(x_i)}{\partial x_i}\right]_{d \times d} \\
    &= \begin{bmatrix}
        \sigma'(x_1) & 0 & ... & 0 \\
        0 & \sigma'(x_2) & ... & 0 \\
        \vdots & \vdots & \vdots & \vdots \\
        0 & 0 & 0 & \sigma'(x_d) 
       \end{bmatrix} \\
    &= \text{diag}(\sigma^\prime(x))

\end{aligned}
$$


### Negative Sampling Loss

$$
J_{neg-sample(v_c,o,U)} = -\log(\sigma(u_o^T v_c)) - \sum_{k=1}^K \log(\sigma(-u_k^T v_c))
$$

对 $v_c$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{neg-sample(v_c,o,U)}}{\partial v_c} &= -\frac{1}{\sigma(u_o^T v_c)}\sigma(u_o^T v_c)(1-\sigma(u_o^T v_c))u_o + \sum_{k=1}^K \frac{1}{\sigma(-u_k^T v_c)} \sigma(-u_k^T v_c)(1-\sigma(-u_k^T v_c))u_k \\
    &= (\sigma(u_o^T v_c)-1)u_o + \sum_{k=1}^K (1-\sigma(-u_k^T v_c))u_k \\
    &= (\sigma(u_o^T v_c)-1)u_o + \sum_{k=1}^K \sigma(u_k^T v_c)u_k \\
    &= -\sigma(-u_o^T v_c)u_o + \sum_{k=1}^K \sigma(u_k^T v_c)u_k
\end{aligned}
$$

对 $u_o$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{neg-sample(v_c,o,U)}}{\partial u_o} &= -\frac{1}{\sigma(u_o^T v_c)}\sigma(u_o^T v_c)(1-\sigma(u_o^T v_c))v_c \\
    &= (\sigma(u_o^T v_c)-1)v_c \\
    &= -\sigma(-u_o^T v_c)v_c
\end{aligned}
$$

对 $u_k$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{neg-sample(v_c,o,U)}}{\partial u_k} &= \frac{1}{\sigma(-u_k^T v_c)} \sigma(-u_k^T v_c)(1-\sigma(-u_k^T v_c))v_c \\
    &= \sigma(u_k^T v_c)v_c
\end{aligned}
$$

### Skip-Gram Loss

$$
\begin{aligned}
    J_{skip-gram(v_c,w_{t-m},...,w_{t+m},U)} = \sum_{j \in [-t,t],j\not ={0}} J_{neg-sample(v_c,w_{t+j},U)}
\end{aligned}
$$

对 $U$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{skip-gram(v_c,w_{t-m},...,w_{t+m},U)}}{\partial U} = \sum_{j \in [-t,t],j\not ={0}} \frac{\partial J_{neg-sample(v_c,w_{t+j},U)}}{\partial U}
\end{aligned}
$$

对 $v_c$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{skip-gram(v_c,w_{t-m},...,w_{t+m},U)}}{\partial v_c} &= \sum_{j \in [-t,t],j\not ={0}} \frac{\partial J_{neg-sample(v_c,w_{t+j},U)}}{\partial v_c} 
\end{aligned}
$$

对 $v_w (w \not ={c})$ 的偏导：

$$
\begin{aligned}
    \frac{\partial J_{skip-gram(v_c,w_{t-m},...,w_{t+m},U)}}{\partial v_c} &= 0 
\end{aligned}
$$

### Code

基于 cs224n 作业2，见 [word2vec](https://github.com/cbqin/Endeavour/blob/main/notes/03-NLP/01-Vector-Semantics/Word2Vec/word2vec.py).