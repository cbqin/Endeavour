# Word2Vec

Word2vec 包括两个模型：CBOW 和 Skip-Gram 以及两个优化算法：Hierarchical Softmax 和 Negative Sampling。

## CBOW

上下文预测中间词。

<center><img src="./images/cbow.png" width = "200" height = "200"/></center>

### Notations

- $w_i$: 词表 $V$ 中第 $i$ 个单词。
- $\mathcal{V}\in R^{n\times|V|}$: 输入单词矩阵。
- $\mathcal{v_i}$: $\mathcal{V}$ 的第 $i$ 列，$w_i$ 的输入向量表示。
- $\mathcal{U}\in R^{|V|\times n}$: 输出单词矩阵。
- $\mathcal{u_i}$: $\mathcal{U}$ 的第 $i$ 行，$w_i$ 的输出向量表示。

### Steps

1. 上下文用独热编码表示，窗口大小为 $m$: $(x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x^{(c+m)} \in R^{|V|})$
2. 输入向量转换: ($\mathcal{v_{c-m}}=\mathcal{V}x^{(c-m)},...,\mathcal{v_{c-1}}=\mathcal{V}x^{(c-1)},\mathcal{v_{c+1}}=\mathcal{V}x^{(c+1)},...,\mathcal{v_{c+m}}=\mathcal{V}x^{(c+m)}$)
3. 平均: $\hat{v}=\frac{v_{c-m}+...+v_{c-1}+v_{c+1}+...+v_{c+m}}{2m}\in{R^n}$
4. 得分: $z=\mathcal{U}\hat{v}\in{R^{|V|}}$
5. 概率: $\hat{y}=softmax(z)\in{R^{|V|}}$
6. 比较: $\hat{y}\in{R^{|V|}}$ 与 $y\in{R^{|V|}}$

### Object Function

一般选择交叉熵来衡量分布 $\hat{y}$ 和 $y$ 的差异。


$$
\begin{aligned}
H(y,\hat{y}) &= -\sum_{j=1}^{|V|}y_j\log{\hat{y}}
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
&= -\log{\frac{exp(u_c^T\hat{v})}{\sum_{j=1}^{|V|}exp(u_j^T\hat{v})}} \\
&= -u_c^T\hat{v} + \log{\sum_{j=1}^{|V|}exp(u_j^T\hat{v})}
\end{aligned}
$$


## Skip-Gram 

中间词预测上下文中的词。

<center><img src="./images/skip-gram.png" width = "200" height = "200"/></center>

### Notations

- $w_i$: 词表 $V$ 中第 $i$ 个单词。
- $\mathcal{V}\in R^{n\times|V|}$: 输入单词矩阵。
- $\mathcal{v_i}$: $\mathcal{V}$ 的第 $i$ 列，$w_i$ 的输入向量表示。
- $\mathcal{U}\in R^{|V|\times n}$: 输出单词矩阵。
- $\mathcal{u_i}$: $\mathcal{U}$ 的第 $i$ 行，$w_i$ 的输出向量表示。

### Steps

1. 中心词用独热编码表示: $x\in{R^{|V|}}$
2. 输入向量转换: $v_c=\mathcal{V}x\in{R^n}$
3. 得分: $z=\mathcal{U}v_c\in{R^{|V|}}$
4. 概率: $\hat{y}=softmax(z)\in{R^{|V|}}$. $\hat{y}_{c-m},...,\hat{y}_{c-1},\hat{y}_{c+1},\hat{y}_{c+m}$ 是与上下文中词对应的概率
5. 比较: 独热编码表示 $y^{c-m},...,y^{c-1},y^{c+1},...,y^{c+m}$ 和对应的概率

### Object Function

$$
\begin{aligned}
minimize\ J &= -\log{P(w_{c-m},...,w_{c-1},w_{c+1},w_{c+m}|w_c)} \\
&= -\log{\prod_{j=0,j \not = m}^{2m}P(w_{c-m+j}|w_c)} \\
&= -\log{\prod_{j=0,j \not = m}^{2m}P(u_{c-m+j}|v_c)}   \\
&= -\log{\prod_{j=0,j \not = m}^{2m} \frac{exp(u_{c-m+j}^Tv_c)}{\sum_{k=1}^{|V|}exp(u_k^Tv_c)}} \\
&= -\sum_{j=0,j \not = m}u_{c-m+j}^Tv_c + 2m\log{\sum_{k=1}^{|V|}exp(u_k^Tv_c)}
\end{aligned}
$$

## Negative Sampling

以上目标函数的每一次更新都要遍历整个词表，当词表很大的时候，计算量就非常大，所以可以负采样少许样本，降低计算量。

此时的目标函数不一样了。考虑一组单词 $(w,c)$ ,分别为是中心词和上下文中的词，用 $P(D=1|w,c)$ 表示 $(w,c)$ 来自于语料库的概率，而 $P(D=0|w,c)$ 则表示 $(w,c)$ 来自非语料库的概率，目标函数就是让这两者的概率都尽可能的大。

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
-\log \delta(u_{c-m+j}^T v_c) - \sum_{k=1}^K \log \delta(-\tilde{u}_k^T v_c)
$$

在 cbow 中，给定上下文向量 $\hat{v}=\frac{v_{c-m}+...+v_{c-1}+v_{c+1}+...+v_{c+m}}{2m}$，对于中心词 $c$ 的目标函数是：

$$
-\log \delta(u_c^T \hat{v}) - \sum_{k=1}^K \log \delta(-\tilde{u}_k^T \hat{v})
$$


以上公式中，$\{\hat{u_k}|k=1,2,..,K\}$ 依概率 $P_n(w)$ 采样，一般为一元模型的 $3/4$ 次方：

$$
P_n(w)={P_{unigram}(w)}^{\frac{3}{4}}
$$

## Hierarchical Softmax