# GloVe

基于全局共现矩阵的单词向量表示方法。

## Object Function

逐步推导。

### Notations

- $X_{ij}$: 单词 $j$ 出现在单词 $i$ 上下文中的次数
-  $X_{i} = \sum_k X_{ik}$: 单词 $i$ 的上下文中单词出现的总次数
-  $P_{ij} = P(j \vert i) = \frac{X_{ij}}{X_i}$: 单词 $j$ 出现在单词 $i$ 上下文中的概率


### Formulas

| Probability and Ratio |  k=solid   |   k=gas    |  k=water   | k=fashion  |
| :-------------------: | :--------: | :--------: | :--------: | :--------: |
|       P(k\|ice)       | 1.9 x 10-4 | 6.6 x 10-5 | 3.0 x 10-3 | 1.7 x 10-5 |
|      P(k\|steam)      | 2.2 x 10-5 | 7.8 x 10-4 | 2.2 x 10-3 | 1.8 x 10-5 |
| P(k\|ice)/P(k\|steam) |    8.9     | 8.5 x 10-2 |    1.36    |    0.96    |

上表显示的是对于单词 $i$ 和 $j$，单词 $k$ 和哪个单词更相关。由最后一行我们可以得出一些结论：

- 若比值大于1很多，则处于分子的单词和 $k$ 更相关，对应 $k=solid$
- 若比值小于1很多，则处于分母的单词和 $k$ 更相关，对应 $k=gas$
- 若比值在1附近，则 $k$ 与 $i$ 和 $j$ 都相关或者都不相关，分别对应 $k=water$ 和 $k=fashion$

因此，通过概率的比值而不是概率本身去学习词向量可能是更好的方法，因此可以构造如下函数：

$$
F(w_i,w_j,\tilde{w}_k) = \frac{P_{ik}}{P_{jk}} \tag{1}
$$

函数 $F$ 的参数和具体形式不确定。因为要表现出概率的比值，对于向量空间的向量来说，可以用做差来体现，于是得到：

$$
F(w_i-w_j,\tilde{w}_k) = \frac{P_{ik}}{P_{jk}} \tag{2}
$$

上式右侧是数量，而左侧是向量，所以把左侧转换为两个向量内积的形式：

$$
F((w_i-w_j)^T\tilde{w}_k) = \frac{P_{ik}}{P_{jk}} \tag{3}
$$

单词 $i$ 和 $w$ 是可以互换的，对于共现矩阵 $X$ 而言，是对称的，所以不仅要求 $w \leftrightarrow \tilde{w}$，而且 $X \leftrightarrow X^T$。最终的模型在交换下标后应该保持不变。我们可以通过两步达到这个目的。首先，要求函数 $F$ 在加法的实数集到乘法的正实数集是同态的，即：

$$
F((w_i-w_j)^T\tilde{w}_k) = \frac{F(w_i^T\tilde{w}_k)}{F(w_j^T\tilde{w}_k)} \tag{4}
$$

由公式 (3)(4) 得：

$$
F(w_i^T\tilde{w}_k) = P_{ik} = \frac{X_{ik}}{X_i} \tag{5}
$$

函数 $F$ 取 $exp$ 得：

$$
w_i^T\tilde{w}_k = \log P_{ik} = \log(X_{ik}) - \log(X_{i}) \tag{6}
$$

为了对称性，$\log X_{i}$ 与 $k$ 无关，将其改为偏置项 $b_i$，并添加 $\tilde{w}_k$ 的偏置项 $\tilde{b}_k$，得：

$$
w_i^T\tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik}) \tag{7}
$$

上式中，$\log$ 函数在 0 处是发散的，解决这个问题最简便的做法是对 $\log$ 函数的参数做一个加 1 平滑，即 $\log(X_{ik}) \rightarrow \log(X_{ik}+1)$，这样既保持了 $X$ 的稀疏性，又解决了发散的问题。这里对共现矩阵的 $\log$ 函数进行分解的操作和 LSA 的思想接近。然而，式 (7) 还有一个缺陷就是对于共现矩阵的每一项，都是平等对待的，权重都一样，但是出现频率低的或者根本没出现的单词携带的信息是较少的，甚至可以看作是噪声，所以有必要根据频次赋予不同的权重。综上，有下面的目标函数：

$$
J = \sum_{i,j=1}^V f(X_{ij})(w_i^T\tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2 \tag{8}
$$

这就将式 (7) 转化为一个最小二乘问题。其中 $f(X_{ij})$ 是一个权重函数，需满足以下几点：

1. $f(0) = 0$ 。如果 $f$ 是连续函数，那么当 $x \rightarrow 0$ 时，它应该比 $\lim_{x \rightarrow 0}f(x)\log^2x$ 更迅速趋于 0。这其实就不用加 1 平滑了。
2. $f(x)$ 应该是非递减的，这保证了低共现的权重较小。
3. 当 $x$ 较大时，$f(x)$ 也应该相对较小，这保证了高共现的权重不会过大。

以上，选择如下函数：

$$
f(x) = \begin{cases}
   (x/x_{max})^{\alpha} &\text{if x < } x_{max} \\
   1 &\text{otherwise}
\end{cases} \tag{9}
$$

<center><img src="./images/glove_weighting_function.png" width = "200" height = "200"/></center>

$\alpha$ 取 3/4，$x_{max}$ 取 100。有趣的是 word2vec 中负采样的权重函数的幂也是 3/4。

## Implementation

见 [word2vec](https://github.com/cbqin/Endeavour/blob/main/notes/03-NLP/01-Vector-Semantics/GloVe/word2vec.py).