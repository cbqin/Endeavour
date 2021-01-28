<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



# 特征工程

## 特征归一化

数值特征存在量纲，归一化可以使特征都统一到一个大致相同的数值区间。

- 线性函数归一化

   将原始数据映射到 [0,1] ，等比缩放。公式如下：
   $$X_{norm}=\frac{X-X_{min}}{X_{max}-X_{min}}$$

- 零均值归一化

   将原始数据映射到均值为 0 ，方差为 1 的分布上。公式如下：
   $$z=\frac{x-\mu}{\delta}$$

以梯度下降为例，归一化可以使特征的更新速度变得一致，更快地找到最优解。
一般需要使用梯度下降的模型是需要归一化的，而决策树则不适用。