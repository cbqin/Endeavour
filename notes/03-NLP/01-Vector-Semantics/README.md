# Vector Semantics and Embeddings

## Distributional Hypothesis

**分布假设**。单词分布的相似度和单词意义的相似度的关系。

## Lexical Semantics

**词汇语义**。语言学中从多个方面考察单词的语义。

- Lemmas and Senses

  > 在形态学和词典编纂中，引理是一组单词的规范形式，词典形式或引用形式。  

  对于每一种引用形式，都可以有多种语义，称为一词多义，每一种语义称作一个 **word scene** 。

- Word Synonymy

  **同义**。从 scene 的角度来讲，如果两个单词的语义之一是一致的，那么可以说这两个单词是同义的。从 word 的角度看，同义的一个更正式的定义可以是：如果两个单词可以在任何句子中相互替换而不改变句子的真实条件，则该两个单词是同义词。

  但是，也许没有真正的同义词，因为词汇的语义的差异总是和在语言学上的形态的差异相关联。比如，*water* 和 $h_2o$ 都是水的意思，但是 $h_2o$ 一般出现在科学相关的文献，而 *water* 更适合于平常的文本中，这种类型的区别也是语义的一部分，所以说同义真实的意思是大概意思一样，并非完全一致。或许可以这么想：在已知已经存在一个词可以描述想表达的意思，那么为什么还要创造一个同样语义的新词？原因可能是旧词的语义和想表达的语义有些许差别，需要一个新词来表示。

- Word Similarity

  **相似**。相似性关注的是单词本身而不是单词的语义。比如 dog 和 cat。

- Word Relatedness

  **关联性**。关联性不要求相似，更关注单词之间的关系、联系。比如 water 和 cup，两者不相似，但是和同一事件相关--喝水。一种较为常见的关系是 **semantic field** 。semantic field 是指一组单词，覆盖特定的语义域并且彼此之间具有结构化关系。比如，对于 hospital 这个 semantic field，可以包含 surgeon, scalpel, nurse, anesthetic, hospital 这些词。Semantic field 与 topic model 相关，二者都可以无监督的学习文本中的主题结构，非常有用。

  这么说来，常说的主题以及主题之间的关系，其实就是单词之间的关联性和结构。

- Semantic Frames and Roles

  Semantic frames 和 semantic field 相似，指的是表示特定事件的不同观点和参与者的一组单词。比如在商业交易中，有 buy，sell，pay 等动作，每个动作也代表着一个角色，分别是 buyer，seller，money。

- Connotation

  **蕴涵**或者**情感意义**。褒义、贬义或者积极、消极，在加一个中立。情感分析也是 nlp 中一个重要的方向。
  
  如何衡量一个词所蕴涵的情感？Osgood 从三个方面来给单词打分：

  - valence: the pleasantness of the stimulus，刺激的愉悦感
  - arousal: the intensity of emotion provoked by the stimulus，被刺激所引起的感情强度
  - dominance: the degree of control exerted by the stimulus，刺激所施加的控制程度

  通过对一个单词从以上三个方面的衡量，就可以得到一个三维空间中的点，这是历史上关于向量语义的第一次表述，突破性的想法💡！！

## Vector Semantics

**向量语义**。实例化分布假设，具有相似上下文的具有相似的含义，根据单词在文本中的分布学习单词的含义。单词的含义也称为词嵌入，即 Embedding，原因是将其嵌入到某个向量空间中。这种获得单词表示的方法是属于表示学习（Presentation Learning），区别于特征工程中人工创造的表示。

### Co-occurrence Matrix

基本上，向量的获得都基于共现矩阵的想法，主要有两种矩阵：term-document matrix 和 term-term matrix。

#### Documents as vectors

在信息检索中，要召回文档，那么如何表示文档呢？文档由单词组成，我们可以用文档中各个单词出现的频次来表示文档，相似的文档拥有相似的单词分布。

|         |  doc_1  |  doc_2  |  doc_3  |  doc_4  |
| :-----: | :-----: | :-----: | :-----: | :-----: |
|  word_1 |    1    |    0    |    7    |    11   |
|  word_2 |   114   |    80   |    62   |    89   |
|  word_3 |    36   |    58   |    1    |    4    |
|  word_4 |    20   |    15   |    2    |    3    |

对于每一列，就是相应文档的向量表示，维度是 |V| ，词表的大小，并且这个向量是稀疏的。

#### Words as vectors: document dimensions

在上表中，如果按行来看，每一行都可以看作是一个单词的向量表示，维度是 |D|，语料库文档的数目，并且这个向量也是稀疏的。相似的词也倾向于出现在相似的文档中。Term-document matrix 既可以用来表示文档也可以用来表示单词。

#### Words as vectors: word dimensions

可不可以用词来衡量词呢？显然可以，这就是 Term-term matrix 的想法，在一个小窗口内，两个词共现的频次。

|         |  aardvark  |  ...  |  computer  |  data  |  result  |  pie  |  sugar  |  ...  |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
|  cherry |    0    |    ...    |    2    |   8   |   9   |   442   |   25   |   ...   |
| strawberry |   0   |   ...   |    0   |   0   |   1   |   60   |   19   |   ...   |
| digital |    0   |   ...   |   1670    |   1683   |   85   |   5   |   4   |   ...   |
| information |    0   |   ...   |   3325    |   3982   |   378   |   5   |   13   |   ...   |

每一个向量的维度是 |V| ，也是稀疏。

### TF-IDF

共现矩阵，不管是 term-document 还是 term-term ，其中每一项都是原始频次。原始频次并不能很好地衡量词之间的关系。对于一个词来说，如果有一个词和它的共现频率很高，那么一般情况下这个词对衡量它是起重要作用的；但是，如果这个词过于频繁的出现，那么这个词对于衡量其共现的词就没那么重要了。解决这个问题的方法有两个：TF-IDF 和 PPMI。

TF-IDF 基于 term-document matrix。

- Term Frequency

  $$tf_{t,d}=log(count(t,d)+1)$$

- Inverse Doucument Frequency

  $$idf_t=log(\frac{N}{df_t})$$

- TF-IDF

  $$w_{t,d}=tf_{t,d} \times idf_t$$

### Positive Pointwise Mutual Information (PPMI)

互信息衡量两个变量间的依赖程度，数值上是点间互信息的期望。直观上，互信息度量 X 和 Y 共享的信息：它度量知道这两个变量其中一个，对另一个不确定度减少的程度。在 NLP 中，可以用点间互信息来度量两个词的相关程度。

$$PPMI(w,c)=max(log\frac{P(w,c)}{P(w)P(c)},0)$$

罕见词倾向于高PMI值。

### Word2vec

见 word2vec.md

## Something About Embeddings

### Visualizing

### Semantics Properties

### Bias