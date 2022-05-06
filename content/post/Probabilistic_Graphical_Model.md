---
title: "概率图模型"
date: "2021-12-10"
draft: true
markup: mmark
---

# 生成式模型和判别式模型  
## 需要解决的问题  
首先，我们需要解决的是通过我们的样本数据中统计和学习到一种规律，然后来解决给定新的特征$x$，我们能够通过这种规律来获得该特征$x$对应的标签$y$。那么，使用概率，我们是在求解条件概率$P(y|x)$。也就是说，我们期望得到一个模型参数，使得能够通过该模型参数计算出给定特征$x$的条件概率$P(y|x)$，来获得对应的标签$y$。  

## 间接求解  
我们不直接对条件概率$P(Y|X)$进行建模，而是通过联合概率$P(X,Y)$，来间接的求解。
首先，通过条件概率公式
$$
P(Y,X) = P(Y)P(X|Y)
$$
我们可以将联合概率分解成$P(Y)$和$P(X|Y)$。
然后我们可以通过贝叶斯公式
$$
P(Y|X) = \frac{P(Y)P(X|Y)}{P(X)}
$$
来间接求解$P(Y|X)$。
其中，边缘概率$P(X)$可以通过$P(X)P(X|Y)$计算得来：
$$
P(X) = \sum_iP(Y=y_i)P(X|Y=y_i)
$$
通过最大后验概率估计，我们的优化目标为
$$
argmax P(Y)P(X|Y)
$$  

## 直接求解  
我们通过对条件概率$P(Y|X)$进行建模，我们的优化目标就是该条件概率。  

## 区别  
间接求解就是生成式模型，而直接求解则是判别式模型。
判别式模型是学习到了一个决策边界，通过该边界来划分新样本的类别。
生成式模型是所有变量的全概率模型，所以可以模拟（“生成”）所有变量的值。通过估计使生成概率最大的特征来获取标签。  

# 生成式模型  
## 朴素贝叶斯  
朴素贝叶斯是生成式模型，也就是对联合概率$P(Y,X)$建模。然后通过贝叶斯公式间接求解$P(Y|X)$。那么对于样本集：
$$
P(Y=y_k,x_1……x_n) = \frac{P(Y=y_k)P(x_1……x_n|Y=y_k)}{\sum_jP(Y=y_j)P(x_1……x_n|Y=y_j)}
$$
通过对特征$x$的独立性假设，我们可以进行改写
$$
P(x_1……x_n|Y=y_k) = \prod_i^nP(x_i|Y=y_k)
$$
那么
$$
P(Y=y_k,x_1……x_n) = \frac{P(Y=y_k)\prod_i^nP(x_i|Y=y_k)}{\sum_jP(Y=y_j)\prod_i^nP(x_i|Y=y_k)}
$$
最后，我们通过最大后验概率估计，优化目标即为
$$
argmax P(Y=y_k)\prod_i^nP(x_i|Y=y_k)
$$  

## 隐马可夫模型  
HMM是解决序列问题，通过对马尔可夫假设，即标签$y_i$除了依赖特征$x_i$，还依赖上一个标签$y_{i-1}$（方向性），那么联合概率即
$$
P(y_i,y_{i-1},x_i) = P(y_i|y_{i-1})P(x_i|y_i)
$$
对于一组序列，联合概率可以写为
$$
P(y_i……y_{n},x_i……x_{n}) =\prod_{i=1}^n P(y_i|y_{i-1})P(x_i|y_i)
$$

---

假设标签$y$的取值空间为$S$，数目为$N$，特征$x$的取值空间为$K$，数目为$M$。那么上式中分别为：  
1. 我们将$P(y_i|y_{i-1})$称之为转移概率，即上一个标签值转移到这一个标签值的概率。那么可以构建一个转移概率矩阵
    $$
    \begin{split}
    & A =\{a_{ij}\} \\
    & a_{ij} = P(y_t = S_i | y_{t-1} = S_j), 1 \leq i,j \leq N  \\
    & a_{ij} \geq 0 \\
    & \sum_{j=1}^Na_{ij} = 1
    \end{split}
    $$

---

2. 我们将$P(x_i|y_i)$称之为发射概率，即通过标签值观察到特征值的概率。那么可以构建一个发射概率矩阵
    $$
    \begin{split}
    & B =\{b_j(k)\} \\
    & b_j(k) = P(x_t = K_k | y_t = S_j), 1 \leq k \leq M,1 \leq j \leq N  \\
    & b_j(k) \geq 0 \\
    & \sum_{k=1}^Mb_j(k) = 1
    \end{split}
    $$

---

3. 我们在构建一个标签值的初始概率分布$\pi = \{\pi_i\}$，作为初始的标签值概率
    $$
    \begin{split}
    & \pi_i = P(y_1 = S_i), 1 \leq i \leq N  \\
    & \pi_i \geq 0 \\
    & \sum_{i=1}^N\pi_i = 1
    \end{split}
    $$

---

那么整个HMM模型由$\mu = (S,K,\pi,A,B)$ 五元组构成。  

# 判别式模型  

## 逻辑回归  
先进行如下定义和假设

1. 标签$Y$是布尔类型（取值只有0，1），并且服从二项分布$\pi = P(Y=1)$
2. 对于所有$X_i$，$P(X_i|Y=y_k)$满足高斯分布$N(\mu_{ij},\sigma_i)$
3. 对于所有$X_i$，给定$Y$，不同的$X$是独立的

公式推导：
1. 通过贝叶斯定理，公式可以写成如下：
    $$
    P(Y=1|X) = \frac{P(Y=1)P(X|Y=1)}{P(Y=1)P(X|Y=1) + P(Y=0)P(X|Y=0)}
    $$
    
2. 分子分母同除分子
    $$
    P(Y=1|X) = \frac{1}{1 + \frac{P(Y=1)P(X|Y=1)}{P(Y=0)P(X|Y=0)}}
    $$
    
3. 变形如下
    $$
    P(Y=1|X) = \frac{1}{1 + \exp(\ln\frac{P(Y=0)P(X|Y=0)}{P(Y=1)P(X|Y=1)})}
    $$
    
4. 通过独立假设，上式可以改写成   
    $$
    \begin{split}
    P(Y=1|X) & = \frac{1}{1 + \exp(\ln\frac{P(Y=0)}{P(Y=1)} + \sum_i\ln\frac{P(X_i|Y=0)}{P(X_i|Y=1)})} \\
    & = \frac{1}{1 + \exp(\ln\frac{1-\pi}{\pi} + \sum_i\ln\frac{P(X_i|Y=0)}{P(X_i|Y=1)})}
    \end{split}
    $$
    其中我们将$P(Y=1)$表示成$\pi$，那么$P(Y=0)$就可以表示成$1-\pi$ 

5. 先关注分母中的求和项，将$P(X_i|Y=y_k)$的高斯分布表示展开如下：
    $$
    \begin{split}
    \sum_i\ln\frac{P(X_i|Y=0)}{P(X_i|Y=1)} & = \sum_i\ln\frac{\frac{1}{\sqrt{2\pi\sigma_i^2}}\exp{(\frac{-(X_i-\mu_{i0})^2}{2\sigma_i^2})}}{\frac{1}{\sqrt{2\pi\sigma_i^2}}\exp{(\frac{-(X_i-\mu_{i1})^2}{2\sigma_i^2})}} \\
    & = \sum_i\ln\frac{\exp{(\frac{-(X_i-\mu_{i0})^2}{2\sigma_i^2})}}{\exp{(\frac{-(X_i-\mu_{i1})^2}{2\sigma_i^2})}} \\
    & = \sum_i\ln\exp(\frac{(X_i-\mu_{i1})^2-(X_i-\mu_{i0})^2}{2\sigma_i^2}) \\ 
    & = \sum_i(\frac{(X_i-\mu_{i1})^2-(X_i-\mu_{i0})^2}{2\sigma_i^2}) \\
    & = \sum_i(\frac{(X_i^2-2X_i\mu_{i1} + \mu_{i1}^2)-(X_i^2-2X_i\mu_{i0} + \mu_{i0}^2)}{2\sigma_i^2}) \\ 
    & = \sum_i(\frac{2X_i(\mu_{i0}-\mu_{i1}) + \mu_{i1}^2 - \mu_{i0}^2}{2\sigma_i^2}) \\ 
    & = \sum_i(\frac{\mu_{i0}-\mu_{i1}}{\sigma_i^2}X_i + \frac{\mu_{i1}^2 - \mu_{i0}^2}{2\sigma_i^2}) 
    \end{split} 
    $$

6. 代回原式
    $$
    \begin{split}
    P(Y=1|X) & = \frac{1}{1 + \exp(\ln\frac{1-\pi}{\pi} + \sum_i(\frac{\mu_{i0}-\mu_{i1}}{\sigma_i^2}X_i + \frac{\mu_{i1}^2 - \mu_{i0}^2}{2\sigma_i^2}))} \\
    & = \frac{1}{1 + \exp(\ln\frac{1-\pi}{\pi} + \sum_i(\frac{\mu_{i0}-\mu_{i1}}{\sigma_i^2}X_i) + \sum_i\frac{\mu_{i1}^2 - \mu_{i0}^2}{2\sigma_i^2})}
    \end{split}
    $$
    
7.  分别设
    $$
    b = \ln\frac{1-\pi}{\pi} + \sum_i\frac{\mu_{i1}^2 - \mu_{i0}^2}{2\sigma_i^2}
    $$
    $$
    w_i = \sum_i\frac{\mu_{i0}-\mu_{i1}}{\sigma_i^2}
    $$
    那么原式可写为
    $$
    P(Y=1|X) = \frac{1}{1 + \exp(b + \sum_{i=1}^nw_iX_i)}
    $$
    
8. 如果写成向量模式，则为：
    $$
    y = \frac{1}{1 + \exp(b+w^Tx)}
    $$

## 线性对数模型  
将逻辑回归推广至多分类，即SoftMax
$$
P(Y=k|X) = \frac{\exp(w_k^Tx)}{\sum_i\exp(w_i^Tx)}
$$

现在设$f(x,y)$为一个特征函数，那么上式可以改写成
$$
P(Y=k|X) = \frac{\exp(wf(x,y))}{\sum_{y'}\exp(wf(x,{y'}))}
$$
其中特征函数可以设置为
$$
f(x,y) = 
\begin{cases}
x, & y=k \\
0, & other
\end{cases}
$$

该模型称之为线性对数模型。

当考虑多个特征函数时，线性对数模型的更一般形式为
$$
P(Y|X) = \frac{\exp\sum_{j=1}^Jw_jf_j(x,y)}{\sum_{y'}\exp\sum_{j=1}^Jw_jf_j(x,{y'})}
$$  

## 条件随机场  
### 对数线性模型角度  
在条件随机场模型中，我们对特征函数做如下约束：  

1. $f_j(x,y)$ 需要把整个$x$输入，即考虑全部的输入
2. 新增一个$x$序列中的词的位置编码信息$i$，即需要告诉特征函数当前输入的$x$序列的位置
3. 当前$x_i$对应的标签$y_i$
4. 上一个标签$y_{i-1}$
那么特征函数可以写成
$$
f_j(x,y) = \sum_{i=1}^Tf_j(x,y_{i-1},y_i,i)
$$
这样模型就可以写成
$$
P(Y=y_t|X) = \frac{\exp\sum_{j=1}^Jw_j\sum_{i=1}^tf_j(x,y_{i-1},y_i,i)}{\sum_{y'}\exp\sum_{j=1}^Jw_j\sum_{i=1}^tf_j(x,y'_{i-1},y'_i,i)}
$$
因为线性链条件随机场处理的是序列数据，那么我们的序列模型可以写成
$$
P(y_i……y_{n}|x_i……x_{n}) =\prod_{i=1}^n P(y_i|x_1……x_i)
$$
综合一下，整个线性链条件随机场的模型公式如下
$$
P(Y|X) = \frac{1}{Z(X)}\prod_{i=1}^n\Psi_i(X_t,y_{i-1},y_i,i)
$$

$$
\Psi_i(X_t,y_{i-1},y_i,i) = \exp\{\sum_{j=1}^Jw_j\sum_{i=1}^tf_j(X_t,y_{i-1},y_i,i)\}
$$
$$
Z(X) = \sum_y\prod_{i=1}^n\Psi_i(X_t,y_{i-1},y_i,i)
$$  

### 隐马可夫模型角度  
HMM模型可以改写成如下  
$$
P(y,x) = \frac{1}{Z}\prod_{t=1}^T\exp\{\sum_{i,j \in S}\theta_{ij}1_{\{y_t=i\}}1_{\{y_{t-1}=j\}} + \sum_{i\in S}\sum_{k\in K}\mu_{ki}1_{\{y_t=i\}}1_{\{x_t=k\}}\}
$$

其中
$$
\begin{split}
& Z = 1 \\
& \theta_{ij} = \ln P(y'=i|y=j) \\
& \mu_{ki} = \ln P(x=k|y=i)
\end{split}
$$
更进一步的引入特征函数的概念，每个特征函数形如$f_k(y_t,y_{t-1},x)$。那么上式中的$f_{ij}(y',y,x)=1_{\{y=i\}}1_{\{y'=j\}}$。同样$f_{ki}(y,y',x)=1_{\{y=i\}}1_{\{x=k\}}$。那么，简化后的表达式
$$
P(y,x) = \frac{1}{Z}\prod_{t=1}^T\exp\{\sum_{k=1}^K\theta_kf_k(y_t,y_{t-1},x\}
$$
在HMM中，我们是对联合概率$P(Y,X)$建模，那么在条件随机场中，我们是对条件概率$P(Y|X)$建模，那么表达式即为
$$
\begin{split}
P(y|x) & = \frac{P(y,y)}{P(x)} = \frac{P(y,x)}{\sum_y'P(y',x)} \\
& = \frac{\prod_{t=1}^T\exp\{\sum_{k=1}^K\theta_kf_k(y_t,y_{t-1},x\}}{\sum_{}y'\prod_{t=1}^T\exp\{\sum_{k=1}^K\theta_kf_k(y'_t,y'_{t-1},x\}}
\end{split}
$$

这就是条件随机场的一种特殊形式。

# 参考文献

[1] http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf

[2] http://www.cs.columbia.edu/~mcollins/loglinear.pdf

[3] http://www.cs.columbia.edu/~mcollins/crf.pdf

[4] https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf

[5] https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf