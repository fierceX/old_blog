---
title: "使用Gluon实现Hulti Head Attention做情感分析"
date: 2018-06-03
draft: true
markup: mmark
---
## 1. Attention和Hulti Head Attention
在NLP任务中，Attention（注意力机制）是很常见也是很重要的一个策略，而在2017年Google的[《Attention is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)中，提出了Hulti Head Attention（多头注意力机制），下面简单介绍一下Attention和Hulti Head Attention
### 1.1 Attention
Attention本质可以描述为一个查询（Query）到一系列键（Key）值（Value）对的映射。而计算Attention一般分为三步
1. 将Query和每个Key进行相似度计算得到权重，在这里有很多计算方式，例如点积，拼接，感知机等。  

    $$
    f(Q,K)=\begin{cases} Q^TK &\text {dot} \\ Q^TWK & \text{general} \\ W[Q:K] &\text{concat} \\v_a^Ttanh(W_aQ+U_aK) & \text{perceptron}\end{cases}
    $$  

2. 使用SoftMax对权重进行归一化  

    $$
    a=softmax(f(Q,K))
    $$  

3. 将权重和对应的键值进行加权求和得到最后的结果  

    $$
    Attention(Q,K,V)=\sum aV
    $$

在NLP中，一般Key和Value是同一个，即Key=Value  
而在Google的论文中，采用的是点积的Attention函数,并引入了一个调节因子$$\sqrt{d_k}$$
<center>![Scaled Dot Product Attention](/img/Scaled_Dot_Product_Attention.png)</center>  

$$
Attention (Q,K,V) = SoftMax(\frac {Q^TK} {\sqrt{d_k}})V
$$

### 1.2 Hulti Head Attention
Hulti Head Attention就是把Q，K，V通过参数矩阵映射一下然后再做Attention，然后把这个步骤重复h次，然后把结果拼接起来 

<center>![Multi Head Attention](/img/Multi_Head_Attention.png)</center>

$$
head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)  
$$

$$
MultiHead(Q,K,V) = Concat(head_1,...,head_h)
$$

## 1.3 Position Embedding
在这篇论文中，还提到了一个Position Embedding的技巧，这是因为Attention无法捕捉序列的顺序关系，所以引入了Position Embedding（位置向量）的概念，在论文中，Position Embedding是由公式直接计算而得的  

$$
PE_{(pos,2i)} =sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

## 2. Hulti Head Attention的Gluon实现
我全部使用了`HybridBlock`，以方便导出成Mxnet格式
### 2.1 Attention
``` python
class Attention(nn.HybridBlock):
    def __init__(self, Q_shape, K_shape, V_shape, **kwargs):
        super(Attention, self).__init__(**kwargs)
        #三个参数矩阵
        with self.name_scope():
            self.Wq = self.params.get('Wq', shape=Q_shape)
            self.Wk = self.params.get('Wk', shape=Q_shape)
            self.Wv = self.params.get('Wv', shape=Q_shape)
            self.qq = Q_shape[1]
        self.Q_shape = Q_shape
        self.K_shape = K_shape
        self.V_shape = V_shape
    
    def SacledDotProductAttention(self, F, q, k, v):
        S = F.batch_dot(q, F.transpose(k, axes=(0, 2, 1))) / (self.qq**0.5)
        return F.batch_dot(F.softmax(S), v)

    def hybrid_forward(self, F, q, k, v, Wq, Wk, Wv):
        return self.SacledDotProductAttention(F, F.dot(q, Wq), F.dot(k, Wk),
                                              F.dot(v, Wv))

    #这个函数是用来打印该层信息的
    def __repr__(self):
        s = '{name}(Q_shape={Q_shape}, K_shape={K_shape}, V_shape={V_shape}, out_shape={out_shape})'
        return s.format(
            name=self.__class__.__name__,
            Q_shape=str((self.Q_shape[1], self.Q_shape[0])),
            K_shape=str((self.K_shape[1], self.K_shape[0])),
            V_shape=str((self.V_shape[1], self.V_shape[0])),
            out_shape=str((self.K_shape[1], self.V_shape[0])))
```
### 2.2 Hulti Head Attention

``` python
class Multi_Head_Attention(nn.HybridBlock):
    def __init__(self, Q_shape, K_shape, V_shape, h, **kwargs):
        super(Multi_Head_Attention, self).__init__(**kwargs)
        with self.name_scope():
            for _ in range(h):
                self.register_child(
                    Attention(
                        Q_shape=Q_shape,
                        K_shape=K_shape,
                        V_shape=V_shape))
            self.Wo = self.params.get('Wo', shape=(h * V_shape[1], V_shape[0]))
        self.h = h

    def hybrid_forward(self, F, q, k, v, Wo):
        H = []
        for block in self._children.values():
            H.append(block(q, k, v))
        return F.dot(F.concat(*H, dim=2), Wo)

    def __repr__(self):
        s = '{name}({Attention}, h_num={h})'
        return s.format(
            name=self.__class__.__name__,
            Attention=list(self._children.values())[0],
            h=self.h)
```
### 2.2 Position Embedding
``` python
class Position_Embedding(nn.HybridBlock):
    def __init__(self, shape, **kwargs):
        super(Position_Embedding, self).__init__(**kwargs)
        self.shape = shape

    def hybrid_forward(self, F, x):
        seq_len = self.shape[0]
        position_size = self.shape[1]
        position_j = 1. / F.broadcast_power(
            10000 * F.ones(shape=(1, 1)),
            (2 * F.arange(position_size / 2) / position_size))
        position_i = F.arange(seq_len, dtype='float32')
        position_i = F.expand_dims(position_i, 1)
        position_ij = F.dot(position_i, position_j)
        position_ij = F.concat(
            *[F.cos(position_ij), F.sin(position_ij)], dim=1)
        position_ij = F.broadcast_add(
            F.expand_dims(position_ij, 0), F.zeros_like(x))

        return position_ij + x
```
## 3. 使用这个Hulti Head Attention来进行情感分析
数据源是文本分类里的MNIST，IMDB的影评，通过影评来分辨是正面还是负面的情绪。  
[http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)  
### 3.1 导入各种包
``` python
from collections import Counter
import os
import random
import time
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from tqdm import tqdm
```
### 3.2 数据读取和处理
``` python
data_dir = './aclImdb/'
context = mx.gpu(1)
batch_size = 32
max_len = 100
embeding_size = 128

#读取源数据
def readIMDB(dir_url, seg='train'):
    pos_or_neg = ['pos', 'neg']
    dataset = []
    for lb in pos_or_neg:
        files = os.listdir(dir_url + '/' + seg + '/' + lb + '/')
        for file in tqdm(files):
            with open(
                    dir_url + '/' + seg + '/' + lb + '/' + file,
                    'r',
                    encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if lb == 'pos':
                    dataset.append([review, 1])
                elif lb == 'neg':
                    dataset.append([review, 0])
    return dataset

train_dataset = readIMDB(data_dir, 'train')
test_dataset = readIMDB(data_dir, 'test')

# shuffle 数据集。
random.shuffle(train_dataset)
random.shuffle(test_dataset)

def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
train_labels = []
for review, score in train_dataset:
    train_tokenized.append(tokenizer(review))
    train_labels.append(score)
test_tokenized = []
test_labels = []
for review, score in test_dataset:
    test_tokenized.append(tokenizer(review))
    test_labels.append(score)

token_counter = Counter()


def count_token(train_tokenized):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1


count_token(train_tokenized)
vocab = text.vocab.Vocabulary(
    token_counter, unknown_token='<unk>', reserved_tokens=None)

# 根据词典，将数据转换成特征向量。
def encode_samples(x_raw_samples, vocab):
    x_encoded_samples = []
    for sample in x_raw_samples:
        x_encoded_sample = []
        for token in sample:
            if token in vocab.token_to_idx:
                x_encoded_sample.append(vocab.token_to_idx[token])
            else:
                x_encoded_sample.append(0)
        x_encoded_samples.append(x_encoded_sample)
    return x_encoded_samples


# 将特征向量补成定长。
def pad_samples(x_encoded_samples, maxlen=100, val=0):
    x_samples = []
    for sample in x_encoded_samples:
        if len(sample) > maxlen:
            new_sample = sample[:maxlen]
        else:
            num_padding = maxlen - len(sample)
            new_sample = sample
            for i in range(num_padding):
                new_sample.append(val)
        x_samples.append(new_sample)
    return x_samples

x_encoded_train = encode_samples(train_tokenized, vocab)
x_encoded_test = encode_samples(test_tokenized, vocab)

x_train = nd.array(pad_samples(x_encoded_train, max_len, 0), ctx=context)
x_test = nd.array(pad_samples(x_encoded_test, max_len, 0), ctx=context)

y_train = nd.array([score for text, score in train_dataset], ctx=context)
y_test = nd.array([score for text, score in test_dataset], ctx=context)

train_data = gluon.data.ArrayDataset(x_train, y_train)
train_dataloader = gluon.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)

test_data = gluon.data.ArrayDataset(x_test, y_test)
test_dataloader = gluon.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True)

#验证函数
def eval(dataloader):
    total_L = 0
    ntotal = 0
    accuracy = mx.metric.Accuracy()
    for data, label in dataloader:
        output = net(data)
        label = label.as_in_context(context)
        L = loss(output, label)
        total_L += nd.sum(L).asscalar()
        ntotal += L.size
        predicts = nd.argmax(output, axis=1)
        accuracy.update(preds=predicts, labels=label)
    return total_L / ntotal, accuracy.get()[1]


```
### 3.3 构建情感分析模型
``` python
class SANet(gluon.HybridBlock):
    def __init__(self, shape, Vocad_len, h,Is_PE=True, **kwargs):
        super(SANet, self).__init__(**kwargs)
        self.embed = nn.Embedding(input_dim=Vocad_len, output_dim=shape[0])
        self.PE = Position_Embedding(shape=(shape[1], shape[0]))
        self.MHA = Multi_Head_Attention(
            Q_shape=shape, K_shape=shape, V_shape=shape, h=h)
        self.liner = gluon.nn.Dense(2)
        self.pool = gluon.nn.GlobalAvgPool1D()
        self.droup = gluon.nn.Dropout(.5)
        self.shape = (shape[1],shape[0])
        self.Is_PE = Is_PE
    
    def hybrid_forward(self, F, x):
        kqv = self.PE(self.embed(x))
        kqv = self.embed(x)
        if self.Is_PE:
            kqv = self.PE(kqv)
        return self.liner(self.droup(self.pool(self.MHA(kqv, kqv, kqv))))
```
### 3.2 初始化并训练
``` python
net = SANet(
    shape=(embeding_size, max_len), Vocad_len=len(vocab), h=8, Is_PE=False)
net.initialize(mx.init.Uniform(.1), ctx=context)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'Adam')
loss = gluon.loss.SoftmaxCrossEntropyLoss()
loss.hybridize()

num_epochs = 2
start_train_time = time.time()
for epoch in range(num_epochs):
    start_epoch_time = time.time()
    total_L = 0
    ntotal = 0

    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            output = net(data)
            L = loss(output, label.as_in_context(context))
        L.backward()
        trainer.step(batch_size)
        total_L += nd.sum(L).asscalar()
        ntotal += L.size
        if i % 30 == 0 and i != 0:
            print('Epoch %d. batch %d. Loss %6f' % (epoch, i,
                                                    total_L / ntotal))
            total_L = 0
            ntotal = 0

    print('performing testing:')
    train_loss, train_acc = eval(train_dataloader)
    test_loss, test_acc = eval(test_dataloader)

    print('[epoch %d] train loss %.6f, train accuracy %.2f' %
          (epoch, train_loss, train_acc))
    print('[epoch %d] test loss %.6f, test accuracy %.2f' % (epoch, test_loss,
                                                             test_acc))
    print('[epoch %d] throughput %.2f samples/s' %
          (epoch,
           (batch_size * len(x_train)) / (time.time() - start_epoch_time)))
    print('[epoch %d] total time %.2f s' % (epoch,
                                            (time.time() - start_epoch_time)))

print('total training throughput %.2f samples/s' %
      ((batch_size * len(x_train) * num_epochs) /
       (time.time() - start_train_time)))
print('total training time %.2f s' % ((time.time() - start_train_time)))
```
---
完整代码：[https://github.com/fierceX/Attention-is-All-You-Need](https://github.com/fierceX/Attention-is-All-You-Need)