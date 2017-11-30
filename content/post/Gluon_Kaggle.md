---
title: "Gluon炼丹（Kaggle 120种狗分类，迁移学习加双模型融合）"
date: 2017-11-29T22:28:36+08:00
draft: true
---

这是在[kaggle](https://www.kaggle.com/c/dog-breed-identification)上的一个练习比赛，使用的是ImageNet数据集的子集。  
注意，`mxnet`版本要高于`0.12.1b2017112`。
下载数据集。  
- [train.zip](https://www.kaggle.com/c/dog-breed-identification/download/train.zip)  
- [test.zip](https://www.kaggle.com/c/dog-breed-identification/download/test.zip)  
- [labels](https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip)  
然后解压在`data`文件夹下  
## 1. 数据
### 1.1 整理数据
将解压后的数据整理成Gluon能够读取的形式，这里我直接使用了[zh.gluon.ai](http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-dog.html)教程上的代码  
导入各种库
``` python
import math
import os
import shutil
from collections import Counter
```
设置一些变量
``` python
data_dir = './data'
label_file = 'labels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
batch_size = 128
valid_ratio = 0.1
```
定义整理数据函数
``` python
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # 训练集中数量最少一类的狗的数量。
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的数量。
    num_valid_per_label = math.floor(min_num_train_per_label * valid_ratio)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
```
调用这个函数整理数据集
``` python
reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio)
```
### 1.2 载入数据
数据整理好之后，需要载入到gluon中，首先需要定义转换函数，因为需要模型融合，所以需要两个输入，分别经过两个不同的模型  
导入各种包
``` python
from mxnet import gluon
from mxnet.gluon.data import vision
from mxnet import image
import numpy as np
from mxnet import nd
```
在训练数据里，开启了一些数据增强，并且做了一些预处理。需要说明的是，这里的图像均值和方差是ImageNet数据集的。这是因为预训练模型的数据集是ImageNet，那么我们就必须按照模型训练的时候处理的方式来处理我们的数据，这样才能保证最好的效果。图像尺寸同样，这里使用两个不同的尺寸就是为了两个不同的网络准备的。  
``` python
def transform_train(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist1:
        im1 = aug(im1)
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2,0,1))
    im2 = nd.transpose(im2, (2,0,1))
    return (im1,im2, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224),
                        mean=np.array([0.485, 0.456, 0.406]), 
                        std=np.array([0.229, 0.224, 0.225]))
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299),
                        mean=np.array([0.485, 0.456, 0.406]), 
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist1:
        im1 = aug(im1)
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2,0,1))
    im2 = nd.transpose(im2, (2,0,1))
    return (im1,im2, nd.array([label]).asscalar().astype('float32'))
```
转换函数定义好之后，就可以载入到gluon里了
``` python
batch_size = 32

train_ds = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                      transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                      transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + train_valid_dir,
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                      transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True,
                          last_batch='keep')
```
## 2. 设计网络
这里为了得到一个更好的名次，并且减少我们的工作量，使用了迁移学习加模型融合。迁移学习就是使用预训练模型里的特征层，然后再自己填补后面的输出层。因为与训练模型都是老司机了，见多识广，所以这些数据也不在话下。  
模型融合这里使用两个模型，分别是`resnet152_v1`和`inception_v3`  
导入各种包
``` python
from mxnet import gluon
from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
from mxnet import image
```
### 2.1 双模型合并
为了让两个网络合并，就需要自定义一个合并两个网络的层。  
在这里，我为每个网络添加了一个`GlobalAvgPool2D`层，这是为了让两个网络输出的尺寸可以合并。
``` python
class  ConcatNet(nn.HybridBlock):
    def __init__(self,net1,net2,**kwargs):
        super(ConcatNet,self).__init__(**kwargs)
        self.net1 = nn.HybridSequential()
        self.net1.add(net1)
        self.net1.add(nn.GlobalAvgPool2D())
        self.net2 = nn.HybridSequential()
        self.net2.add(net2)
        self.net2.add(nn.GlobalAvgPool2D())
    def hybrid_forward(self,F,x1,x2):
        return F.concat(*[self.net1(x1),self.net2(x2)])
``` 
这样就可以构造出一个特征提取层
``` python
def get_features2(ctx):
    resnet = vision.inception_v3(pretrained=True,ctx=ctx)
    return resnet.features

def get_features1(ctx):
    resnet = vision.resnet152_v1(pretrained=True,ctx=ctx)
    return resnet.features

def get_features(ctx):
    features1 = get_features1(ctx)
    features2 = get_features2(ctx)
    net = ConcatNet(features1,features2)
    return net
```
### 2.3 输出层
输出层比较简单，两层全链接，中间加了一层`Dropout`
``` python
def get_output(ctx,ParamsName=None):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dropout(.7))
        net.add(nn.Dense(120))
    if ParamsName is not None:
        net.collect_params().load(ParamsName,ctx)
    else:
        net.initialize(init = init.Xavier(),ctx=ctx)
return net
```
### 2.4 连接成一个网络
有了可以合并两个网络的层，还有一个输出层，那么我们需要将这两个网络连接起来。
``` python
class  OneNet(nn.HybridBlock):
    def __init__(self,features,output,**kwargs):
        super(OneNet,self).__init__(**kwargs)
        self.features = features
        self.output = output
    def hybrid_forward(self,F,x1,x2):
        return self.output(self.features(x1,x2))
```
这样就可以构造出一个完整的网络
``` python
def get_net(ParamsName,ctx):
    output = get_output(ctx,ParamsName)
    features = get_features(ctx)
    net = OneNet(features,output)
    return net
```
## 3. 训练
有着前面的准备，就可以开始干活了。首先第一步是提取特征，因为是迁移学习，会锁定特征层。那干脆让所有训练数据都过一遍特征网络，这样既节约时间，有节省显存。何乐而不为。  
导入各种包
``` python
from tqdm import tqdm
import os
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import pandas as pd
import mxnet as mx
import pickle
```
### 3.1 提取特征
提取特征我们使用上面定义好的特征提取网络
``` python
net = get_features(mx.gpu())
net.hybridize()

def SaveNd(data,net,name):
    x =[]
    y =[]
    print('提取特征 %s' % name)
    for fear1,fear2,label in tqdm(data):
        fear1 = fear1.as_in_context(mx.gpu())
        fear2 = fear2.as_in_context(mx.gpu())
        out = net(fear1,fear2).as_in_context(mx.cpu())
        x.append(out)
        y.append(label)
    x = nd.concat(*x,dim=0)
    y = nd.concat(*y,dim=0)
    print('保存特征 %s' % name)
    nd.save(name,[x,y])

SaveNd(train_data,net,'train_r152i3.nd')
SaveNd(valid_data,net,'valid_r152i3.nd')
SaveNd(train_valid_data,net,'input_r152i3.nd')
```
为了最后输出提交文件做准备，保存一下需要的东西
``` python
ids = ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
synsets = train_valid_ds.synsets
f = open('ids_synsets','wb')
pickle.dump([ids,synsets],f)
f.close()
```
### 3.2 载入预训练后的数据
### 3.3 训练模型
训练之前先把各种参数设置一下
``` python
num_epochs = 100
batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-4
pngname='train.png'
modelparams='r152i3.params'
```
然后载入特征提取后的数据
``` python
train_nd = nd.load('train_r152i3.nd')
valid_nd = nd.load('valid_r152i3.nd')
input_nd = nd.load('input_r152i3.nd')
f = open('ids_synsets','rb')
ids_synsets = pickle.load(f)
f.close()

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_nd[0],train_nd[1]), batch_size=batch_size,shuffle=True)
valid_data = gluon.data.DataLoader(gluon.data.ArrayDataset(valid_nd[0],valid_nd[1]), batch_size=batch_size,shuffle=True)
input_data = gluon.data.DataLoader(gluon.data.ArrayDataset(input_nd[0],input_nd[1]), batch_size=batch_size,shuffle=True)
```
设置训练函数和loss函数
``` python
def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx):
    trainer = gluon.Trainer(
        net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    train_loss = []
    if valid_data is not None:
        test_loss = []
    
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        _loss = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            _loss += nd.mean(loss).asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)
        train_loss.append(__loss)
        
        if valid_data is not None:  
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Valid loss %f, "
                         % (epoch,__loss , valid_loss))
            test_loss.append(valid_loss)
        else:
            epoch_str = ("Epoch %d. Train loss: %f, "
                         % (epoch, __loss))
            
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        

    plt.plot(train_loss, 'r')
    if valid_data is not None: 
        plt.plot(test_loss, 'g')
    plt.legend(['Train_Loss', 'Test_Loss'], loc=2)


    plt.savefig(pngname, dpi=1000)
    net.collect_params().save(modelparams)
```
接下来就可以训练了
``` python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
ctx = mx.gpu()
net = get_output(ctx)
net.hybridize()

train(net, train_data,valid_data, num_epochs, learning_rate, weight_decay, ctx)
```
## 输出测试结果
训练之后，就可以把测试集的数据跑出来了  
首先定义一些变量  
``` python
netparams = 'r152i3.params'
csvname = 'kaggle.csv'
ids_synsets_name = 'ids_synsets'
f = open(ids_synsets_name,'rb')
ids_synsets = pickle.load(f)
f.close()
```
从原始图像载入数据，并定义测试输出函数
``` python
test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                     transform=transform_test)
def SaveTest(test_data,net,ctx,name,ids,synsets):
    outputs = []
    for data1,data2, label in tqdm(test_data):
        data1 =data1.as_in_context(ctx)
        data2 =data2.as_in_context(ctx)
        output = nd.softmax(net(data1,data2))
        outputs.extend(output.asnumpy())
    with open(name, 'w') as f:
        f.write('id,' + ','.join(synsets) + '\n')
        for i, output in zip(ids, outputs):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')
```
开跑
``` python
net = get_net(netparams,mx.gpu())
net.hybridize()
SaveTest(test_data,net,mx.gpu(),csvname,ids_synsets[0],ids_synsets[1])
```
最后就可以把输出的`csv`文件提交到kaggle上了。  
使用kaggle提供的数据，最后拿到了`0.27760`的分数。如果更进一步，那就是用[Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)数据集。
## 感悟
首先这次的kaggle比赛算是我第一次结束正式的图像分类比赛，在gluon论坛里也学到了好多东西。  
使用迁移学习的话，那前期就先把数据过一遍特征网络，省时省力。如需要训练前面特征网络的时候，再连起来训练就可以了。  
大多数图像分类都可以使用预训练模型进行迁移训练，因为经过ImageNet的模型都是老司机了，见多识广。  
使用预训练模型进行迁移学习，那么数据处理要和原模型的一致，比如图像尺寸，归一化等。  
最后感谢[沐神的直播课程](https://discuss.gluon.ai/c/5-category)，和论坛里的大神[杨培文](https://github.com/ypwhs/DogBreed_gluon)提供的思路和借鉴代码。  
完整代码: [https://github.com/fierceX/Dog-Breed-Identification-Gluon](https://github.com/fierceX/Dog-Breed-Identification-Gluon)