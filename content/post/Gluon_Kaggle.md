---
title: "Gluon_Kaggle"
date: 2017-11-29T22:28:36+08:00
draft: true
---
# 使用Gluon来做120种狗分类比赛（迁移学习，双模型融合）
这是在[kaggle](https://www.kaggle.com/c/dog-breed-identification)上的一个练习比赛，使用的是ImageNet数据集的子集，首先就是下载数据集。
- [train.zip](https://www.kaggle.com/c/dog-breed-identification/download/train.zip)
- [test.zip](https://www.kaggle.com/c/dog-breed-identification/download/test.zip)
- [labels](https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip)  
然后解压在`data`文件夹下
## 1. 数据
### 1.1 整理数据
首先需要将解压后的数据整理成Gluon能够读取的形式，这里我直接使用了[zh.gluon.ai]()教程上的代码  
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
然后调用这个函数整理数据集
``` python
reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio)
```
### 1.2 载入数据
数据整理好之后，需要载入到gluon中，首先需要定义转换函数，因为需要模型融合，所以需要两个输入，分别经过两个不同的模型  
首先导入各种包
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
首先导入各种包
``` python
from mxnet import gluon
from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
from mxnet import image
```
### 2.1 双模型合并
首先，需要将两个网络合并，那么就需要自定义一个合并两个网络的层。  
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
首先导入各种包
``` python
from mxnet import nd
import mxnet as mx
import pandas as pd
import pickle
from tqdm import tqdm
import os
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
然后为了最后输出提交文件做准备，保存一下需要的东西
``` python
ids = ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
synsets = train_valid_ds.synsets
f = open('ids_synsets','wb')
pickle.dump([ids,synsets],f)
f.close()
```
### 3.2 训练
未完待续