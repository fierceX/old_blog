---
title: "使用MxNet新接口Gluon提供的预训练模型进行微调"
date: 2017-09-27
draft: true
---
## 1. 导入各种包


```python
from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import cv2
from mxnet import image
from mxnet import autograd
```

## 2. 导入数据
我使用cifar10这个数据集，使用gluon自带的模块下载到本地并且为了配合后面的网络，我将大小调整到224*224


```python
def transform(data, label):
    data = image.imresize(data, 224, 224)
    return data.astype('float32'), label.astype('float32')
cifar10_train = gluon.data.vision.CIFAR10(root='./',train=True, transform=transform)
cifar10_test = gluon.data.vision.CIFAR10(root='./',train=False, transform=transform)
```


```python
batch_size = 64
train_data = gluon.data.DataLoader(cifar10_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(cifar10_test, batch_size, shuffle=False)
```

## 3. 加载预训练模型
gluon提供的很多预训练模型，我选择一个简单的模型AlexNet  
首先下载AlexNet模型和模型参数  
使用下面的代码会获取AlexNet的模型并且加载预训练好的模型参数，但是鉴于网络的原因，我提前下好了


```python
alexnet = mx.gluon.model_zoo.vision.alexnet(pretrained=True)#如果pretrained值为True，则会下载预训练参数，否则是空模型
```

获取模型并从本地加载参数


```python
alexnet = mx.gluon.model_zoo.vision.alexnet()
alexnet.load_params('alexnet-44335d1f.params',ctx=mx.gpu())
```

看下AlexNet网络结构，发现分为两部分，features,classifier,而features正好是需要的


```python
print(alexnet)
```

    AlexNet(
      (features): HybridSequential(
        (0): Conv2D(64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (2): Conv2D(192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (3): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (4): Conv2D(384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): Conv2D(256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Conv2D(256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (8): Flatten
      )
      (classifier): HybridSequential(
        (0): Dense(4096, Activation(relu))
        (1): Dropout(p = 0.5)
        (2): Dense(4096, Activation(relu))
        (3): Dropout(p = 0.5)
        (4): Dense(1000, linear)
      )
    )
    

## 4. 组合新的网络
截取想要的features，并且固定参数。这样防止训练的时候把预训练好的参数给搞坏了


```python
featuresnet = alexnet.features
for _, w in featuresnet.collect_params().items():
    w.grad_req = 'null'
```

自己定义后面的网络，因为数据集是10类，就把最后的输出从1000改成了10。


```python
def Classifier():
    net = nn.HybridSequential()
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(10))
    return net
```

接着需要把两部分组合起来，并且对第二部分机进行初始化


```python
net = nn.HybridSequential()
with net.name_scope():
    net.add(featuresnet)
    net.add(Classifier())
    net[1].collect_params().initialize(init=mx.init.Xavier(),ctx=mx.gpu())
net.hybridize()
```

## 5. 训练
最后就是训练了，看看效果如何


```python
#定义准确率函数
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()
def evaluate_accuracy(data_iterator, net, ctx=mx.gpu()):
    acc = 0.
    for data, label in data_iterator:
        data = data.transpose([0,3,1,2])
        data = data/255
        output = net(data.as_in_context(ctx))
        acc += accuracy(output, label.as_in_context(ctx))
    return acc / len(data_iterator)
```


```python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.01})
```


```python
for epoch in range(1):
    train_loss = 0.
    train_acc = 0.
    test_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(mx.gpu())
        data = data.transpose([0,3,1,2])
        data = data/255
        with autograd.record():
            output = net(data.as_in_context(mx.gpu()))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), 
        train_acc/len(train_data),test_acc))
```

    Epoch 0. Loss: 1.249197, Train acc 0.558764, Test acc 0.696756
    
