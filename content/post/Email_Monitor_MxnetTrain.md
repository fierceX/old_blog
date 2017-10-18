---
title: "使用邮件监控Mxnet训练"
date: 2017-10-12
draft: true
---

## 1. 前言
受到小伙伴的启发，就自己动手写了一个使用邮件监控Mxnet训练的例子。整体不算复杂。  
## 2. 设置一些全局参数 
邮箱服务的pop,smtp地址，邮箱账号，接受邮箱号和密码以及当前训练状态  
还有训练的超参数和保存路径和文件名参数等  
```python
pophost = 'pop.126.com'
smtphost = 'smtp.126.com'
useremail = 'trainmonitor@126.com'
toemail = 'fiercewind@outlook.com'
password = '123456'

running = False

params = {'ep': 10, 'lr': 0.002, 'bs': 128, 'wd': 0.0}
nameparams = {'dir':'./','params':'NN.params','png':'NN.png'}
```
## 3. 打包训练代码
需要进行监控训练，所以需要将训练的代码打包进一个函数内，通过传参的方式进行训练。还是使用FashionMNIST数据集  
这样训练的时候就调用函数传参就行了
### 3.1 训练主函数
训练需要的一些参数都采用传参的形式  
这里我新加了一个名叫`nameparams`的参数，用于设置曲线图，保存的参数文件的路径和文件名  
```python
def NN_Train(net, train_data, test_data,params,nameparams):
    msg = ''

    epochs = int(params['ep'])
    batch_size = int(params['bs'])
    learning_rate = params['lr']
    weight_decay = params['wd']

    train_loss = []
    train_acc = []
    dataset_train = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loss = []
    test_acc = []
    dataset_test = gluon.data.DataLoader(test_data, batch_size, shuffle=True)

    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        _loss = 0.
        _acc = 0.
        t_acc = 0.
        for data, label in dataset_train:
            data = nd.transpose(data, (0, 3, 1, 2))
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            _loss += nd.mean(loss).asscalar()
            _acc += accuracy(output, label)
        __acc = _acc / len(dataset_train)
        __loss = _loss / len(dataset_train)
        train_loss.append(__loss)
        train_acc.append(__acc)

        t_acc, t_loss = evaluate_accuracy(dataset_test, net)
        test_loss.append(t_loss)
        test_acc.append(t_acc)

        msg += ("Epoch %d. Train Loss: %f, Test Loss: %f, Train Acc %f, Test Acc %f\n" % (
            epoch, __loss, t_loss, __acc, t_acc))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(train_loss, 'r')
    ax1.plot(test_loss, 'g')
    ax1.legend(['Train_Loss', 'Test_Loss'], loc=2)
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(train_acc, 'b')
    ax2.plot(test_acc, 'y')
    ax2.legend(['Train_Acc', 'Test_Acc'], loc=1)
    ax2.set_ylabel('Acc')

    plt.savefig(os.path.join(nameparams['dir'],nameparams['png']), dpi=600)
    net.save_params(os.path.join(nameparams['dir'],nameparams['params']))
    return msg
```
### 3.2 打包网络模型
同样，需要把网络也打包进函数内  
```python
def GetNN():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(10))
    net.initialize(init=mx.init.Xavier(), ctx=ctx)
    net.hybridize()
    return net
```
### 3.3 打包数据读取
然后把数据读取也搞进函数内
```python
def GetDate():
    fashion_train = gluon.data.vision.FashionMNIST(
        root='./', train=True, transform=transform)
    fashion_test = gluon.data.vision.FashionMNIST(
        root='./', train=True, transform=transform)
    return fashion_train, fashion_test
```

## 4. 搞定邮件的接收发送
使用邮件监控，就要搞定在Python上使用邮件的问题，还好Python内置了邮件库  
这样接收发送邮件也只用调用函数就好了  
### 4.1 接受邮件
我只接受纯文本的内容，因为HTML内容的太过复杂  
```python
def ReEmail():
    try:
        pp = poplib.POP3(pophost)
        pp.user(useremail)
        pp.pass_(password)
        resp, mails, octets = pp.list()
        index = len(mails)
        if index > 0:
            resp, lines, octets = pp.retr(index)
            msg_content = b'\r\n'.join(lines).decode('utf-8')
            pp.dele(index)
            pp.quit()
            msg = Parser().parsestr(msg_content)
            message = Get_info(msg)
            subject = msg.get('Subject')
            date = msg.get('Date')
            return message,subject,date
    except ConnectionResetError as e:
        print('ConnectionResetError')
    return None,None,None
```
### 4.2 发送邮件
发送邮件我是用了一个第三方邮件库`envelopes`，因为简单方便。
```python
def SentEmail(message, subject,imgpath):
    envelope = Envelope(
        from_addr=(Global.useremail, u'Train'),
        to_addr=(Global.toemail, u'FierceX'),
        subject=subject,
        text_body=message
    )
    if imgpath is not None:
        envelope.add_attachment(imgpath)

    envelope.send(Global.smtphost, login=Global.useremail,
                  password=Global.password, tls=True)
```
### 4.3 解析邮件内容
然后需要解析邮件内容，这段基本从网上抄来的，因为邮件格式很复杂，没深究  
```python
def Get_info(msg):
    if (msg.is_multipart()):
        parts = msg.get_payload()
        for n, part in enumerate(parts):
            return Get_info(part)
    if not msg.is_multipart():
        content_type = msg.get_content_type()
        if content_type=='text/plain':
            content = msg.get_payload(decode=True)
            charset = guess_charset(msg)
            if charset:
                content = content.decode(charset)
            return content
```
## 5. 使用责任链模式解析命令
在解析命令里，我使用了责任链模式，并且设置了一个前台类，可以添加新的命令解析类，具体看代码  
### 5.1 责任链基类
我在责任链基类里实现了判断当前命令是否是该对象可执行的命令，这样在编写命令解析类时，就可以忽略判断条件，直接重写解析方法`Work`即可  
```python
class BaseCmd:
    def __init__(self, cmd):
        self.Next = None
        self.cmd = cmd

    def SetNext(self, n):
        self.Next = n

    def DoAnalysis(self, cmd, params):
        if cmd == self.cmd:
            self.Work(params)
        elif self.Next is not None:
            self.Next.DoAnalysis(cmd, params)

    def Work(self, params):
        pass
```
### 5.2 责任链前台类
在前台类里，我添加了一个`Add`方法，用于添加新的命令解析类，在此方法里我自动添加该解析类到责任链的尾部。
```python
class CmdAnaly:
    def __init__(self):
        self.CmdList = []
        self.Add(ExitCmd('exit'))
        self.Add(TrainCmd('train'))
        self.Add(SetNameParamsCmd('setname'))

    def Add(self, cmd):
        self.CmdList.append(cmd)
        if len(self.CmdList) > 1:
            self.CmdList[len(self.CmdList) - 2].SetNext(self.CmdList[len(self.CmdList) - 1])

    def Analy(self, cmd, params):
        self.CmdList[0].DoAnalysis(cmd, params)
```
### 5.3 命令解析类
我只编写了三个命令解析类
#### 训练类
```python
class TrainCmd(BaseCmd):
    def __init__(self, cmd):
        BaseCmd.__init__(self, cmd)

    def Work(self, msg):
        print('train')
        if Global.running == False:
            xx = msg.split('\r\n')
            for k in xx:
                ks = k.split(' ')
                if len(ks) > 1:
                    Global.params[ks[0]] = float(ks[1])
            t = threading.Thread(target=run)
            t.start()
        else:
            message = ('Training is underway\n%s\n%s') % 
            (str(Global.params),str(Global.nameparams))
            EmailTool.SentEmail(message,
                                'Training is underway',
                                None)
```
#### 退出类
```python
class ExitCmd(BaseCmd):
    def __init__(self, cmd):
        BaseCmd.__init__(self, cmd)

    def Work(self, params):
        print('exit')
        os._exit(0)
```
#### 设置图片，参数文件名称和保存路径类
```python
class SetNameParamsCmd(BaseCmd):
    def __init__(self,cmd):
        BaseCmd.__init__(self,cmd)
    
    def Work(self,msg):
        xx = msg.split('\r\n')
        for k in xx:
             ks = k.split(' ')
             if len(ks) > 1:
                 Global.nameparams[ks[0]] = ks[1]
        print(Global.nameparams)
        EmailTool.SentEmail(str(Global.nameparams),'NameParams',None)
```
## 6. 使用多线程多进程监控训练
由于Python的多线程的性能局限性，我使用了子进程进行训练，这样不会受到主进程循环监控的影响  
```python
def nn(params,nameparams):
    train, test = NN_Train.GetDate()
    print(params)
    print(nameparams)
    msg = ('%s\n') % str(params)
    msg += ('%s\n') % str(nameparams)
    msg += NN_Train.NN_Train(
        NN_Train.GetNN(),
        train_data=train,
        test_data=test,
        params = params,
        nameparams = nameparams)
    EmailTool.SentEmail(msg, 'TrainResult',os.path.join(nameparams['dir'],nameparams['png']))

def run():
    p = Process(target=nn,args=(Global.params,Global.nameparams,))
    print('TrainStrart')
    Global.running = True
    p.start()
    p.join()
    Global.running = False
```
## 7. 使用循环监控邮箱
在主进程中，使用循环监控邮箱内容并解析邮件命令，交给命令解析类解析处理。
```python
if __name__ == '__main__':
    Global.running = False
    cmdana = CmdAnalysis.CmdAnaly()
    print('Start')
    a = 1
    while(True):
        time.sleep(10)
        print(a, Global.running)
        try:
            msg, sub, date = EmailTool.ReEmail()
        except TimeoutError as e:
            print('TimeoutError')
        cmdana.Analy(sub, msg)
        a += 1
```
## 8. 效果
### 发送训练邮件
![png](../../img/emmt1.png)
### 训练结束返回结果
![png](../../img/emmt2.png)
## 9. 结语
使用邮件监控并不太复杂，主要在于邮件的解析。邮件格式太复杂，如果全都在主题里，参数多了会显得很乱。  
根据需要添加新的命令解析类，然后在前台类里里使用`Add`方法添加进去就行了。    
总之我认为在aws上训练还是可以一用的，总不能一直连着终端。  
[完整代码](https://github.com/fierceX/Email_Monitor_MxnetTrain)