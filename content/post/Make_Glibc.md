---
title: "无ROOT权限更新Glibc库并调用"
date: 2018-08-13
draft: true
---
折腾这个主要原因是在部署项目的时候，客户表示生产环境没有GCC编译工具链，不会给你升级系统库，也不让用docker，说你自己在实验机器上编译好然后搬到生产环境上的机器上吧。然后我无Fack可说。  
下面是折腾了好几天总结出来解决方案。由于目标机器上的GCC版本过低，所以这次也对GCC进行了升级。  
## 1. GCC的编译升级
GCC编译安装依赖以下三个库:

- gmp
- mpfr
- mpc  

### 1.1 自动配置
运行`contrib/download_prerequisites`这个脚本，可以自动下载配置依赖，可以省很多精力时间
```bash
tar -xzvf gcc-5.5.0.tar.gz
cd gcc-5.5.0
./contrib/download_prerequisites
```
### 1.2 编译安装
然后编译安装
```bash
mkdir gcc-build
cd gcc-build
../configure --prefix=$HOME/install/gcc-5.5.0 -enable-checking=release -enable-languages=c,c++ -disable-multilib
make -j4
make install
```

## 2. 使用新的GCC编译Glibc
需要注意的是，上一步编译的GCC需要把bin文件夹和lib文件夹键入环境变量
```bash
export PATH=$HOME/install/gcc-5.5.0/bin:$PATH
export LIBRARY_PATH=$HOME/install/gcc-5.5.0/lib64
```
查看一下是否能够找到新的GCC
```
gcc -v
```
下面就是编译安装Glibc
``` bash
tar -xvf glibc-2.17.tar.gz
cd glibc-2.17
mkdir build
cd build
../configure --prefix=$HOME/install/glibc --disable-profile --enable-add-ons
make && make install
```

## 3. 使用新的Glibc调用程序
在安装目录下的`lib`目录下有个`ld-2.17.so`文件，这个其实是一个程序而不是一个动态链接库，下面我们就要用这个程序来调用启动需要新Glibc库的程序，调用方式如下：
```bash
$HOME/install/glibc/lib/ld-2.17.so --library-path $HOME/install/glibc/lib:$HOME/install/gcc-5.5.0/lib64:/lib64:/lib:/usr/lib64:/usr/lib <command>
```
需要注意的是，`<command>` 必须是可执行文件的绝对路径。但是有个取巧的办法是用`` `which <command>` ``
``` bash
$HOME/install/glibc/lib/ld-2.17.so --library-path $HOME/install/glibc/lib:$HOME/install/gcc-5.5.0/lib64:/lib64:/lib:/usr/lib64:/usr/lib `which ls`
```
那么为了方便，可以上面的调用方式写在一个脚本里  
新建一个脚本文件`new_glibc`，写入以下内容
```bash
$HOME/install/glibc/lib/ld-2.17.so --library-path $HOME/install/glibc/lib:$HOME/install/gcc-5.5.0/lib64:/lib64:/lib:/usr/lib64:/usr/lib `which $1`
```
添加可执行权限，并且把`new_glibc`所在路径添加到`PATH`  
```bash
chmod +x new_glibc
export PATH=$(pwd):$PATH
```
然后就可以这样使用了
```bash
new_glibc ls
```
**注: glibc2.17使用这种方式调用anaconda python3.6会出现段错误。但是升级系统库则不会。**

## 4. 源码安装Python3
由于上面的调用方式经试验，无法调用anaconda版的python3.6，所以我们需要手动编译Python3，不怕，gcc都编译了还怕啥。  
确保新版的GCC的相关东西都在环境变量里。  
``` bash
wget https://www.python.org/ftp/python/3.6.6/Python-3.6.6.tgz
tar zxvf Python-3.6.6.tgz
cd Python-3.6.6
./configure --prefix=$HOME/install/python3
make
make install
```
编译好后把Python的路径添加到PATH中
```bash
export PATH=$HOME/install/python3/bin:$PATH
```
接下来就是安装需要的工具，比如Mxnet
```bash
pip3 install mxnet
```
然后就可以用上述的方式调用了
```bash
new_glibc python3
```
进入python3环境后，试试`import mxnet`，看看能不能成功
## 5. 移植  

将上面编译好的glibc库和gcc库移植到新机器即可，调用方式和上面一样。注意环境变量的路径。  

---
## 6. Gunicorn部署方案
上述方式在调用安装到python中的一些应用时会报错，比如`new_glibc pip3`就会报错，具体原因不是太清楚，一般来说不会这样使用，但是如果使用`gunicorn`部署python项目的话，就会出现这个问题。针对`gunicorn`也是有解决方案的。  
在查看`gunicorn`源码后，发现命令行调用后主要是通过`gunicorn/app/wsgiapp.py`下的`run`方法，那么解决方案就来了，把这个文件单独拿出来，然后这样运行
```bash
new_glibc python3 wsgiapp.py .....
```