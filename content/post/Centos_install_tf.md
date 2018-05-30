---
title: "CentOS安装TensorFlow笔记"
date: 2018-05-30
draft: true
---
最近公司需要在多台CentOS上部署TensorFlow用来并行推断，因为CentOS上的一些库比较老旧，没法安装TensorFlow，所以将安装流程记录下来，以备后用。
### 1. 安装miniconda版本的python3
``` bash
>> wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
>> bash Miniconda3-latest-Linux-x86_64.sh
```
安装的时候会询问一些东西，一律yes和回车就行
### 2. 检查glibc共享库版本
``` bash
>> ll /lib64/libc.so.6
lrwxrwxrwx. 1 root root 12 Jun 13  2015 /lib64/libc.so.6 -> libc-2.12.so
```
如果是版本低于2.17,则需要手动编译安装glibc
``` bash
>> wget https://ftp.gnu.org/gnu/glibc/glibc-2.17.tar.gz
>> tar -xvf glibc-2.17.tar.gz
>> cd glibc-2.17
>> mkdir build
>> cd build
>> ../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin
>> make && make install
```
### 3. 检查libstdc++.so.6版本
``` bash
>> strings /usr/lib64/libstdc++.so.6 | grep 'CXXABI'
CXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
```
如果没有1.3.7版本，则需要安装新版本,首先查看第一步安装的miniconda目录下的libstdc++版本
``` bash
>> ls ./miniconda3/lib/libstdc++.so.6*
/home/admin/miniconda3/lib/libstdc++.so.6  /home/admin/miniconda3/lib/libstdc++.so.6.0.24
```
把该目录下的文件拷贝至系统库目录，目前的版本为`libstdc++.so.6.0.24`
```
>> cp ./miniconda3/lib/libstdc++.so.6.0.24 /usr/lib64/
>> rm -rf /usr/lib64/libstdc++.so.6
>> ln -s /usr/lib64/libstdc++.so.6.0.24 /usr/lib64/libstdc++.so.6
```

### 4. 安装tensorflow
``` bash
>> pip install tensorflow==1.4.0 flask
```

