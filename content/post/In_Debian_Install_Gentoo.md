---
title: "Debian下安装Gentoo"
date: 2016-10-27
draft: true
---
之前用虚拟机安装了Gentoo，发现安装也不算太费事，主要就是联网，然后就可以开滚了，因为我用的是校园网，所以没有安装好是没法联网的，安装Gentoo更不要说无线了，后来我想到，用ubuntu的盘装Gentoo联网应该好办，然后从网上下了个ubuntu的镜像，解压到U盘（我的是UEFI启动模式，解压到U盘就可以启动）启动以后，发现我们学校的认证客户端无法启动（我们学校用的是锐捷，万恶的锐捷），又没有无线，然后就放弃了，后来我一想，用liveCD可以安装，安装过程中就是分区挂载然后chroot到新系统，那用已经跑在机器上的linux系统不也一样吗，所以我就立马重启进Debian，开始我的探索之旅  
## 准备工作
首先下载所需文件：  
1. Stage Tarball快照包:stage3-amd64-20150903.tar.bz2  
2. 系统Portage快照包:portage-20150828.tar.bz2  
因为我们使用已经安装在机器里的别的Linux系统安装，所以不需要安装镜像，那前提是需要机器里有Linux系统的。  
## 磁盘分区
磁盘分区，可以使用所安装的Linux上的图形化分区工具，也可以用fdisk这个命令行工具，凭各自喜好吧，另外，在已经安装好的Linux上分区，可能需要root权限，如果不想切换到root用户的话，可以使用sudo，没有这个命令的可以百度一下或者切换到root用户，不再多说  
### 1. 划分磁盘
首先要有一个根分区，然后启动分区可以不分，交换分区可以用现有系统上的交换分区，所以，我图省事就分了一个根分区，分区命令我就不写了，不会的可以用图形化分区工具。  
### 2. 格式化分区并挂载分区
格式化分区我也就不多说了，格式化成ext4，我说一下挂载，因为我们是在现有的系统，所以挂载命令可能需要root权限，为了方便，可以在自己的主目录新建一个文件夹：  
`madir gentoo`  
然后挂载到这个目录上:  
`sudo mount /dev/sda11 gentoo`  
我的分区是sda11，注意看一下自己新建的分区是多少，因为我没有分boot分区，所以就在跟分区下新建一个boot目录  
`mkdir gentoo/boot`  
交换分区就不用挂载了，我不觉得你正在跑的系统没有交换分区，如果没有，那Gentoo也不需要有  
## 安装基本系统
### 1. 安装Stage和Portage
和正常安装一样，首先安装Stage Tarball，进入挂载好的gentoo目录，把下载好的Stage包复制到gentoo目录下，注意挂载的时候是用root权限挂载的，所以对这个目录写文件同样使用root权限：  
```bash
cd gentoo
sudo cp ../stage3-amd64-20150903.tar.bz2 ./
sudo cp ../portage-20150828.tar.bz2 ./
sudo tar xvjpf stage3-*.tar.bz2
sudo tar xvjf portage-*.tar.bz2 -C /mnt/gentoo/usr
```
注意解压参数第一个是**xvjf**第二个是**xvjpf**并且"-C"中的**C**是大写  
### 2. 配置镜像站和设置必要的信息
因为我们用的是现有的系统，所以没有测试软件源速度的工具，但是，现有系统用的是哪个镜像站的软件源，Gentoo就使用哪个，比如我的用的是中科大的：  
`sudo echo GENTOO_MIRRORS="http://mirrors.ustc.edu.cn/gentoo/" >> etc/portage/makd.conf`  
然后拷贝DNS设置到新系统中：  
`cp -L /etc/resolv.conf etc/`  
最后挂载几个重要的目录：  
```bash
sudo mount -t proc none proc
sudo mount --rbind /sys sys
sudo mount --rbind /dev dev
```
### 3. 进入新系统
进入系统需要root权限，别忘了你挂载的时候是用的root权限,  
```bash
cd ..
sudo chroot ./gentoo bin/bash
source /etc/profile
export PS1="(chroot)$PS1"
```
好了，我们现在进入这个新系统了，不需要干嘛都加sudo了。  
### 4.编译安装内核
查看可用的角色：  
`eselect profile list`  
选择桌面系统，默认的是1，我选择3，桌面环境  
`eselect profile set 3`  
安装内核源码：  
`emerge gentoo-sources`  
编译之前，我们先在编译配置文件里加上一句话：  
`vi /etc/portage/make.conf`  
`MAKEOPTS="-j4"`  
安装自动化编译工具：  
`emerge genkernel`  
开始用自动编译工具构建内核：  
`genkernel all`  
编译时间会很长，慢慢等。。。  
### 5. 必要的配置
安装vim，vi不好用，我习惯vim  
`emerge vim`  
查看磁盘挂载列表是否正确，这里我是错误的，记住把你现在用的交换分区写到里面：  
`vim /etc/fstab`  
设置root用户密码：  
`passwd`  
输入两次密码  
我们可以不用安装启动管理器，因我我们可以用现用系统上的启动管理器，更新一下你现用系统上的启动管理器，有惊喜，比如我的，另开一个终端更新grub：  
`sudo update-gurb`  
好了，系统安装就算完成了，安装一些别的软件：  
```bash
emerge syslog-ng
emerge vixie-cron
emerge dhcpcd
emerge sshd
rc-update add syslog-ng default
rc-update add vixid-cron default
rc-update add dhcpcd default
rc-update add sshd default
```
### 6. 结束安装
退出这个新系统：  
`exit`  
然后，然后重启你的机器  

