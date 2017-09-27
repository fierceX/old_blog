---
title: "Gentoo安装笔记"
date: 2016-09-27
draft: true
---
## 准备工作
首先下载所需的文件：  
1. 最小化安装镜像:install-amd64-minimal-20150903.iso  
2. Stage Tarball快照包:stage3-amd64-20150903.tar.bz2  
3. 系统Portage快照包:portage-20150828.tar.bz2  
## 磁盘分区
首先用虚拟机加载最小化安装镜像开机进入光盘系统，并开启ssh服务：  
`/etc/init.d/sshd start`  
并设定密码用来ssh登陆：  
`passwd`  
然后使用ssh登陆虚拟机  
### 1. 划分磁盘  
首先要先划分磁盘，这里划分三个区，启动分区boot，交换分区swap，还有根分区\使用fdisk这个分区工具进行分区  
* /dev/sda1 启动分区  
* /dev/sda2 交换分区  
* /dev/sda3 根分区  
以下是具体划分磁盘与步骤：  
`fdisk /dev/sda`  
首先创建boot分区：  
```bash  
n
p
1
(回车)
+200M
```
这个分区是boot分区，我们要把这个分区设置成启动分区：  
```bash
a
1
```
然后创建交换分区：  
```bash
n
p
2
(回车)
+1G
```
这个是我们的交换分区swap，我们要把分区类型改成交换分区的类型，82  
```bash
t
2
82
```
然后下面我们建立根分区:  
```bash
n
p
3
(回车)
(回车)
```
完成以上步骤，我们就创建好了三个分区，键入p可以查看分区表  
### 2. 格式化分区并挂载分区
格式化分区：  
```bash
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sda3
mkswap /dev/sda2
```
挂载分区  
```bash
swapon /dev/sda2
mount /dev/sda3 /mnt/gentoo
mkdir /mnt/gentoo/boot
mount /dev/sda1 /mnt/gentoo/boot
```
注意，这里挂载分区顺序不要出错，否则会出现挂载不上的情况   
## 安装基本系统
### 1. 安装Stage和Portage
首先安装Stage Tarball,进入Gentoo的挂载点  
`cd /mnt/gentoo`  
用FTP将事先下载好的Stage包传到虚拟机里然后解压：  
`tar xvjpf stage3-*.tar.bz2`  
注意解压参数不要出错**xvjpf**  
然后再用FTP将事先安装好的Portage包传到虚拟机里解压安装：  
`tar xvjf portage-*.tar.bz2 -C /mnt/gentoo/usr`  
注意解压参数“-C”的**C**为大写    
### 2. 配置镜像站和设置必要信息
接下来配置镜像站：  
`mirrorselect -i -o >> /mnt/gentoo/etc/portage/makd.conf`
`mirrorselect -i -r -o >> /mnt/gentoo/etc/portage/make.conf`
拷贝DNS设置到系统中，以便进入新系统后可以正常上网：  
`cp -L /etc/resolv.conf /mnt/gentoo/etc/`  
将几个重要的当前目录挂载到新系统下：  
```bash
mount -t proc none /mnt/gentoo/proc
mount --rbind /sys /mnt/gentoo/sys
mount --rbind /dev /mnt/gentoo/dev
```
### 3. 进入新系统
然后切换到新系统中：  
```
chroot /mnt/gentoo /bin/bash
source /etc/profile
export PS1="(chroot)$PS1"
```
更新Portage树：  
`emerge --sync`  
### 4. 编译安装内核
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
编译好了看一下编译出来的内核文件名，后面要用到：  
`ls -a /boot`  
### 5. 必要的配置
安装vim，vi不好用，我习惯vim  
`emerge vim`  
查看磁盘挂载列表是否正确，一般都是正确的：  
`vim /etc/fstab`  
设置root用户密码：  
`passwd`  
输入两次密码  
下载启动管理器grub:  
`emerge grub`  
注意：这里安装编译的是grub2而不是grub，所以用一下命令安装grub2到磁盘并生成配置文件：  
```bash
grub2-install --no-floppy /dev/sda
grub2-mkconfig -o /boot/grub/grub.cfg
```
查看grub配置文件中的内核文件是否正确：  
`vim /boot/grub/grub.cfg`  
系统安装就完成了，然后我们需要安装一些必要的软件：  
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
到这里基本系统已经安装完成了，首先退出挂载：  
`exit`  
然后回到光盘主目录卸载刚刚挂载的文件系统  
```bash
umount /mnt/gentoo/boot
umount /mnt/gentoo/dev
umount /mnt/gentoo/proc
umount /mnt/gentoo
```
关机重启：  
`reboot`  


