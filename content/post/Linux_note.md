---
title: "Linux使用笔记"
date: 2014-12-01
draft: true
---
## SSH登录及密钥登录
ssh登录可以用普通的账户加口令登录也可以用密钥登录  
密钥是非加密密钥，一般采用RSA非对称加密算法，分为私钥和公钥，私钥由客户机保存，公钥交给服务器，登录过程中由客户端发送公钥到服务器端，由服务器端验证公钥是否配对，若配对，则由服务器发送由公钥加密的随机数据到客户端，客户端再用私钥解密并把原数据发送给服务器，由服务器验证是否正确来判断登录身份  
linux一般用OpenSSH，生成密钥命令为：  
`ssh-keygen -t rsa`  
会提示公钥和私钥的保存位置以及密钥口令，默认会在用户目录下创建.ssh文件夹并把公钥私钥放到此文件夹下，如果口令为空则是无密码自动登录。  
如果此计算机为远程连接的计算机则把公钥追加到.ssh目录下的authorized_keys文件中，若没有该文件或文件夹可以自己创建并添加公钥到此文件中，注意.ssh文件夹权限为700，authorized_keys文件的权限为600:  
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```
## Debian系语言配置文件  
`vim /etc/default/locale`  
LANG为字符集  
LANGUAGE为语言  
## openSUSE 更改主机名
修改`/etc/hostname`  
然后运行`/etc/rc.d/boot.localnet start`  
## 修改openSUSE启动字符界面  
`/etc/systemd/system`目录下的`default.target`链接到`/usr/lib/systemd/system`的启动参数。  
inittab 六种运行级 （run level）:
0. 停机（记住不要把initdefault设置为0，因为这样会使Linux无法启动）  
1. 单用户模式，就像Win9X下的安全模式。  
2. 多用户，但是没有 NFS 。  
3. 完全多用户模式，标准的运行级。  
4. 一般不用，在一些特殊情况下可以用它来做一些事情。  
5. X11，即进到 X-Window 系统。  
6. 重新启动 （记住不要把initdefault设置为6，因为这样会使Linux不断地重新启动）。  
解除`default.target`的链接：  
`unlink default.target`  
链接到`runlevel3.target`  
`ln -s /usr/lib/systemd/system/runlevel3.target`  
重启就会默认启动字符界面  
## Linux分区工具 fdisk及格式化命令mksf
查看设备下所有磁盘状态：  
`fdisk -l`  
查看某一磁盘状态：  
`fdisk -l /dev/sda`  
键入`fdisk /dev/sda`则进入该磁盘互交状态：  
**m**为帮助  
**n**为新建分区  
会让你选择分区类型，p主分区，e为扩展分区，然后会有分区的起始柱面和截止柱面，起始柱面默认即可，截至柱面可以自定义分区大小  如 `+1G`  
**p**为查看当前磁盘分区状况  
**w**为保存并退出  
**q**为放弃并推出  
**t**为更改分区类型  
**l**为分区类型的编码  
**a**设定磁盘启动区，*为当前启动区  
### 格式化命令mkfs
`mkfs.ext3 /dev/sda2`  
`mkfs /dev/sda3`  
## 挂载命令mount与卸载命令umount以及自动挂载文件
挂载命令`mount`  
挂载光盘到/media/cdrom下  
`mount /dev/cdrom /media/cdrom`  
挂载分区sda2到/media/sda2目录下  
`mount /dev/sda2 /media/sda2`  
卸载命令`umount`  
卸载挂载的光盘  
`umount /media/cdrom`  
卸载挂载在/media/sda2下的磁盘分区  
`umount /media/sda2`  
自动挂载文件位置  
`/etc/fstab`  
Deiban 默认无法写NTFS文件系统的磁盘分区  
安装  `ntfs-3g`  
`apt-get install ntfs-3g`  
直接挂载  
`mount -t ntfs-3g /dev/sda2 /media/D` 
## 磁盘拷贝命令dd  
`dd`用于磁盘拷贝  
`dd if=原磁盘或文件 of=目标磁盘及文件 bs=一次拷贝块的大小 count=拷贝bs数量`  
磁盘对拷  
`dd if=/dev/sdb of=/dev/sdc`  
拷贝成镜像  
`dd bs=512 count=[fdisk命令中最大的end数+1] if/dev/sdb of=name.img`  
在另一个终端输入  
`watch -n 5 killall -USR1 dd`  
可以查看进度  
## 系统恢复
系统恢复模式
输入root密码后进入恢复模式，此状态下的根分区为只读模式，需要重新挂载根分区为读写模式  
`mount -o remount,rw /`  
## ZSH配置文件
zsh 的配置文件放到~/.zshrc中，用zsh作为登录shell可以在.zshrc中添加自定义环境变量
## Make安装软件
1. 下载源码以后运行`./configure`来生成`makefile`文件  
`./configure`  
可以通过加参数对安装加以控制，如`./configure –prefix=/usr`意思就是将该软件安装到`/usr`下面，执行文件就会在`/usr/bin`下，资源文件就会在`/usr/share`下面  
2. `make`编译  
3. `make install` 安装  
出现的一些错误  
`./configure`没有运行权限，赋予`configure`这个文件执行权限  
`chmod 777 configure`  
没有权限在安装目录下建立文件，需要更改用户以提高权限，另外有些代码里有别的工具和`configure`文件一样没有运行权限，需要把该文件加上运行权限。  
