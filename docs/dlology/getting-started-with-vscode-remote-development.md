# VS 代码远程开发入门

> 原文：<https://www.dlology.com/blog/getting-started-with-vscode-remote-development/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年三个月前

([评论](/blog/getting-started-with-vscode-remote-development/#disqus_thread))

![remote_dev](img/4a11e2200fbc908430565b7224fb8e12.png)

假设您在云或无头物理机上有一个 GPU 虚拟实例，有几个选项，如远程桌面或 Jupyter notebook，可以为您提供类似桌面的开发体验，但是，VS CODE 远程开发扩展可以比 Jupyter Notebook 更灵活，比远程桌面响应更快。我将一步一步地向你展示如何在 Windows 上设置它。

## 启动 OpenSSH 服务

首先，让我们确保您已经在您的服务器上设置了 SSH，很可能您的在线服务器实例将预先配置 OpenSSH 服务器，下面的命令可以检查它是否正在运行。

```py
service sshd status
```

如果您看到类似这样的内容，您就可以开始了，否则，安装或启动 OpenSSH 服务器

```py
● ssh.service - OpenBSD Secure Shell server
 Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: enabled)
 Active: active (running) since Tue 2019-09-17 19:58:43 CST; 4 days ago
 Main PID: 600 (sshd)
 Tasks: 1 (limit: 1109)
 CGroup: /system.slice/ssh.service
 └─600 /usr/sbin/sshd -D
```

对于 Ubuntu 系统，您可以安装 OpenSSH 服务器，并有选择地像这样更改默认的 22 端口

设置好之后，从您的开发机器使用 IP 地址、用户名和密码 ssh 到这个服务器，只是为了验证没有小故障。

## Windows 上的 OpenSSH 客户端

这一步是无痛的，对于 Windows 10 用户来说，它只是在设置页面中启用一个功能，它可能已经被启用了。无论如何，下面是验证该特性是否启用的步骤。

在设置页面，转到应用程序，然后点击“管理可选功能”，向下滚动并检查“OpenSSH 客户端”已安装。

![openssh1](img/c7890d411af8c207bcac10de4e545de0.png)

![openssh2](img/01a0d0b613fc5294e224a211e59ffdea.png)

![openssh3](img/3e73c609eb608e2c6525f07575604880.png)

## 设置 SSH 密钥

你不想每次登录服务器时都输入用户名和密码吧？

#### 在 Windows(你的开发机)

这里我们将在命令提示符下生成一个 SSH 密钥，

```py
ssh-keygen -t rsa
```

接受默认值，按照提示操作时，您可以将关键阶段留空。

复制该命令的输出，

```py
cat ~/.ssh/id_rsa.pub
```

然后使用用户名和密码 ssh 到服务器(如果还没有的话),然后运行下面的命令行，打开刚才复制到服务器上`~/.ssh/authorized_keys`的内容。

*如果你不熟悉 vi，“Shift+END”到最后，键入“a”进入追加模式，右键粘贴剪贴板的内容。完成后，按“Shift +”然后打“wq”写并相当。希望在此之后，我们不需要在 vi 中以同样的方式编辑代码。*

*要验证 SSH 是否设置好了，在您的 Windows 机器上启动一个新的命令行提示符并键入`ssh <username>@<server ip>`，它应该会自动登录而不需要询问密码。*

 *## 安装远程开发 VS 代码扩展

打开 VSCOD，单击 Extension 选项卡，然后搜索“远程开发”并安装它。

![install_extension](img/fa5d46ac0cd22870e9c620d184aa997a.png)

一旦安装完毕，你会看到一个名为“远程浏览器”的新标签，点击它和齿轮按钮。

![remote_explorer](img/25954496cf7ff630610085661130944e.png)

选择第一个条目，在我的例子中，它就像`C:\Users\hasee\.ssh\config`，一旦你打开它，填写别名、主机名和用户。别名可以是任何有助于记忆的文本，主机名可能是远程机器的 IP 地址。

一旦你这样做了，只需点击“连接到新窗口中的主机”按钮。

最后一步，在新窗口中点击侧边栏中的“打开文件夹”来选择远程机器上的文件夹路径，然后输入“Ctrl +`”来打开远程机器上的终端，就像在本地做的一样。

## 结论和进一步阅读

现在你有了它，一个快速的教程向你展示如何从头开始设置 VS 代码远程开发，让你在一个无头的远程服务器上享受桌面开发的体验。

[官方 VS 代码远程开发页面](https://code.visualstudio.com/docs/remote/remote-overview)请参考网站。

*   标签:
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/getting-started-with-vscode-remote-development/&text=Getting%20started%20with%20VS%20CODE%20remote%20development) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/getting-started-with-vscode-remote-development/)

*   [←物体检测深度学习的最新进展-第二部分](/blog/recent-advances-in-deep-learning-for-object-detection-part-2/)
*   [如何使用自定义 COCO 数据集训练检测器 2→](/blog/how-to-train-detectron2-with-custom-coco-datasets/)*