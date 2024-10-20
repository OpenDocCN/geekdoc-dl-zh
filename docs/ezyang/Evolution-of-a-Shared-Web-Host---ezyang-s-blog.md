<!--yml

category: 未分类

date: 2024-07-01 18:18:08

-->

# 共享 Web 主机的演变：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/09/evolution-of-a-shared-web-host/`](http://blog.ezyang.com/2010/09/evolution-of-a-shared-web-host/)

## 共享 Web 主机的演变

*爱德华继续发布他的系统文章。波士顿的空气中一定有什么东西。*

昨天，我在 [SIPB 线索导论](http://cluedumps.mit.edu/wiki/SIPB_Cluedump_Series) 上介绍了 [scripts.mit.edu](http://scripts.mit.edu) 的使用和实施，这是 SIPB 为 MIT 社区提供的共享主机服务。我几乎所有的系统管理员经验都来自于帮助维护这项服务。

> Scripts 是 SIPB 为 MIT 社区提供的共享托管服务。然而，它所做的远不止普通的 $10 主机：哪些共享托管服务可以直接集成到你的 Athena 帐户中，通过 Linux-HA 管理的服务器集群上复制你的网站，让你请求 *.mit.edu 的主机名，提供常见 Web 软件的自动安装，允许你自定义并为你进行升级？Scripts 是一个蓬勃发展的开发平台，拥有超过 2600 名用户和许多有趣的技术问题。

我最终将演讲分为两个部分：一个简短的[高级用户脚本](http://web.mit.edu/~ezyang/Public/scripts-powerusers.pdf)演示，以及一个更长的技术文章，名为[共享 Web 主机的演变](http://web.mit.edu/~ezyang/Public/scripts-evolution.pdf)。演讲中还分发了一份[速查表](http://web.mit.edu/~ezyang/Public/scripts-cheatsheet.pdf)。

在此演讲中讨论的技术包括 Apache、MySQL、OpenAFS、Kerberos、LDAP 和 LVS。
