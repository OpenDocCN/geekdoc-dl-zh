<!--yml

category: 未分类

date: 2024-07-01 18:17:24

-->

# 功能加密：ezyang's 博客

> 来源：[`blog.ezyang.com/2012/11/functional-encryption/`](http://blog.ezyang.com/2012/11/functional-encryption/)

## 功能加密

最近，Joe Zimmerman 向我分享了一种关于各种加密方案的新思路，称为*功能加密*。更深入地阐述了这一概念的是丹·博内等人在一篇非常易于理解的[最新论文](http://eprint.iacr.org/2010/543.pdf)中。下面是摘录的摘要第一段：

> 我们通过给出概念及其安全性的精确定义，开始了对功能加密的正式研究。粗略地说，功能加密支持受限制的密钥，使密钥持有者能够学习加密数据的特定函数，但不会了解数据的其他信息。例如，给定一个加密程序，密钥可能使持有者能够学习在特定输入上程序的输出，而不会了解程序的其他任何信息。

值得注意的是，功能加密泛化了许多现有的加密方案，包括[公钥加密](http://en.wikipedia.org/wiki/Public-key_cryptography)、[基于身份的加密](http://en.wikipedia.org/wiki/ID-based_encryption)和[同态加密](http://en.wikipedia.org/wiki/Homomorphic_encryption)。不幸的是，在某些安全模型中，功能加密总体上存在一些不可能的结果（链接的论文对仿真模型有一个不可能的结果）。[功能加密](http://en.wikipedia.org/w/index.php?title=Functional_encryption&action=edit&redlink=1)还没有维基百科页面；也许你可以写一个！

*说来也奇怪，* 我的一位数学博士朋友最近问我：“你认为 RSA 有效吗？” 我说：“不，但也许目前没有人知道如何破解它。” 然后我问他为什么这么问，他提到他正在上密码学课程，考虑到所有的假设，他很惊讶其中任何一个都能工作。我回答说：“是的，听起来大概是这样。”
