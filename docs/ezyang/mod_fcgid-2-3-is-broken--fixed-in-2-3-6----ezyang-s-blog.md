<!--yml

category: 未分类

date: 2024-07-01 18:18:03

-->

# mod_fcgid 2.3 是有问题的（在 2.3.6 中修复）：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/11/mod_fcgid-is-broke/`](http://blog.ezyang.com/2010/11/mod_fcgid-is-broke/)

## mod_fcgid 2.3 是有问题的（在 2.3.6 中修复）

这篇文章旨在为一个问题获得谷歌关键词排名，该问题基本上阻止了 [Scripts](http://scripts.mit.edu) 从 Fedora 11 切换到 Fedora 13。新机群一直因为负载而崩溃，我们一直在琢磨，“为什么？”

结果，[以下提交](http://svn.apache.org/viewvc?view=revision&revision=753578) 在相当可怕的方式下破坏了 mod_fcgid：基本上，mod_fcgid 无法管理运行中的 FastCGI 进程池，因此它会不断生成新的进程，直到系统内存耗尽。这在拥有大量生成虚拟主机的系统中尤为明显，例如使用 mod_vhost_ldap 的用户。这在上周发布的 mod_fcgid 2.3.6 中得到了修复。

*与此无关的*，我一直在头脑中转悠着一系列*计算机科学哲学*的文章，试图在我们领域之外的一些有趣哲学问题中进行识别（咳咳 AI 咳咳）。希望能引入一些传统上与科学哲学、数学哲学、生物学哲学等相关的主题，也许提出一些我自己的问题。哲学家们最喜欢提出听起来合理的理论，然后提出一些令人困惑的例子，似乎能够打破它们，听起来这本身就可以产生一些禅宗公案，这总是很有趣的。
