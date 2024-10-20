<!--yml

category: 未分类

date: 2024-07-01 18:17:38

-->

# 如何在 Ubuntu 上构建 i686 glibc：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/12/how-to-build-i686-glibc-on-ubuntu/`](http://blog.ezyang.com/2011/12/how-to-build-i686-glibc-on-ubuntu/)

## 如何在 Ubuntu 上构建 i686 glibc

一个“简单”的两步过程：

1.  [为 i686 应用此补丁](http://www.eglibc.org/archives/patches/msg00073.html)。（为什么他们还没有在主干中修复这个问题，我不知道。）

1.  使用 `CFLAGS="-U_FORTIFY_SOURCE -fno-stack-protector -O2"` 进行配置（这会禁用 Ubuntu 默认启用的 fortify source 和 stack protection，这些干扰了 glibc。你需要保持优化开启，因为没有优化 glibc 将无法构建。）你需要做一些额外的步骤，比如创建一个单独的构建目录并指定一个前缀。

希望这对其他人有所帮助。如果你想知道为什么我要构建 glibc，那是因为我在 iconv 中报告了这两个 bug：
