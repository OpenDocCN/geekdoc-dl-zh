<!--yml

category: 未分类

date: 2024-07-01 18:17:31

-->

# 通过禁用 mDNS 来降低 Ubuntu 的延迟：ezyang 博客

> 来源：[`blog.ezyang.com/2012/03/reduce-ubuntu-latency-by-disabling-mdns/`](http://blog.ezyang.com/2012/03/reduce-ubuntu-latency-by-disabling-mdns/)

## 通过禁用 mDNS 来降低 Ubuntu 的延迟

这是一个非常快速和简单的修复，使我维护的 Ubuntu 服务器的延迟从*三到四秒*降至瞬间。如果您注意到 ssh 或 scp（甚至像 remctl 这样的其他软件）存在高延迟，并且您可以控制您的服务器，请在服务器上尝试：`aptitude remove libnss-mdns`。原来 Ubuntu 上的多播 DNS 存在[长期存在的 bug](https://bugs.launchpad.net/ubuntu/+source/nss-mdns/+bug/94940)，他们没有正确调整超时，导致 IP 没有名称时进行反向 DNS 查找的性能极差。

移除多播 DNS 将会破坏一些依赖多播 DNS 的应用程序；不过，如果您正在运行 Linux，*可能*不会注意到这一点。我在上述链接的 bug 中列出了一些其他解决方案，您也可以尝试。
