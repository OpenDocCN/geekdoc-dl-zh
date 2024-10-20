<!--yml

category: 未分类

date: 2024-07-01 18:17:21

-->

# 一个 Zerocoin 的谜题：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/04/a-zerocoin-puzzle/`](http://blog.ezyang.com/2013/04/a-zerocoin-puzzle/)

## 一个 Zerocoin 的谜题

我很少发布链接垃圾信息，但考虑到我过去曾写过关于[比特币匿名化](http://blog.ezyang.com/2012/07/secure-multiparty-bitcoin-anonymization/)的主题，这个链接似乎很相关：[Zerocoin：使比特币匿名化](http://blog.cryptographyengineering.com/2013/04/zerocoin-making-bitcoin-anonymous.html)。他们的核心创新是在区块链本身中建立一个*持续运行*的混合池；他们通过使用零知识证明来实现这一点。神奇！

这里有一个给本博客读者的难题。假设我是一个想要匿名化一些比特币的用户，并且愿意在兑换我的 Zerocoins 之前等待期望时间*N*。那么，我从哪个正确的概率分布中选择等待时间呢？此外，假设一个 Zerocoin 参与者的群体，他们都使用这个概率分布。进一步假设，每个参与者都有一些效用函数，权衡匿名性和预期等待时间（请随意做出使分析变得容易的假设）。这个群体处于纳什均衡状态吗？
