<!--yml

类别：未分类

date: 2024-07-01 18:17:27

-->

# 所以你想在 IMAP 上进行黑客攻击... : ezyang’s blog

> 来源：[`blog.ezyang.com/2012/08/so-you-want-to-hack-on-imap/`](http://blog.ezyang.com/2012/08/so-you-want-to-hack-on-imap/)

## 所以你想在 IMAP 上进行黑客攻击...

(Last IMAP themed post for a while, I promise!)

首先，你的信息是错误的：你实际上**不**想要在 IMAP 上进行黑客攻击。但是假设，出于某种 masochistic 的原因，你需要深入研究你的邮件同步程序并修复 bug 或添加一些功能。在开始旅程之前，有几件有用的事情需要知道...

+   阅读你的 RFC。[RFC 3501](http://tools.ietf.org/html/rfc3501) 是实际规范，而 [RFC 2683](http://tools.ietf.org/html/rfc2683) 给出了许多有助于解决实践中存在的 IMAP 服务器棘手问题的建议。你还应该了解 UIDPLUS 扩展，[RFC 4315](http://tools.ietf.org/html/rfc4315)，它被广泛支持，极大地简化了客户端实现者的生活。

+   IMAP 幸运地是一个基于文本的协议，因此你可以在命令行上进行实验。一个很好的工具是 `imtest`，它有各种花哨的功能，如 SASL 认证。（不要忘记用 `rlwrap` 包装它！）确保在你的命令前加上标识符（`UID` 是一个有效的标识符，所以输入 `UID FETCH ...` 不会做你想要的事情。）

+   通常使用 UID 而不是序列号是个更好的主意，因为它们更稳定，但要小心：根据规范，以 `UID` 为前缀的命令*永远不会*失败，因此你需要检查响应中的未标记数据，以查看是否实际发生了任何事情。（如果你有一个糟糕的 IMAP 库，它可能不会在请求之间清除未标记的数据，因此要小心陈旧的数据！）哦，还要查一下 `UIDVALIDITY`。

+   有很多软件与 IMAP 接口，多年来为了应对 IMAP 服务器的 bug 累积了许多特例。值得一探究竟，以便了解需要处理的问题类型。
