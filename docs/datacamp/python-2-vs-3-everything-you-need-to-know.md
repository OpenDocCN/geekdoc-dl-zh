# Python 2 vs 3:您需要知道的一切

> 原文：<https://web.archive.org/web/20221129041202/https://www.datacamp.com/blog/python-2-vs-3-everything-you-need-to-know>

![Python 2 vs 3](img/7e2c22c59122b646d6f41f7ddd9d51ca.png)

如果你正在考虑进入数据科学领域，你可能听说过 Python。Python 是一种开源的通用编程语言，广泛应用于数据科学和其他软件领域，如 web 和游戏开发、网络安全和区块链。

Python 的流行在最近几年蓬勃发展。它在各种编程语言流行指数中排名第一，包括 [TIOBE](https://web.archive.org/web/20221212135819/https://www.tiobe.com/tiobe-index) 指数和 [PYPL](https://web.archive.org/web/20221212135819/https://pypl.github.io/PYPL.html) 指数。你可以在另一篇文章中了解更多关于[Python 用于](https://web.archive.org/web/20221212135819/https://www.datacamp.com/blog/what-is-python-used-for)的内容。

由于其简单易读的语法，Python 经常被认为是新程序员最容易学习的编程语言之一。如果你是数据科学的新手，不知道先学哪种语言，Python 是最好的选择之一。现在，您可以参加 DataCamp 的[Data Scientist with Python](https://web.archive.org/web/20221212135819/https://www.datacamp.com/tracks/data-scientist-with-python)career track，开始您的数据科学之旅。

然而，事情可能会变得有点混乱。您可能也听说过 Python 2 和 3。这是什么？难道没有一种单一的 Python 编程语言吗？你应该学哪一个？无需强调:除了特殊情况，到目前为止，您将一直使用 Python 3。无论如何，知道它们的区别总是好的。

在接下来的章节中，我们将解释什么是 Python 2 和 3，它们之间的主要区别，以及哪一个是最值得学习和使用的。

## 什么是 Python 2

Python 是由吉多·范·罗苏姆在 20 世纪 80 年代末开发的，并于 1991 年首次公之于众。经过九年的发展和普及，Python 2.0 于 2000 年发布。

Python 2 附带了一个全新的技术规范，名为 [Python 增强提案](https://web.archive.org/web/20221212135819/https://www.datacamp.com/tutorial/pep8-tutorial-python-code) (PEP)，旨在为编写 Python 代码提供指南和最佳实践。它还提供了一些新特性，比如列表理解、Unicode 支持和循环检测垃圾收集器。

但是 Python 2 中最重要的变化是开发过程本身。Python 被认为是一种易于初学者学习的编程语言。为了实现这个目标，负责开发 Python 的团队——以 Guido ban Rossum 为首——决定转向更加透明和社区支持的开发过程。

Python 2 随着时间的推移继续发展。连续的版本给编程语言增加了新的功能。Python 2 的最新版本是 Python 2.7，发布于 2010 年。对该版本的支持于 2020 年 1 月 1 日结束。

## 什么是 Python 3

Python 3 是下一代编程语言。它是在 2008 年 12 月发布的，附带了一些改进和新特性。

Python 3 不仅仅是调试后的另一个版本的 Python 2 代码。新版本彻底改变了语言，以解决以前版本中的安全问题和设计缺陷。Python 3 提供了一种新的语法，旨在防止冗余或重复的代码，即以不同方式执行相同任务的代码。通过提供单一、清晰的做事方式，Python 3 的易用性和可读性有了很大的提高。

Python 3 中的一些主要变化包括将打印声明改为内置函数、改进整数除法和改进 Unicode 支持。这些变化的本质是 Python 3 与 Python 2 不兼容，换句话说，它是向后不兼容的。

## 为什么会有不同的 Python 版本？

让我们看一些编码例子来说明 Python 2 和 3 的区别！

### 新的 print()函数

Python 2 中的 print 语句已经被 Python 3 中的 print()函数所取代，这意味着我们必须将想要打印的对象放在括号中。

Python 2:

```py
>>> print 'Hello DataCamp'
Hello DataCamp
```

Python3:

```py
>>> print('Hello Datacamp')
Hello Datacamp
```

### 整数除法

在 Python 2 中，当除以整数时，结果总是整数。在 Python3 中，结果总是包含小数，使得整数除法更加直观。

Python 2:

```py
>>> 5/2
2
```

Python 3:

```py
>>> 5/2
2.5
```

### Unicode 支持

在 Python 2 中，每个 Unicode 字符串都必须用前缀“u”标记，因为默认情况下它使用 ASCII 字符。相比之下，Python 3 默认将字符串存储为 Unicode，这比 ASCII 字符串更通用。

Python 2:

```py
>>> print type("Hello DataCamp!") # this is an ASCII string
<type 'str'>

>>> print type(u"Hello DataCamp!") # this is a Unicode string
<type 'unicode'>
```

Python 3:

```py
>>> print(type("Hello DataCamp!")) # this is a Unicode string
<class 'str'>
```

### 距离函数

Python 2 中的 xrange()函数在 Python 3 中不再存在。它已被 range()函数所取代，该函数提高了遍历序列时的性能

Python 2:

```py
>>> xrange(1,10)
xrange(1, 10)
>>> type(xrange(1,10))
<type 'xrange'>
```

Python 3:

```py
>>> range(1,10)
range(1, 10)
>>> type(range(1,10))
<class 'range'>
```

在下表中，您可以找到 Python 2 和 3 之间的主要区别。

|   | Python 2 | python3 |
| 发布日期 | Two thousand | Two thousand and eight |
| 句法 | 更加复杂和难以解释 | 易读易懂 |
| 表演 | 设计缺陷导致性能下降 | 与 Python 2 相比，提高了代码运行时的性能 |
| 打印功能 | 打印“欢迎来到数据营” | 印刷(“欢迎来到数据营”) |
| 整数除法 | 结果是一个整数值。小数总是被截断 | 结果总是一个浮点值 |
| Unicode 支持 | 默认情况下使用 ASCII 字符。要存储 Unicode 值，您需要使用“u”来定义它们 | 字符串的默认存储方式是 Unicode |
| 范围 | xrange()函数创建一个数字序列 | range()函数在迭代时比 xrange()更有效 |
| 向后兼容性 | 将 Python2 移植到 Python 3 相对容易。 | Python 3 不向后兼容 python 2 |
| 图书馆 | Python 2 的许多旧库不是向前兼容的 | Python 3 的大多数新库不能在 python 2 中使用 |

## 可以把 Python 2 转换成 3 吗？

进化是每一种编程语言的重要特征。语言随着时间的推移而发展，以提高性能并保持与用户的相关性。从 Python 2 到 3 的迁移是 Python 历史上最大的变化。

虽然公司和开发人员最终将其代码库迁移到 Python 3 是有意义的——特别是在对 Python 2 的支持结束之后——但将代码从旧版本移植到新版本可能是一个困难和令人生畏的过程。这在 Python 中尤为严重，因为许多 Python 2 库并不是向前兼容的。

幸运的是，有一些资源可以帮助将 Python 2 代码转换成 Python3。

*   [2to3](https://web.archive.org/web/20221212135819/https://docs.python.org/3/library/2to3.html) :一个 Python 程序，它获取 Python 2.x 源代码，并应用一系列修复程序将其转换成有效的 Python 3.x 代码。
*   [Python-Future](https://web.archive.org/web/20221212135819/https://python-future.org/#:~:text=python%2Dfuture%20is%20the%20missing,Python%203%20with%20minimal%20overhead.) :允许您使用一个干净的兼容 Python 3.x 的代码库，以最小的开销同时支持 Python 2 和 Python 3。
*   [六](https://web.archive.org/web/20221212135819/https://pypi.org/project/six):一个 Python 2 和 3 兼容库。它提供了一些实用函数来消除 Python 版本之间的差异，目的是编写与两个 Python 版本都兼容的 Python 代码。

## Python 2 vs 3:哪个最好？

虽然这在几年前是一个有争议的问题，但今天，毫无疑问 Python 3 是一个更好的选择。首先，从 2020 年开始不再支持 Python 2。所以用 Python 3.x 写新项目是有意义的，其次，由于不再支持 Python 2，所以所有的开发都在 Python 3 这边。在每次新的升级之后，该语言都经历了不断的改进(最新版本是 Python 3.10.5，并且已经有了 Python 3.11 的测试版，它具有[重要的新特性](https://web.archive.org/web/20221212135819/https://www.datacamp.com/blog/whats-new-in-python-311-and-should-you-even-bother-with-it))。

在这种背景下，Python 3 的大量采用就不足为奇了。根据 JetBrains 进行的 2021 年 Python 开发者调查，平均而言，使用 Python 3 的 Python 开发者比例为 95%。此外，Python 2 用户的份额每年减少 5 个百分点。

采用 Python 2 还是 Python 3

![Adoption of Python 2 vs Python 3](img/e06ad7bb464d5ee56967e1fd3a44b0e4.png)

来源: [JetBrains](https://web.archive.org/web/20221212135819/https://lp.jetbrains.com/python-developers-survey-2021)

## 使用哪个 Python 版本？

Python 3 是当今最好的 Python 版本。放手一搏是最安全的选择，尤其是对于新手程序员。自从停止支持以来，Python 2 很快就失去了活力，越来越多的公司将他们的代码迁移到 Python 3。鉴于它在许多软件领域不断增长的需求、广泛的使用和大量社区支持的库，学习 Python 3 是有意义的。

然而，需要注意的是，在某些情况下，您仍然需要使用 Python 2。首先，许多公司仍然有用 Python 2 编写的遗留代码库，很难移植到 Python 3。如果你正在工作的公司或项目是这种情况，你需要学习 Python 2。第二，有一些软件领域，比如 DevOps，仍然需要 Python 2。

无论如何，如果你正在考虑在数据科学领域开始新的职业生涯，Python 3 是明智的选择。有了 DataCamp，一切都为您准备好了。查看我们的 Python 课程和教程，立即开始您的数据之旅！

*   [Python 编程技能轨迹](https://web.archive.org/web/20221212135819/https://www.datacamp.com/tracks/python-programming)
*   [Python 程序员职业轨迹](https://web.archive.org/web/20221212135819/https://www.datacamp.com/tracks/python-programmer)
*   [编写高效的 Python 代码教程](https://web.archive.org/web/20221212135819/https://www.datacamp.com/courses/writing-efficient-python-code?hl=GB)

与 Python 2 相比，Python 3 是无可争议的赢家，被广泛采用。

数据科学最流行的编程语言是 Python 和 r。data camp 有一个庞大的目录 [Python 课程](https://web.archive.org/web/20221212135819/https://www.datacamp.com/learn/python) 可以帮助你入门

是的！在几乎所有的测试中。Python 3 通常比 Python 2 快。

Python 2.7。2020 年，该版本支持维护结束。

与 Python 2 相比，Python 3 是一种更好的语言，并提供了一套更好的标准库。另外，自 2020 年以来，语言和标准库仅在 Python 3 中有所改进。

有一些资源可以用来将代码更新到 Python 3，比如 2to3、Python-Future 和 Six。

是的。您可以通过为 Python 2 和 3 维护单独的虚拟环境来做到这一点。