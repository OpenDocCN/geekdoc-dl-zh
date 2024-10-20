<!--yml

category: 未分类

date: 2024-07-01 18:18:25

-->

# Python 中的变异追踪：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/03/mutation-sleuthing-in-python/`](http://blog.ezyang.com/2010/03/mutation-sleuthing-in-python/)

Python 是一种赋予你很大自由度的语言，特别是任何特定的封装方案都只是弱约束，可以被足够精明的黑客绕过。我属于“我的编译器应该阻止我做愚蠢事情”的阵营，但我肯定会说，动态能力确实非常方便。但问题来了：*语言必须告诉你，你做了什么愚蠢的事情*。

在这种情况下，我们希望看到当你错误地改变了一些内部状态时。你可能会嗤之以鼻地说：“嗯，我知道当*我*改变*我的*状态时”，但当你调试两个你没有编写的第三方库之间的交互时，情况肯定不是这样。具体来说，我应该能够指向一个变量（它可能是一个局部变量、全局变量或类/实例属性），并对 Python 说：“告诉我这个变量何时发生了变化。”当变量改变时，Python 应该告诉我谁改变了变量（通过一个回溯）以及变量变成了什么。我应该能够说：“告诉我这个变量何时改变成了这个值。”

好的，这里有一个小模块可以做到这一点：[变异侦察员](http://github.com/ezyang/mutsleuth/blob/master/mutsleuth.py)。导入这个模块并通过 `mutsleuth.watch` 传递一个表达式，这个表达式会评估为你想要检查的变量。

这里有一个例子：假设我有以下文件：

`good.py`：

```
b = "default value"

```

`evil.py`：

```
import good
good.b = "monkey patch monkey patch ha ha ha"

```

`test.py`：

```
import mutsleuth
mutsleuth.watch("good.b")

import good
import evil

```

当你运行 test.py 时，你将得到以下的追踪：

```
ezyang@javelin:~/Dev/mutsleuth$ python test.py
Initialized by:
  File "test.py", line 5, in <module>
    import evil
  File "/home/ezyang/Dev/mutsleuth/good.py", line 1, in <module>
    b = "good default value"
Replaced by:
  File "test.py", line 5, in <module>
    import evil
  File "/home/ezyang/Dev/mutsleuth/evil.py", line 2, in <module>
    good.b = "monkey patch monkey patch ha ha ha"

```

有几个注意事项：

+   跟踪不会开始，直到你进入另一个本地作用域，无论是调用一个函数还是导入一个模块。对于大多数较大的应用程序，你肯定会获得这个作用域，但对于一次性脚本可能不是这样。

+   为了保持性能的可接受性，我们仅对实例之间进行浅层比较，因此你需要专门关注一个值，以获取有关它的真实变异信息。

欢迎提交 Bug 报告、建议和改进！我测试过了，找到了一个旧 Bug，我本来希望有这个模块（它涉及由两个不同站点初始化的日志代码被初始化两次），并验证了它可以工作，但我还没有“冷”测试过它。

感谢 [Bengt Richter](http://mail.python.org/pipermail/python-list/2002-September/164261.html) 最初建议这种追踪方式。
