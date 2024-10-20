# 环境变量数据科学家指南

> 原文：<https://web.archive.org/web/20230101103400/https://www.datacamp.com/blog/a-data-scientists-guide-to-environment-variables>

你可能遇到过一个软件要求你允许修改你的`PATH`变量，或者另一个程序的安装说明隐晦地告诉你必须“正确设置你的`LD_LIBRARY_PATH`变量”。

作为一名数据科学家，在与您的计算堆栈交互时，您可能会遇到其他环境变量问题(尤其是如果您不能像我一样完全控制它的话)。这篇文章旨在揭示什么是环境变量，以及它如何在数据科学环境中使用。

## 什么是环境变量？

首先，让我通过深入研究`PATH`环境变量来解释什么是环境变量。我鼓励您在 bash 终端中执行这里的命令(做适当的修改——阅读文本以了解我在做什么！).

当您登录到您的计算机系统时，比方说，通过 SSH 登录到您的本地计算机终端或您的远程服务器，您的 bash 解释器需要知道在哪里寻找特定的程序，比如`nano`(文本编辑器)，或者`git`(您的版本控制软件)，或者您的 Python 可执行文件。这是由 PATH 变量控制的。它指定了可执行程序所在文件夹的路径。

按照历史惯例，命令行程序，比如`nano`、`which`、`top`，都在`/usr/bin`目录下。(按照历史惯例，`/bin`文件夹是存放软件二进制文件的，这就是它们被命名为`/bin`的原因。)这些是与您的操作系统捆绑在一起的，因此需要特殊权限才能升级。

在您的终端中尝试一下:

```py
$ which which
/usr/bin/which
$ which top
/usr/bin/top 
```

其他程序(无论什么原因)被安装到`/bin`中。`ls`就是一个例子:

```py
$ which ls
/bin/ls 
```

还有一些程序可能安装在其他特殊目录中:

```py
$ which nano
/usr/local/bin/nano 
```

您的 Bash 终端如何知道去哪里寻找东西呢？它使用了`PATH`环境变量。它看起来像这样:

```py
$ echo $PATH
/usr/bin:/bin:/usr/local/bin 
```

关于`PATH`变量，要记住的最重要的事情是它是“冒号分隔的”。也就是说，每个目录路径由下一个使用“冒号”(`:`)字符分隔。bash 终端查找程序的顺序是从左到右的:

*   `/usr/bin`
*   `/bin`
*   `/usr/local/bin`

在我的特定计算机上，当我输入`ls`时，我的 bash 解释器将首先查看`/usr/bin`目录。它会发现`ls`在`/usr/bin`中不存在，因此它会移动到下一个目录`/bin`。由于我的`ls`存在于`/bin`下，它将从那里执行`ls`程序。

你可以看到，这对于定制你的计算环境来说是非常灵活的，但是如果一个程序在你不知道的情况下修改了你的`PATH`变量，这也是非常令人沮丧的。

等等，你真的可以修改你的`PATH`变量？是的，有几种方法可以做到这一点。

## 如何修改`PATH`变量

### 使用 Bash 会话

第一种方式是暂时的，只发生在特定的 bash 会话中。通过将文件夹“预先挂起”到`PATH`变量，可以使文件夹具有比现有路径更高的优先级:

```py
$ export PATH=/path/to/my/folder:$PATH
$ echo $PATH
/path/to/my/folder:/usr/bin:/bin:/usr/local/bin 
```

或者，我可以通过将它“附加”到`PATH`变量，使它具有比现有路径更低的优先级:

```py
$ export PATH=$PATH:/path/to/my/folder
$ echo $PATH
/usr/bin:/bin:/usr/local/bin:/path/to/my/folder 
```

这是暂时的，因为我只在当前的 bash 会话中导出它。

### `bashrc`或`.bash_profile`文件

如果我想让我的更改更加永久，那么我会在我的`.bashrc`或`.bash_profile`文件中包含。(我推荐使用`.bashrc`文件。)该`.bashrc` / `.bash_profile`文件位于您的主目录中(您的`$HOME`环境变量指定了这一点)，并且是您的 bash 解释器将在第一次加载时执行的文件。它将执行里面的所有命令。这意味着，您可以通过简单地在您的`.bashrc`中放入:

```py
...other stuff above...
# Make /path/to/folder have higher priority
export PATH=/path/to/folder:$PATH

# Make /path/to/other/folder have lower priority
export PATH=$PATH:/path/to/folder
...other stuff below... 
```

## 数据科学和`PATH`环境变量

现在，**这与数据科学家有什么关系？好吧，如果你是一名数据科学家，你很可能使用 Python，并且你的 Python 解释器来自 Anaconda Python 发行版(这是一个非常棒的东西，去得到它吧！).Anaconda Python 安装程序所做的是在`PATH`环境变量中优先考虑`/path/to/anaconda/bin`文件夹。您的系统上可能安装了其他 Python 解释器(即 Apple 自带的解释器)。然而，这个`PATH`修改确保了每次在 Bash 终端中键入`python`时，都会执行 Anaconda Python 发行版附带的 Python 解释器。在我的例子中，安装了 Anaconda Python 发行版后，我的`PATH`看起来像这样:**

```py
$ echo $PATH
/Users/ericmjl/anaconda/bin:/usr/bin:/bin:/usr/local/bin 
```

更好的是，conda 环境所做的是在环境被激活时预先考虑到 conda 环境二进制文件夹的路径。例如，对于我的博客，我将它保存在一个名为`lektor`的环境中。因此...

```py
$ echo $PATH
/Users/ericmjl/anaconda/bin:/usr/bin:/bin:/usr/local/bin
$ which python
/Users/ericmjl/anaconda/bin/python
$ source activate lektor
$ echo $PATH
/Users/ericmjl/anaconda/envs/lektor/bin:/Users/ericmjl/anaconda/bin:/usr/bin:/bin:/usr/local/bin
$ which python
/Users/ericmjl/anaconda/envs/lektor/bin/python 
```

注意 bash 终端现在如何优先选择优先级更高的`lektor`环境中的 Python。

如果你已经到了这一步，那么你将有希望意识到这里列出了一些重要的概念。让我们回顾一下:

*   `PATH`是一个存储为纯文本字符串的环境变量，bash 解释器使用它来确定在哪里可以找到可执行程序。
*   `PATH`是冒号分隔的；优先级较高的目录位于字符串的左侧，而优先级较低的目录位于字符串的右侧。
*   `PATH`可以通过在环境变量前添加或附加目录来修改。它可以通过在命令提示符下运行`export`命令在 bash 会话中暂时完成，也可以通过在`.bashrc`或`.bash_profile`中添加一个`export`行在 bash 会话中永久完成。

## 其他感兴趣的环境变量

现在，数据科学家可能会遇到哪些其他环境变量？这些是您可能会看到并且可能必须修复的示例，尤其是在您的系统管理员外出度假(或者需要很长时间才能响应)的情况下。

**对于一般用途的**，你肯定想知道你的`HOME`文件夹在哪里——在 Linux 系统上，通常是`/home/username`，而在 macOS 系统上，通常是`/Users/username`。您可以通过以下操作来弄清楚`HOME`是什么:

```py
$ echo $HOME
/Users/ericmjl 
```

**如果你是 Python 用户**，那么`PYTHONPATH`是一个可能有用的变量。它由 Python 解释器使用，并指定在哪里可以找到 Python 模块/包。

**如果你必须处理 C++库**，那么了解你的`LD_LIBRARY_PATH`环境变量将会非常重要。我在这方面还不够精通，不能明智地支持它，所以我会遵从[这个网站](https://web.archive.org/web/20220703062943/http://xahlee.info/UnixResource_dir/_/ldpath.html)关于使用`LD_LIBRARY_PATH`变量的最佳实践的更多信息。

**如果你正在使用 Spark** ，那么`PYSPARK_PYTHON`环境变量将会是你感兴趣的。这实际上是告诉 Spark 为其驱动程序和工作程序使用哪种 Python 如果需要，您还可以将`PYSPARK_DRIVER_PYTHON`设置为独立于`PYSPARK_PYTHON`环境变量。

### 破解您的环境变量

这是最有趣的地方！通过修改环境变量，你可以做一些事情。

**黑客#1:访问 PyPy。**我偶尔会关注 PyPy 的开发，但是因为 PyPy 还不是默认的 Python 解释器，并且还不具备`conda install`能力，所以我不得不把它放在自己的`$HOME/pypy/bin`目录中。为了能够访问 PyPy 解释器，我必须确保我的`/path/to/pypy`出现在`PATH`环境变量中，但是优先级比我的常规 CPython 解释器低。

黑客#2:允许访问其他语言解释器/编译器。这类似于 PyPy。我曾经尝试使用 Lua 的 JIT 解释器来使用 Torch 进行深度学习，并需要在我的`.bashrc`中添加一个路径。

**黑客#3:将 Python 包安装到您的主目录。**在使用`modules`系统而不是`conda`环境的共享 Linux 计算系统上，您加载的`modulefile`可能配置有一个虚拟环境*，而您没有权限修改该虚拟环境*。如果需要安装 Python 包，可能要`pip install --user my_pkg_name`。这将把它安装到`$HOME/.local/lib/python-[version]/site-packages/`。在这种情况下，确保您的`PYTHONPATH`包含足够高优先级的`$HOME/.local/lib/python-[version]/site-packages`将非常重要。

**黑客 4:出错时调试。**如果出现了错误，或者您有了意外的行为——我以前遇到过的情况是在加载了我所有的 Linux 模块之后，我的 Python 解释器没有被正确地找到——那么调试的一种方法是临时将您的 PATH 环境变量设置为一些合理的“默认值”,并找到这些值，有效地“重置”您的 PATH 变量，以便您可以在调试时手动预先计划/附加。

为此，将下面一行代码放在主目录中名为`.path_default`的文件中:

```py
export PATH=""  # resets PATH to an empty string.
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH  # this is a sensible default; customize as needed. 
```

出现问题后，可以使用“source”命令重置 PATH 环境变量:

```py
$ echo $PATH
/some/complicated/path:/more/complicated/paths:/really/complicated/paths
$ source ~/.path_default
$ echo $PATH
/usr/bin:/bin:/usr/local/bin 
```

注意——您也可以在 bash 会话中执行完全相同的命令；交互性可能也是有帮助的。

### 结论

我希望您喜欢这篇文章，并且每当您遇到这些环境变量时，它将为您提供一条前进的道路！