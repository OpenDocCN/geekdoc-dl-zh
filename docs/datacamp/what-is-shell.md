# 壳牌是什么？

> 原文：<https://web.archive.org/web/20230101102949/https://www.datacamp.com/blog/what-is-shell>

![Computer Script Graphic](img/e7332c98094f4604bd3b9091ecc3d16a.png)

*   [什么是贝壳？](#what-is-a-shell?)
*   [为什么壳牌这么受欢迎？](#why-is-shell-so-popular?)
*   Shell 脚本有什么缺点？
*   壳牌是什么时候创立的？
*   [壳牌如何得名](#how-shell-got-its-name)
*   [外壳特征](#shell-features)
*   [壳牌多年来的发展历程](#how-shell-has-evolved-over-the-years)
*   [谁用壳牌？](#who-uses-shell?)
*   [外壳示例](#shell-examples)
*   [壳牌的职业生涯](#careers-with-shell)
*   [结论](#conclusion)

如果您正在使用微软、苹果或 Linux 操作系统(OS ),您可能在不知不觉中使用了 shell 脚本。事实上，每次启动 Linux 操作系统时，您都会与 shells 进行交互。

Shell 脚本帮助程序员、数据科学家和临时用户节省时间，并通过自动化避免重复任务。例如，脚本可以执行每日备份、安装补丁、监控系统和活动，以及执行例行审计。

Shells 读取相当直观的人类命令，并将它们转换成系统可以理解的东西。

## 什么是贝壳？

shell 是一种叫做命令行解释程序的计算机程序，它让 Linux 和 Unix 用户通过命令行界面控制他们的操作系统。Shells 允许用户与他们的操作系统直接有效地通信。

Shell 脚本不是一种单一的语言，但是因为它使用一些自然语言命令，所以即使没有编程背景也很容易学习。然而，每种 shell 脚本语言都被认为是一种语言，如果您计划更复杂的活动，shell 需要大量的实践。

Shell 脚本是设置自动化的最简单的方法之一。使用 Linux 或 Unix 命令，shell 脚本为数据科学家、开发人员和技术人员提供了重复命令的条件和循环控制结构。

Shell 脚本并不总是使用相同的名称。`Sh,` Bash(最常见)，`csh,`和 tesh 都是 shell 脚本。而在 IBM 的 VM 操作系统中，它们被称为 EXEC 在 DOS 中，shell 脚本被称为批处理文件。

shell 有两个类别，命令行 shell 和图形 shell。命令行 shells 是使用命令行界面访问的，其中系统以人类可读的命令接收输入，然后使用命令行界面显示输出。

图形外壳使用图形用户界面(GUI)来执行交互和基本操作，如打开、关闭和保存文件。

以下是一些[示例 shell 命令](https://web.archive.org/web/20220525032049/https://www.datacamp.com/community/tutorials/shell-commands-data-scientist):

要找出您所在的目录:`pwd`

在手册中查找命令:`man command `

要使文本文件可滚动:`less file 1` 或`more file 1 `

### 不同类型的贝壳

#### 伯恩·谢尔

顾名思义,《谍影重重》与间谍和高速汽车追逐毫无关系。Bourne shell 是第二个最常用的 Unix shell，由 Stephen Bourne 在 1979 年在贝尔实验室创建。和它的前身 Thompson shell 一样，Bourne 的可执行文件名是`sh`。

Bourne shell 是 Solaris 操作系统的缺省 shell。尽管年代久远，Bourne shell 仍因其速度快、结构紧凑而广受欢迎。然而，Bourne shell 的交互性不是很好，并且不能调用它的`command history.`,它也缺乏逻辑和算术表达式。

伯恩提示包括:

*   命令全路径:`/bin/sh`和`/sbin/sh`
*   非根用户默认:`$`
*   根用户默认:`#`

#### 命令行

C shell，文件名`csh`，和它的前身文件名`tcsh`，是另一个 70 年代后期的 Unix shell。它是由比尔·乔伊创造的，他当时是加州大学伯克利分校的一名研究生。

与 Bourne shell 不同，C shell 是交互式的，可以识别`command history`和`aliases`。C shells 还包括类似 C 的表达式语法和内置算法。

命令行提示包括:

*   命令全路径:`/bin/csh`
*   非根用户默认主机名:`%`
*   根用户默认主机名:`#`

#### 科恩壳牌公司

KornShell，文件名`ksh`，是由贝尔实验室的大卫·科恩在 20 世纪 80 年代早期开发的。KornShell 包含了 C shell 的许多特性，它是 Bourne shell 的超集，因此是向后兼容的。

科恩谢尔跑得比 C 壳快。它运行 Bourne shell 脚本，并具有类似 C 的数组、函数和字符串操作工具。此外，它还具有内置运算功能。

KornShell 提示包括:

*   命令全路径:`/bin/ksh`
*   非超级用户默认值:$
*   根用户默认值:#

#### GNU Bourne-又一个 shell

GNU Bourne-Again 或 Bash shell 是 Bourne shell 的一个开源替代产品。它是由 Brian Fox 为 GNU 项目设计的，于 1989 年发布。它不仅与 Bourne shell 完全兼容，还吸收了 KornShells 和 C shells 的许多最佳特性。GNU Bourne-Again shell 已经自动映射了用于编辑和命令调用的箭头键。

GNU Bourne-Again 提示包括:

*   命令全路径: `/bin/bash`
*   非 root 用户默认提示:`bash-x.xx$ `
*   根用户默认:`bash-x.xx#`

### shell 脚本做什么

这个类比有点简单，但是可以把 shell 脚本想象成程序员的自动填充。shell 使用单个脚本中的命令序列，而不是一次一个地在键盘上键入命令，用户可以在命令行中用简单的文件名启动这些脚本。Shell 脚本执行程序执行、文本换行和文件操作。

外壳脚本可以执行许多任务，包括监视磁盘使用、执行备份、创建命令工具、操作文件、运行程序、将程序链接在一起、完成批处理等任务。

### 壳牌的主要替代品

在 IBM 的 VM 操作系统中，它们被称为 EXEC 在 DOS 中，shell 脚本被称为批处理文件。不过，这些仍然是 shell 脚本。其他脚本语言，如 Javascript、Python、Perl 和 TCL 也是脚本语言。毫不奇怪，计算机专业人员都有他们的最爱，我们稍后将讨论 shells 的优缺点。

## 壳牌为什么这么受欢迎？

Shell 在几乎每个操作系统上都很常见，因为它们高效且易于更新。它监控你的电脑系统，并执行例行备份，你不必考虑它。

没有必要切换语法，因为 shell 的语法和命令与命令行中输入的是相同的。此外，编写 shell 脚本既简单又快捷:它们启动迅速，而且易于调试。

## Shell 脚本的缺点是什么？

Shell 脚本可能容易出错，并且在出现问题时难以诊断。Shells 不是为复杂或大型任务设计的，它们执行起来很慢。此外，shells 不提供太多的数据结构，并且可能存在语法或设计缺陷。

## 壳牌是什么时候创立的？

第一个 shell 脚本是由麻省理工学院的计算机职员 Louis Pouzin 在 20 世纪 60 年代早期创建的。他的第一个命令行是`RUNCOM`，它将计算机科学家从重复的任务中解放出来，如编译、更新、重命名和移动。

虽然 Pouzin 不是计算机语言专家，或者可能因为他不是，他相信命令行本身就是一种语言。

到 20 世纪 60 年代中期，Pouzin 与英国科学家 Christopher Strachey 合作。Strachey 设计了一个宏生成器，作为 Pouzin 命令语言的基础，运行在当时最先进的 Multics 操作系统上。

## 壳牌是如何得名的

Pouzin 将他的新语言命名为“shell ”,因为它是操作系统的最外层。

## 外壳特征

Shells 因为简洁而受到程序员的欢迎。然而，这并不意味着它们是基本的。外壳提供了几个特性，包括:

### 后台处理

shell 脚本的最大好处之一是它们可以在后台运行。根据命令的不同，shells 可以在前台或后台运行。前台进程在屏幕上可见，并且只能按顺序运行。

后台进程不会出现在屏幕上，可以不按顺序运行。要在后台运行 shell 脚本，用户只需在脚本末尾添加一个&符号。

### 通配符替换

通配符替换允许系统一次处理多个命令，或者从文本文件中查找短语片段。例如，`*` 告诉系统匹配任何字符串，甚至是空字符串。`?`匹配单个字符，`[example]`匹配任何字符(“example”只是一个例子)，而`[1-9]`(另一个例子)指示 shell 匹配范围内的字符。

### 命令别名

外壳别名是快捷命令。有些别名是一个单词，有些是一个字母。要查看别名列表，用户只需输入`·alias`。

### 命令历史

shells 有很多方法可以节省时间和精力，但是一个特别方便的特性是它的命令历史。`history`命令显示该会话期间使用的所有命令，而不是重新键入命令。

### 文件名替换

文件名替换也称为“全局替换”当一个单词包含诸如`?`、`*`或`[`之类的字符，或者以`~`开头时，shell 会将该单词视为一种模式，并替换为与该模式匹配的文件名的字母列表。

### 输入/输出重定向

输入/输出(i/o)重定向允许用户交换与显示屏、键盘或文件相关联的标准输入(stdin)和标准输出(stdout)。

### 平静的

壳管道是另一种重定向，它将命令/进程/程序的输出发送到另一个命令/进程/程序。这使得命令/过程/程序同时运行，并允许连续的数据传输，而不必通过显示屏或临时文本文件。

### 外壳变量替换

当 shell 遇到包含特殊字符的表达式时，它会将代码翻译成用户更容易识别的内容。这个过程被称为变量替换或简称为变量。

更有经验的程序员也使用变量。例如，如果程序员在执行程序之前不知道实际值，他们可以在代码准备好执行之前使用变量作为占位符。

## 贝壳是如何变化的

有两种主要的 shell，“shell”(sh)和“bash”。两者都在 Unix 操作系统中运行。Shell 是任何 shell 中脚本的通称。顾名思义，bash (Bourne 又叫 shell)是一个改进和扩展的 Shell。

Bash 使用升级来实现更多功能，支持作业控制，比 sh 更容易使用，并且支持命令历史。Sh 不支持命令历史记录；它的脚本可以在任何 shell 中运行，并且更具可移植性。

## 壳牌多年来的发展历程

这些年来，贝壳一直在进化，但基本保持不变。Bourne shell 比最初的 Thompson shell 向前迈进了一大步，但是许多最初的功能仍然存在。当然，计算机和我们的期望都变了。

有多少用户就有多少外壳；但从根本上说，有两种类型的 shells 命令行或 Bourne。其他的都是方言而不是不同的语言。

### Unix shells

Unix shells 是最初的版本，现在仍然很强大。Unix shells 工作在 Unix 和 Unix 相关的操作系统上，比如 MAC。

### 命令行

命令行(csh)是最常见的 Unix shells。命令行擅长交互式工作，包括别名、cdpath、作业控制、路径散列、目录堆栈等。它们还提供编辑和历史功能。

### Tenex 命令行

Tenex 命令行(tcsh)是由 Caregia Melo 大学的 Ken Greer 开发的。Tenex 被认为是对原始命令行版本的改进。与最初的 C 不同，Tenex 提供了命令行编辑和完成等功能。1981 年，tsch 与 csh 合并。

### 科恩炮弹公司

Korn shell(ksh)是另一种 Unix shell，但它是 C 和 Bourne shells 之间的一种折衷，向后兼容后者。Korn shell 是由贝尔实验室的 David Korn 在 20 世纪 80 年代早期开发的。

### 伯恩再次炮轰

Bourne Again shell (Bash)是一个开源的 Unix shell，由 Brian Fox 在 20 世纪 90 年代后期为 GNU 项目编写，作为 Bourne shell 的升级。

### 奇异的贝壳

虽然它们都是建立在 C 或 Bourne 外壳上，但程序员们已经设计了 100 种甚至 1000 种不同的方言。有些有用，有些纯粹是好玩。

## 谁用壳牌？

几乎所有使用计算机的人都从 shell 中受益，但它们尤其有益于系统管理员、开发人员、程序员和数据科学家。Shells 用于自动化任务和应用程序、安装包、备份或恢复数据，当然也用于编程。

## 外壳示例

外壳可以是基本的，也可以是复杂的，这取决于用户的意愿。例如，一个简单的 bash 问候语是`echo “hello $USER”`，而[复杂的 shell 脚本](https://web.archive.org/web/20220525032049/https://www.linuxtopia.org/online_books/advanced_bash_scripting_guide/moreadv.html)几乎是无限的。以下是一些基本的例子:

**Example 1: Using a while loop in bash**

Create a file in bash using a suitable editor. Here we use vi editor.`$ vi example.sh`This opens an editor with a file named example.sh
Press the 'i' key to start inserting the code:

```py
#!/bin/bash
valid=true
count=1
while [ $valid ]
do
echo $count
if [ $count -eq 10 ];
then
break
fi
((count++))
done
```

Press escape follwed by ':wq' to save and return to the terminal. 
Execute by using `bash example.sh `**Example 2:** **Accepting inputs from the terminal by users**Create another file in vi editor: 
`$vi example2.sh`Enter code in the editor:

```py
#!/bin/bash
echo "Enter first number"
read x
echo "Enter second number"
read y
((sum=x+y))
echo "The result of addition = $sum"
```

Execute by using `bash example2.sh`To learn more useful shell commands for data science, check out this tutorial on [Useful Shell Commands](https://web.archive.org/web/20220525032049/https://www.datacamp.com/tutorial/shell-commands-data-scientist).

## 壳牌的职业生涯

与其说是职业道路，不如把 shell 脚本看作是数据科学家武器库中的一个有用工具。各种类型的数据专业人员都需要知道多种语言(或者至少他们应该知道)，shell 脚本可以帮助他们更有效地使用这些语言。

尽管如此，一些公司有一些角色特别需要 shell 脚本。平均工资[在 7.8 万美元](https://web.archive.org/web/20220525032049/https://www.payscale.com/research/US/Skill=Shell_Scripting/Salary)左右。也就是说，shell 脚本是像系统管理员这样的角色的组成部分，可以带来六位数的薪水。

## 结论

Shell 就像一个字符串，贯穿于每一种编程语言。复杂的命令可以简化为简洁直观的命令。对于任何数据科学家或计算机工程师来说，理解 shell 都是一项重要技能。在 DataCamp 的[shell 简介](https://web.archive.org/web/20220525032049/https://www.datacamp.com/courses/introduction-to-shell)课程中了解 Shell。

内核是操作系统最重要的部分。内核负责为计算机程序分配内存和时间，并管理通信和文件存储以响应系统调用。

外壳是内核和用户之间的接口。

我们已经知道 shell 是高效的，它会处理用户的许多无聊任务；但是除此之外，shells 允许用户定制脚本来满足自己的需求。

一个错误可能会造成巨大的损失，而且很难识别。另外，shells 在操作系统之间的移植性不是很好。

Linux 是由 Linus Torvalds 开发的 Unix 克隆。Linux 本身不是一个操作系统；它是一个内核。然而，Unix 包括内核、外壳和程序。

Bash 或 Bourne Again shell 是最常用的 shell，但是 Linux 和 Unix 都支持命令行、KornShells 和 TCSH。

在自然语言中，句法指的是单词和短语是如何排列的。Shell 语法非常相似；它指的是操作的顺序。

解释器将单个语句解释成机器代码。

编译器把高级程序翻译成更复杂的机器语言。

使用`chmod+x script-name-here.sh`设置权限

然后，用`./script-name-here.sh`或`sh script-name-here.sh.`运行它