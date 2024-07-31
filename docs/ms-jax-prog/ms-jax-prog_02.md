# `第二章：开始使用 Jax`

* * *

欢迎来到您的 Jax 冒险的发射台！在本章中，我们为进入 Jax 编程之旅做好准备。随着我们设置舞台，安装必要的工具，并且通过实际编码迈出第一步，让我们一起迎接 Jax 世界的探险。

## `2.1 设置 Jax 环境`

在我们开始 Jax 之旅之前，让我们确保我们的工具箱已经准备就绪。设置 Jax 环境是一个关键的第一步，在本节中，我们将指导您完成这个过程。从选择您的平台到安装必要的库，让我们确保您能够充分发挥 Jax 的威力。

设置舞台：选择您的平台

Jax 非常灵活，适用于多种平台，如 macOS、Linux 和 Windows。通过检查您的操作系统的兼容性和硬件要求，确保您选择了正确的路线。

第一步：安装 Python

Jax 依赖于 Python，如果您尚未安装，请立即安装。确保您安装了 Python 3.7 或更新版本。您可以从官方 Python 网站下载最新版本。

第二步：安装 Jax

安装了 Python 后，使用 pip 包安装程序获取 Jax。

打开您的终端或命令提示符，并输入：`pip install jax`

第三步：确认安装

通过打开 Python 解释器并输入以下内容来确认 Jax 是否正确安装：`import jax; print(jax.__version__)` 这将显示已安装的 Jax 版本。

准备您的工具箱：安装关键库

Jax 与其他库合作，以增强功能。让我们安装一些关键的库。

第一步：安装 NumPy

NumPy 是 Jax 的得力助手，用于数值操作。使用以下命令安装 NumPy：`pip install numpy`

第二步：可选库

考虑额外的库以扩展功能。例如，Matplotlib 用于绘图，scikit-learn 用于机器学习。根据需要安装它们。

当您的 Jax 环境设置好并且必要的库已经就位时，您已经准备好了。未来的旅程涉及编写代码、探索数据结构并利用 Jax 的潜力。

## `2.2 使用 NumPy 风格语法编写基本的 Jax 程序`

现在我们的 Jax 环境已经启动，是时候动手编写一些基本的 Jax 程序了。在本节中，我们将探索 NumPy 风格的语法，这不仅使 Jax 强大，而且使其非常熟悉。让我们开始编写代码，释放 Jax 的简洁之美。

第一步：导入 Jax

首先导入 Jax 库。这为您在 Jax 风格中进行所有数值计算铺平了道路。

`import jax`

第二步：创建数组

Jax 采用 NumPy 风格的语法来创建数组。让我们深入了解并创建一个简单的数组。

# `创建一个 Jax 数组`

`x = jax.numpy.array([1, 2, 3])`

第三步：执行操作

Jax 之美在于它能够以 NumPy 的简便方式对数组进行操作。

让我们将一个数学函数应用到我们的数组上。

# 对数组应用正弦函数

`y = jax.numpy.sin(x) + 2`

第 4 步：打印结果

现在，让我们打印结果，见证我们 Jax 计算的成果。

# 显示结果

`print(y)`

探索数据结构和数学运算

Jax 支持各种数据结构和数学运算，使其成为数值计算的多才多艺的工具。

Jax 中的数据结构

# 创建一个 Jax 向量

`vector = jax.numpy.array([4, 5, 6])`

# 创建一个 Jax 矩阵

`matrix = jax.numpy.array([[1, 2], [3, 4]])`

Jax 中的数学运算

# 执行算术运算

`result_addition = x + vector`

`result_multiplication = matrix * 2`

恭喜！你刚刚使用类似 NumPy 的语法编写了你的第一个 Jax 程序。NumPy 的简单性和熟悉性与 Jax 的强大功能合二为一。随着你继续你的 Jax 之旅，这些基础构件将为更复杂和令人兴奋的数值计算奠定基础。

## 在 Jax 中工作：数组，数据结构和数学运算

Jax 中的数组：拥抱数值简洁

Jax 的核心优势在于其对数组的出色处理能力。利用类似 NumPy 的语法，创建和操作数组变得轻而易举。

使用 Jax 创建数组

# 创建一个 Jax 数组

`x = jax.numpy.array([1, 2, 3])`

在数组上执行操作：

当涉及到数组操作时，Jax 展现出其强大的能力。无论是简单的算术运算还是复杂的数学函数，Jax 都能无缝处理。

# 应用操作于数组

`y = jax.numpy.sin(x) + 2`

Jax 中的数据结构：释放多才多艺

Jax 不仅局限于基本数组，还将其能力扩展到各种数据结构，为你的数值计算增添了灵活性。

在 Jax 中创建向量和矩阵

# 创建一个 Jax 向量

`vector = jax.numpy.array([4, 5, 6])`

# 创建一个 Jax 矩阵

`matrix = jax.numpy.array([[1, 2], [3, 4]])`

Jax 中的数学奇迹：超越基础运算

Jax 在处理各种操作时展现出其数学能力，使其成为多样化数值任务的强大工具。

Jax 中的算术运算

# 在数组上执行算术运算

`result_addition = x + vector`

`result_multiplication = matrix * 2`

Jax 中的复杂数学函数

# 应用更复杂的函数

`result_exp = jax.numpy.exp(x)`

`result_sqrt = jax.numpy.sqrt(matrix)`

正如你所见，Jax 将数值计算转化为简单和高效的游戏。通过 Jax 处理数组，探索数据结构，参与数学运算变得直观而强大。这标志着你掌握 Jax 编程艺术的又一步。

编码挑战：使用 Jax 进行数组操作

挑战：创建一个 Jax 程序，接受一个数组`A`并执行以下操作：

1\. 计算`A`中每个元素的平方。

2\. 计算平方元素的累积和。

3\. 求得结果数组的均值。

解决方案

`import jax`

`def array_manipulation_challenge(A):`

`# 第一步：计算 A 中每个元素的平方`

`squared_elements = jax.numpy.square(A)`

`# 第二步：计算平方元素的累积和`

`cumulative_sum = jax.numpy.cumsum(squared_elements)`

`# 第三步：找出结果数组的平均值`

`mean_result = jax.numpy.mean(cumulative_sum)`

`return mean_result`

# `使用示例:`

`input_array = jax.numpy.array([1, 2, 3, 4, 5])`

`result = array_manipulation_challenge(input_array)`

`print("结果:", result)`

这个挑战鼓励您使用 Jax 的能力来操作数组。随意尝试不同的输入数组，并探索 Jax 如何简化数值数据上的复杂操作。

这就是您的 Jax 之旅的开端！您已经奠定了基础，从设置环境到通过基本的 Jax 程序展示编码技巧。当我们结束时，请记住，这只是个开始。编码的乐园等待您去探索，而 Jax 的多功能性是您的工具箱。
