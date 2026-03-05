# Python 和 Jupyter 笔记本简介

> 原文：<https://winder.ai/introduction-to-python-and-jupyter-notebooks/>

# Python 和 Jupyter 笔记本简介

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

本研讨会是对使用 Python 和 Jupyter 笔记本的快速介绍。

## 计算机编程语言

对于大多数数据科学任务，有两种竞争的开源语言。有数学背景的人更喜欢 R。 [Python](https://www.python.org/) 有编程背景者首选；我所有的工作坊目前都是 Python 语言的。

我更喜欢 Python，因为我可以用很少的代码完成很多事情。我们可以利用广泛的库生态系统和 Python 的脚本能力来提高效率。

这也可能是不利的一面。有了经验，你会倾向于创建大的一行程序；在一行中做很多事情的代码。这可能会让新用户难以理解。在可能的情况下，我尽量说得清楚些。

## jupyter 笔记本

(又名。IPython 笔记本电脑)

笔记本是一种不可思议的工作方式。它们允许你在同一个文件中展示文档和工作代码。人们可以拿起它，通读内容并立即看到运行的代码。

*单元格*有两种类型:降价和编码。这是一个降价单元。代码单元运行实际的 python 代码！

## 恋恋笔记本

当使用单元格时，尽量将不同的代码片段分开。

在每个单元的末端，有一些输出的空间。输出可以是空白，图片，html。几乎所有你能想到的。

```py
print(1) # This is like a printf. It will print underneath the current cell
print("This is a %s  %d  %0.2f" % ("word", 1, 0.2343)) # This is how you print variables
"abc"
"123"    # By default, the last non-supressed element that isn't asigned to a variable is also printed 
```

```py
1
This is a word 1 0.23

'123' 
```

您可以通过按 ctrl-enter 来运行单元格。如果您按下 shift-enter，这将运行单元格并前进。

您可以通过查看“帮助->键盘快捷键”找到更多方便的键盘快捷键

# 变量和数组类型

默认情况下，Python 是非类型化的。所以变量超级容易定义。

除了变量，我们一整天都要用到的两个构造是列表(也称为数组)和字典(也称为映射)。为什么他们不叫它们数组和地图，我不知道。

```py
my_variable = 2                         # A simple variable
my_list = [1, 2, 3]                     # A simple list
print(my_list[0])                       # Zero indexing, print is an inbuilt printf like function.
another_list = ["a", "string", "list"]
print(another_list[:])                  # Colon means "all"
print(another_list[0:2])                # Index ranges are exclusive.
character_list = 'abc'
print(character_list[-1])               # -1 means the last entry, -2 means last but one 
```

```py
1
['a', 'string', 'list']
['a', 'string']
c 
```

```py
first_dict = {'bob': 32, 'steve': 94}               # Simple dictionary
key = "bob"
print("%s is aged %d" % (key, first_dict[key]))     # print accepts parameters after a % sign. Note the brackets around the terms. 
```

```py
bob is aged 32 
```

###示例

下面的代码将"

*   创建一个键的 Python 列表
*   创建一个包含指向某些值的键的映射
*   使用 Python 函数`print(...)`打印列表中的一个值或一系列值

```py
keys = ["a", "b", "c"]
d = {"a": 1, "b": 2, "c": 3}
for k in keys:
    print(d[k]) 
```

```py
1
2
3 
```

# 功能

函数的定义方式与其他语言类似，但是您可能不习惯这种语法。

因为类型是在运行时解释的，所以它不是很严格(你可以告诉它强制类型)。

```py
def printData(x=[1, 2, 3]):     # An = in the paramter list means "default to". Note the colon
    for x_i in x:               # Note the tab indentation in the function. This is required.
        print(x_i)              # The "in" construct iterates over values in x.

printData()
printData([11, 12])
printData(["a", "string"]) 
```

```py
1
2
3
11
12
a
string 
```

# 便捷的功能

python 有许多方便的扩展。你可能不需要使用这些。但是这里有一些&mldr;

```py
str_list = [str(x) for x in range(3)]               # "List comprehension", i.e. create a list from a for loop
print(str_list)
print(', '.join(str(x) for x in range(3, 0, -1)))   # Joining strings

# The following will only work in python 3, the first (of many) difference between 2 and 3.
l = lambda x: print(x**2)                           # A "lambda", a function. ** = power. Most times you can just define a function.
l(3) 
```

```py
['0', '1', '2']
3, 2, 1
9 
```