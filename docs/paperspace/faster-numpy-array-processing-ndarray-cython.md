# 采用 Cython 的 NumPy 阵列处理:速度提高 1250 倍

> 原文：<https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/>

本教程将向您展示如何使用 Cython 加速 NumPy 数组的处理。通过在 Python 中显式指定变量的数据类型，Cython 可以在运行时大幅提高速度。

本教程涵盖的部分如下:

*   遍历 NumPy 数组
*   NumPy 数组的 Cython 类型
*   NumPy 数组元素的数据类型
*   NumPy 数组作为函数参数
*   对 NumPy 数组进行索引，而不是迭代
*   禁用边界检查和负索引
*   摘要

关于 Cython 及其使用方法的介绍，请查看我在[上发表的使用 Cython 提升 Python 脚本](https://blog.paperspace.com/boosting-python-scripts-cython/)的文章。要不，我们开始吧！

# 遍历 NumPy 数组

我们将从与上一教程相同的代码开始，除了这里我们将遍历一个 NumPy 数组而不是一个列表。NumPy 数组是使用 arrange()函数在 *arr* 变量中创建的，该函数从 0 开始以 1 为步长返回 10 亿个数字。

```py
import time
import numpy

total = 0
arr = numpy.arange(1000000000)

t1 = time.time()

for k in arr:
    total = total + k
print("Total = ", total)

t2 = time.time()
t = t2 - t1
print("%.20f" % t)
```

我在一台配有 Core i7-6500U CPU @ 2.5 GHz 和 16 GB DDR3 RAM 的机器上运行这个程序。Python 代码在 458 秒(7.63 分钟)内完成。太长了。

让我们看看编辑在[之前的教程](https://blog.paperspace.com/boosting-python-scripts-cython/)中创建的 Cython 脚本后需要多长时间才能完成，如下所示。唯一的变化是在循环的*中包含了 NumPy 数组。请注意，在使用 Cython 脚本之前，您必须使用下面的命令来重新构建它。*

```py
python setup.py build_ext --inplace
```

当前形式的 Cython 脚本在 128 秒(2.13 分钟)内完成。仍然很长，但这是一个开始。让我们看看如何让它更快。

# NumPy 数组的 Cython 类型

之前我们看到，在为所使用的变量显式定义 C 类型后，Cython 代码运行得非常快。NumPy 数组也是这种情况。如果我们保留 NumPy 数组的当前形式，Cython 的工作方式与常规 Python 完全一样，为数组中的每个数字创建一个对象。为了让事情运行得更快，我们还需要为 NumPy 数组定义一个 C 数据类型，就像为任何其他变量一样。

NumPy 数组的数据类型为 ***ndarray*** ，代表 **n 维数组**。如果您使用关键字 *int* 来创建整数类型的变量，那么您可以使用 *ndarray* 来创建 NumPy 数组的变量。注意，必须使用 NumPy 调用*n array*，因为*n array*在 NumPy 内部。因此，创建 NumPy 数组变量的语法是 *numpy.ndarray* 。下面列出的代码创建了一个名为 *arr* 的变量，数据类型为 NumPy *ndarray* 。

首先要注意的是，NumPy 是使用第二行中的常规关键字 *import* 导入的。在第三行，您可能会注意到 NumPy 也是使用关键字 *cimport* 导入的。

现在我们来看看 Cython 文件可以分为两类:

1.  定义文件(。pxd)
2.  实现文件(。pyx)

定义文件的扩展名为。pxd，用于保存 C 声明，例如要导入并在其他 Cython 文件中使用的数据类型。另一个文件是带有扩展名的实现文件。pyx，我们目前用它来编写 Cython 代码。在这个文件中，我们可以导入一个定义文件来使用其中声明的内容。

下面的代码将被写入扩展名为. pyx 的实现文件中。 *cimport numpy* 语句在 Cython 中导入一个名为“numpy”的定义文件。这样做是因为 cy thon“numpy”文件具有处理 NumPy 数组的数据类型。

下面的代码定义了前面讨论的变量，分别是 *maxval* 、 *total* 、 *k* 、 *t1* 、 *t2* 和 *t* 。有一个名为 *arr* 的新变量保存数组，数据类型`numpy.ndarray`。之前使用了两个导入语句，即`import numpy`和`cimport numpy`。这里哪一个是相关的？这里我们将使用需要的`cimport numpy`，而不是常规的`import`。这让我们可以访问在 Cython numpy 定义文件中声明的 *numpy.ndarray* 类型，因此我们可以将 *arr* 变量的类型定义为 numpy.ndarray

*maxval* 变量被设置为等于 NumPy 数组的长度。我们可以从创建一个长度为 10，000 的数组开始，并在以后增加这个数字，以比较 Cython 相对于 Python 的改进。

```py
import time
import numpy
cimport numpy

cdef unsigned long long int maxval
cdef unsigned long long int total
cdef int k
cdef double t1, t2, t
cdef numpy.ndarray arr

maxval = 10000
arr = numpy.arange(maxval)

t1 = time.time()

for k in arr:
    total = total + k
print "Total =", total

t2 = time.time()
t = t2 - t1
print("%.20f" % t)
```

在创建了一个类型为`numpy.ndarray`的变量并定义了它的长度之后，接下来是使用`numpy.arange()`函数创建数组。注意，这里我们使用的是 Python NumPy，它是使用`import numpy`语句导入的。

通过运行上面的代码，Cython 只用了 0.001 秒就完成了。对于 Python，代码耗时 0.003 秒。在这种情况下，Cython 比 Python 快近 3 倍。

当`maxsize`变量设置为 100 万时，Cython 代码运行 0.096 秒，而 Python 需要 0.293 秒(Cython 也快了 3 倍)。当处理 1 亿个时，Cython 需要 10.220 秒，而 Python 需要 37.173 秒。对于 10 亿，Cython 需要 120 秒，而 Python 需要 458 秒。尽管如此，Cython 可以做得更好。让我们看看怎么做。

# NumPy 数组元素的数据类型

第一个改进与数组的数据类型有关。NumPy 数组`arr`的数据类型根据下一行定义。注意，我们所做的只是定义数组的类型，但是我们可以给 Cython 更多的信息来简化事情。

请注意，没有任何东西可以警告您有一部分代码需要优化。一切都会起作用；你必须调查你的代码，找出可以优化的部分，以便运行得更快。

```py
cdef numpy.ndarray arr
```

除了定义数组的数据类型，我们还可以定义两条信息:

1.  数组元素的数据类型
2.  维度数量

数组元素的数据类型是`int`，根据下面的代码行定义。使用 **cimport** 导入的 numpy 有一个对应于 NumPy 中每种类型的类型，但在末尾有 **_t** 。比如常规 NumPy 中的 **int** 对应 Cython 中的 **int_t** 。

参数是`ndim`，它指定了数组中的维数。这里设置为 1。请注意，它的默认值也是 1，因此可以从我们的示例中省略。如果使用了更多的维度，我们必须指定它。

```py
cdef numpy.ndarray[numpy.int_t, ndim=1] arr
```

不幸的是，只有当 NumPy 数组是函数中的一个参数或者函数中的一个局部变量时——而不是在脚本体中——才允许这样定义 NumPy 数组的类型。我希望 Cython 尽快解决这个问题。我们现在需要编辑前面的代码，将其添加到下一节将要创建的函数中。现在，让我们在定义数组之后创建它。

注意，我们将变量`arr`的类型定义为`numpy.ndarray`，但是不要忘记这是容器的类型。该容器包含元素，如果没有指定其他内容，这些元素将被转换为对象。为了强制这些元素为整数，根据下一行将`dtype`参数设置为`numpy.int` 。

```py
arr = numpy.arange(maxval, dtype=numpy.int)
```

这里使用的 numpy 是使用`cimport` 关键字导入的。一般来说，无论何时发现用于定义变量的关键字 numpy，都要确保它是使用`cimport`关键字从 Cython 导入的。

# NumPy 数组作为函数参数

准备好数组后，下一步是创建一个函数，该函数接受类型为`numpy.ndarray`的变量，如下所示。这个函数被命名为`do_calc()`。

```py
import time
import numpy
cimport numpy

ctypedef numpy.int_t DTYPE_t
def do_calc(numpy.ndarray[DTYPE_t, ndim=1] arr):
    cdef int maxval
    cdef unsigned long long int total
    cdef int k
    cdef double t1, t2, t

    t1 = time.time()

    for k in arr:
        total = total + k
    print "Total = ", total

    t2 = time.time()
    t = t2 - t1
    print("%.20f" % t)
```

```py
import test_cython
import numpy
arr = numpy.arange(1000000000, dtype=numpy.int)
test_cython.do_calc(arr)
```

构建完 Cython 脚本后，接下来我们根据下面的代码调用函数`do_calc()`。这种情况下的计算时间从 120 秒减少到 98 秒。这使得 Cython 在对 10 亿个数字求和时比 Python 快 5 倍。正如你现在所期望的，对我来说这仍然不够快。我们将在下一节看到另一个加速计算的技巧。

# NumPy 数组上的索引与迭代

Cython 只是将计算时间减少了 5 倍，这并不鼓励我使用 Cython。但这不是 Cython 的问题，而是使用的问题。问题在于循环是如何产生的。让我们仔细看看下面给出的循环。

在之前的教程中，提到了非常重要的一点，那就是 Python 只是一个接口。界面只是让用户觉得事情更简单。请注意，简单的方法并不总是做某事的有效方法。

python[接口]有一种迭代数组的方法，这些数组在下面的循环中实现。循环变量 *k* 在 *arr* NumPy 数组中循环，从数组中一个接一个地取出元素，然后将该元素赋给变量 *k* 。以这种方式循环遍历数组是 Python 中引入的一种风格，但它不是 C 用于循环遍历数组的方式。

```py
for k in arr:
    total = total + k
```

对于编程语言来说，循环遍历数组的通常方式是从 0[有时从 1]开始创建索引，直到到达数组中的最后一个索引。每个索引用于索引数组以返回相应的元素。这是循环遍历数组的正常方式。因为 C 不知道如何以 Python 风格遍历数组，所以上面的循环是以 Python 风格执行的，因此执行起来要花很多时间。

为了克服这个问题，我们需要创建一个普通样式的循环，使用索引`for`访问数组元素。新的循环实现如下。

首先，有一个名为*arr _ shape***的新变量用于存储数组中元素的数量。在我们的示例中，只有一个维度，它的长度通过使用索引 0 索引*数组形状*的结果来返回。**

然后 *arr_shape* 变量被提供给`range()`函数，该函数返回访问数组元素的索引。在这种情况下，变量 *k* 代表一个索引，而不是一个数组值。

在循环内部，通过索引 *k* 索引变量 *arr* 来返回元素。

```py
cdef int arr_shape = arr.shape[0]
for k in range(arr_shape):
    total = total + arr[k]
```

让我们编辑 Cython 脚本来包含上面的循环。下面列出了新脚本。旧循环被注释掉了。

```py
import time
import numpy
cimport numpy

ctypedef numpy.int_t DTYPE_t

def do_calc(numpy.ndarray[DTYPE_t, ndim=1] arr):
    cdef int maxval
    cdef unsigned long long int total
    cdef int k
    cdef double t1, t2, t
    cdef int arr_shape = arr.shape[0]

    t1=time.time()

#    for k in arr:
#        total = total + k

    for k in range(arr_shape):
        total = total + arr[k]
    print "Total =", total

    t2=time.time()
    t = t2-t1
    print("%.20f" % t)
```

通过构建 Cython 脚本，在将循环改为使用索引后，对 10 亿个数字求和的计算时间现在大约只有一秒钟。所以，时间从 120 秒减少到仅仅 1 秒。这是我们对 Cython 的期望。

请注意，当我们使用 Python 风格遍历数组时，不会发生任何错误。没有迹象可以帮助我们找出代码没有优化的原因。因此，我们必须仔细寻找代码的每一部分，寻找优化的可能性。

注意，普通 Python 执行上述代码需要 500 多秒，而 Cython 只需要 1 秒左右。因此，对于 10 亿个数的求和，Cython 比 Python 快 500 倍。超级棒。记住，为了减少计算时间，我们牺牲了 Python 的简单性。在我看来，将时间减少 500 倍，值得使用 Cython 优化代码。

代码速度提高 500 倍固然很好，但还有一个改进，这将在下一节讨论。

# 禁用边界检查和负索引

如 [Cython 文档](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html)中所述，导致代码变慢的因素有很多，包括:

1.  边界检查，以确保索引在数组的范围内。
2.  使用负索引访问数组元素。

当 Cython 执行代码时，这两个特性是活动的。可以使用负索引(如-1)来访问数组中的最后一个元素。Cython 还确保没有索引超出范围，如果出现这种情况，代码不会崩溃。如果您不需要这些功能，您可以禁用它以节省更多时间。这是通过添加以下行来实现的。

```py
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
The new code after disabling such features is as follows.
import time
import numpy
cimport numpy
cimport cython

ctypedef numpy.int_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def do_calc(numpy.ndarray[DTYPE_t, ndim=1] arr):
    cdef int maxval
    cdef unsigned long long int total
    cdef int k
    cdef double t1, t2, t
    cdef int arr_shape = arr.shape[0]

    t1=time.time()

#    for k in arr:
#        total = total + k

    for k in range(arr_shape):
        total = total + arr[k]
    print "Total =", total

    t2=time.time()
    t = t2-t1
    print("%.20f" % t)
```

构建并运行 Cython 脚本后，时间不在 0.4 秒左右。与 Python 脚本的计算时间(大约 500 秒)相比，Cython 现在比 Python 快了大约 1250 倍。

# 摘要

本教程使用 Cython 来提高 NumPy 数组处理的性能。我们通过四种不同的方式实现了这一目标:

## 1.定义 NumPy 数组数据类型

我们从使用`numpy.ndarray`指定 NumPy 数组的数据类型开始。我们看到这个类型在使用`cimport`关键字导入的定义文件中是可用的。

## 2.指定数组元素的数据类型+维数

仅仅将`numpy.ndarray`类型赋给一个变量是一个开始——但这还不够。仍然需要提供两条信息:数组元素的数据类型和数组的维数。两者都对处理时间有很大影响。

只有当 NumPy 数组被定义为函数参数或函数内部的局部变量时，这些细节才会被接受。因此，我们在这些点上添加了 Cython 代码。您还可以指定函数的返回数据类型。

## 3.使用索引遍历 NumPy 数组

减少处理时间的第三种方法是避免 Pythonic 循环，在这种循环中，一个变量由数组中的值逐个赋值。相反，只需使用索引遍历数组。这导致时间的大量减少。

## 4.禁用不必要的功能

最后，您可以通过禁用 Cython 中为每个函数默认完成的一些检查来减少一些额外的毫秒数。这些包括“边界检查”和“回绕”禁用这些功能取决于您的具体需求。例如，如果使用负索引，则需要启用回绕功能。

# 结论

本教程讨论了使用 Cython 操作 NumPy 数组，其速度是 Python 单独处理的 1000 倍以上。减少计算时间的关键是指定变量的数据类型，并对数组进行索引而不是迭代。

在下一篇教程中，我们将通过使用 Cython 来减少遗传算法的 [Python 实现的计算时间，从而总结和推进我们迄今为止的知识。](https://github.com/ahmedfgad/GeneticAlgorithmPython)