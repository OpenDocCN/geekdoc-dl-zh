<!--yml

category: 未分类

date: 2024-07-01 18:16:57

-->

# 一个帮助您编写张量形状检查的编译时调试器：ezyang's blog

> 来源：[`blog.ezyang.com/2018/04/a-compile-time-debugger-that-helps-you-write-tensor-shape-checks/`](http://blog.ezyang.com/2018/04/a-compile-time-debugger-that-helps-you-write-tensor-shape-checks/)

运行时调试器允许您查看程序中的具体值，对其进行更改，并继续运行程序。一个**编译时调试器**允许您查看程序中的符号值，对其进行推理，并编写程序的其余部分，例如填写缺失的张量大小检查。

这是编译时调试器实际操作的一个例子。

假设您正在编写一个简单的程序，从两个文件中读取一对张量并对它们进行矩阵乘法。“简单”，您想，然后编写以下程序：

```
main() {
  x = load("tensor1.t")
  y = load("tensor2.t")
  return matmul(x, y)
}

```

然而，有一个曲折：这个矩阵乘法是一个*未经检查*的矩阵乘法。如果传递的张量无法有效地相乘，这是未定义的行为。您的编译器已经意识到了这一点，并拒绝编译您的程序。您启动编译时调试器，它将您放到程序中出现错误的地方：

```
# Ahoy there Edward!  I stopped your program, because I could not
# prove that execution of this line was definitely safe:

   main() {
     x = load("tensor1.t")
     y = load("tensor2.t")
->   return matmul(x, y)
   }

# Here's what's in scope:

  _x_size : List(Nat)  # erases to x.size()
  _y_size : List(Nat)  # erases to y.size()
  x : Tensor(_x_size)
  y : Tensor(_y_size)

# I don't know anything else about these values

```

让我们仔细看一下作用域中的变量。我们的编译时调试器通过写`x : t`来告诉我们变量 x 的类型。我们有各种普通类型，如自然数（Nat）和自然数列表（List(Nat)）。更有趣的是，*张量*是由自然数列表参数化的，这些列表指定它们在每个维度上的大小。（为简单起见，假设张量的底层字段是固定的。）

我们的调试器有一个命令行，因此我们可以询问它关于程序中事物类型的问题（`:t`用于类型）：

```
> :t 1
# Here's the type of 1, which you requested:
Nat

> :t [1, 2, 0]
# Here's the type of [1, 2, 0], which you requested:
List(Nat)

> :t matmul
# Here's the type of matmul, which you requested:
forall (a, b, c : Nat). (Tensor([a, b]), Tensor([b, c])) -> Tensor([a, c])

```

矩阵乘法的类型应该是合理的。我们说矩阵乘法接受两个大小为 AxB 和 BxC 的二维张量，并生成大小为 AxC 的张量。如上所述，另一种表达方式是说，“对于任何自然数 A、B 和 C，矩阵乘法将接受大小为 AxB 和 BxC 的张量，并给出大小为 AxC 的张量”。

查看`load`的类型也是有教育意义的：

```
> :t load
# Here's the type of load, which you requested:
String ~> exists (size : List(Nat)). Tensor(size)

```

我们不知道从文件加载的张量的维度是多少；我们只能说*存在*一些大小（自然数列表），描述了所讨论的张量。我们的编译时调试器友好地为我们提供了作用域内张量大小的名称 `_x_size` 和 `_y_size`，并告诉我们如何在运行时计算它们（`x.size()` 和 `y.size()`）。

Enough of this. Let's remind ourselves why our program has failed to typecheck:

```
> matmul(x, y)

# I'm sorry!  I was trying to find values of a, b and c which
# would make the following equations true:
#
#     [a, b] = _x_size
#     [b, c] = _y_size
#
# But I don't know anything about _x_size or _y_size (they are skolem
# variables), so I couldn't do it.  Cowardly bailing out!

```

编译器是完全正确的。我们对 x 或 y 的大小一无所知；它们可能是 2D，也可能是 100D，或者根本不具有匹配的维度。

作为一种附加说明：有时候，不了解大小的任何信息也是可以的。考虑将张量加到自身的情况：

```
> add
# Here's the type of add, which you requested!
add : forall (size : List(Nat)). Tensor(size) -> Tensor(size) -> Tensor(size)

> add(x, x)
Tensor(_x_size)

# This type-checked OK!  I set size = _x_size and all of the arguments
# checked out.  You're good to go.

```

我们对`_x_size`一无所知，但`add`并不在乎；它会接受任何`List(Nat)`，而`_x_size`肯定是其中之一。

回到正题。我们将插入动态检查，以完善我们对 x 和 y 的知识，直到显然矩阵乘法将成功。

什么是动态检查？操作上，动态检查测试某个条件是否为真，并在条件不满足时中止。如果我们成功运行动态检查，我们现在对我们作用域中的符号类型有了新的*信息*。例如，添加一个*运行时*测试，检查两个数字是否相等后，我们随后可以在*编译时*假设这些数字是相等的：

```
> :t assert_eq_nat!
(x : Nat) -> (y : Nat) ~> x = y

```

首先，我们要确定我们的张量是 2D 张量：

```
> assert_eq_nat!(len(_x_size), 2)

# OK!  I added assert_eq_nat!(len(x.size()), 2) to your program, and
# here's what I know now:

  _x_size : List(Nat)
  _y_size : List(Nat)
  x : Tensor(_x_size)
  y : Tensor(_y_size)
  len(_x_size) = 2

# By the way, I can profitably destruct _x_size into its constituent
# parts; would you like to do this? (Y/n)

> Y

# OK, your new context is this:

  _x0, _x1 : Nat
  _y_size : List(Nat)
  x : Tensor([_x0, _x1])
  y : Tensor(_y_size)

# I don't know anything about the new variables _x0 and _x1, but I
# learned enough about _x_size that I was able to eliminate it from
# the context (_x_size = [_x0, _x1])

```

列表长度是一个非常有用的属性，用于对比测试，因为它极大地指定了所讨论列表的形状。我们也可以对`_y_size`做同样的事情：

```
> assert_eq_nat!(len(_y_size), 2)

# OK!  I added assert_eq_nat!(len(y.size()), 2) to your program, and
# here's what I know now:

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y_size : List(Nat)
  x : Tensor([_x0, _x1])
  y : Tensor(_y_size)
  len(_y_size) = 2

# By the way, I can profitably destruct _y_size into its constituent
# parts; would you like to do this? (Y/n)

> Y

# OK, your new context is this:

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y0, _y1 : Nat  # erases to y.size(0), y.size(1)
  x : Tensor([_x0, _x1])
  y : Tensor([_y0, _y1])

# I don't know anything about the new variables _y0 and _y1, but I
# learned enough about _y_size  that I was able to eliminate it from
# the context (_y_size = [_y0, _y1])

```

我们现在非常接近了。我们所需要做的就是断言内部维度是相等的：

```
> assert_eq_nat!(_x1, _y0)

# OK!  I added assert_eq_nat!(x.size(1), y.size(0)) to your program.
# After doing this, I learned _x1 = _y0, so I replaced all occurrences
# of _y0 with _x1\. Now the context looks like this.

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y1 : Nat  # erases to y1.size(1)
  x : Tensor([_x0, _x1])
  y : Tensor([_x1, _y1])

```

胜利！

```
> matmul(x, y)

# This type-checked OK!  I set a = _x0, b = _x1, c = _y1 and all of the
# arguments checked out.  You're good to go.

```

将此会话内容提取回我们的代码，我们现在有：

```
  main() {
    x = load("tensor1.t")
    y = load("tensor2.t")
    assert_eq_nat!(x.size(), 2)
    assert_eq_nat!(y.size(), 2)
    assert_eq_nat!(x.size(1), y.size(0))
    matmul(x, y)
}

```

* * *

此时，我必须坦白：我上面描述的编译时调试器实际上并不存在。但它与交互式证明助理的证明模式并没有太大不同，这是自动定理证明社区今天使用的。但与定理证明不同的是，我们有一个秘密武器：在困难时刻，强者会变成运行时检查。传统智慧认为自动定理证明需要过于理想化的设置才能在今天的软件编写中有用。传统智慧错了。
