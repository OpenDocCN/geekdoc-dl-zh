<!--yml

类别：未分类

日期：2024-07-01 18:16:57

-->

# Python 中用数据类和 Union 定义惯用的代数数据类型：ezyang 博客

> 来源：[`blog.ezyang.com/2020/10/idiomatic-algebraic-data-types-in-python-with-dataclasses-and-union/`](http://blog.ezyang.com/2020/10/idiomatic-algebraic-data-types-in-python-with-dataclasses-and-union/)

在非 Haskell 编程语言中，我最怀念的特性之一就是代数数据类型（ADT）。ADT 在其他语言中类似于对象，但有更多限制：对象是一个开放的宇宙，客户端可以实现在定义时未知的新子类；ADT 是一个封闭的宇宙，ADT 的定义精确地指定了所有可能的情况。我们经常认为限制是一件坏事，但在 ADT 的情况下，限制为封闭的宇宙使程序更易于理解（理解一组固定的案例，而不是可能无限的案例）并且允许新的表达方式（模式匹配）。ADT 使得准确建模数据结构非常容易；它们鼓励您选择精确的类型，使非法状态不可表示。但是，尝试在您使用的每种其他编程语言中手动重新实现您喜爱的 Haskell 语言特性通常不是一个好主意，因此多年来，我在 Python 中遭受了 ADT 无法使用的印象。

然而，最近我注意到 Python 3 中的许多新特性使得可以在 Python 中以惯用的方式使用对象，几乎没有样板文件。关键特性：

+   使用 mypy 的结构静态类型检查系统；特别是声明`Union`类型的能力，这让您可以表示可能是一组其他类型中的一个的值，并通过对其执行`isinstance`检查来细化变量的类型。

+   数据类库允许您方便地定义（可能是不可变的）数据结构，而无需为构造函数编写样板文件。

核心思想是：将每个构造函数定义为一个数据类，将构造函数组合成一个 ADT 使用 Union 类型，并使用`isinstance`测试对结果进行模式匹配。结果与 ADT 一样好（或者可能更好；它们的结构性质更类似于 OCaml 的多态变体）。

下面是它的工作原理。假设您想要定义一个具有两个结果的代数数据类型：

```
data Result
   = OK Int
   | Failure String

showResult :: Result -> String
showResult (OK result) = show result
showResult (Failure msg) = "Failure: " ++ msg

```

首先，我们将每个构造函数定义为一个数据类：

```
from dataclasses import dataclass

@dataclass(frozen=True)
class OK:
    result: int

@dataclass(frozen=True)
class Failure:
    msg: str

```

使用数据类自动生成的构造函数，我们可以使用`OK(2)`或`Failure("something wrong")`构造这些数据类的值。接下来，我们为这两个类的联合定义一个类型同义词：

```
Result = Union[OK, Failure]

```

最后，我们可以通过执行`isinstance`测试对结果进行模式匹配：

```
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))

def showResult(r: Result) -> str:
    if isinstance(r, OK):
        return str(r.result)
    elif isinstance(r, Failure):
        return "Failure: " + r.msg
    else:
        assert_never(r)

```

`assert_never` 是在 [mypy 中做穷尽性检查](https://github.com/python/typing/issues/735) 的一个 [众所周知的技巧](https://github.com/python/typing/issues/735)。如果我们用足够的 `isinstance` 检查未覆盖所有情况，mypy 将会抱怨 `assert_never` 被赋予了 `UnhandledCtor` 类型，而它期望的是 Python 中的不可居住类型 `NoReturn`。

就是这么简单。作为额外的奖励，这种联合类型的写法与 [结构化模式匹配 PEP](https://www.python.org/dev/peps/pep-0634/) 兼容，如果它被实际接受的话。我在最近重写 PyTorch 代码生成器时，已经成功地使用了这种模式。如果你有机会在静态类型的 Python 代码库中工作，不妨试试这种代码风格！
