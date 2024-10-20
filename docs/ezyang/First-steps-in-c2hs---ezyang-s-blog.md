<!--yml

category: 未分类

date: 2024-07-01 18:18:17

-->

# c2hs 的第一步：ezyang 的博客

> 来源：[ezyang 博客](http://blog.ezyang.com/2010/06/first-steps-in-c2hs/)

这是 [关于 c2hs 的六部分教程系列中的第四部分](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/)。今天我们讨论 c2hs 中的简单事物，即类型、枚举、指针、导入和上下文指令。

*Prior art.* c2hs 支持的所有指令都在 [“tutorial”页面](http://www.cse.unsw.edu.au/~chak/haskell/c2hs/docu/implementing.html) 中简要描述（也许更准确地说是“参考手册”，而非教程）。此外，在 c2hs 的 [研究论文](http://www.cse.unsw.edu.au/~chak/papers/Cha99b.html) 中，对大多数指令也有更为非正式的介绍。

*Type.* C 代码偶尔包含宏条件重新定义类型的情况，具体取决于某些构建条件（以下是真实代码）：

```
#if       defined(__ccdoc__)
typedef platform_dependent_type ABC_PTRUINT_T;
#elif     defined(LIN64)
typedef unsigned long ABC_PTRUINT_T;
#elif     defined(NT64)
typedef unsigned long long ABC_PTRUINT_T;
#elif     defined(NT) || defined(LIN) || defined(WIN32)
typedef unsigned int ABC_PTRUINT_T;
#else
   #error unknown platform
#endif /* defined(PLATFORM) */

```

如果你想要编写引用使用 `ABC_PTRUINT_T` 函数的 FFI 代码，你可能需要对 Haskell 中值的真实情况进行猜测或使用 C 预处理器重新实现条件。使用 c2hs，你可以通过 `type` 获取 typedef 的真实值：

```
type ABC_PTRUINT_T = {#type ABC_PTRUINT_T #}

```

考虑一个 64 位 Linux 系统的情况（`__ccdoc__`未定义，`LIN64`已定义），则结果是：

```
type ABC_PTRUINT_T = CLong

```

*Enum.* 枚举在编写良好的（即避免魔术数字）C 代码中经常出现：

```
enum Abc_VerbLevel
{
   ABC_PROMPT   = -2,
   ABC_ERROR    = -1,
   ABC_WARNING  =  0,
   ABC_STANDARD =  1,
   ABC_VERBOSE  =  2
};

```

然而，在底层，这些实际上只是整数（ints），因此希望在 Haskell 代码中将枚举值传递给函数的代码必须：

1.  创建一个新的数据类型来表示枚举，并

1.  编写一个函数，将该数据类型映射到 C 整数，然后再次映射回来，以便创建 `Enum` 实例。

我们可以让 c2hs 为我们完成所有工作：

```
{#enum Abc_VerbLevel {underscoreToCase} deriving (Show, Eq) #}

```

变成了：

```
data Abc_VerbLevel = AbcPrompt | AbcError | AbcWarning | AbcStandard | AbcVerbose
  deriving (Show, Eq)
instance Enum Abc_VerbLevel
  fromEnum AbcPrompt = -2
  -- ...

```

注意，由于 `ABC_PROMPT` 在 Haskell 中是一个非常难看的构造函数，我们使用如上述的 `underscoreToCase` 算法转换名称。您也可以明确列出这些重命名：

```
{#enum Abc_VerbLevel {AbcPrompt, AbcError, AbcWarning, AbcStandard, AbcVerbose} #}

```

或者更改数据类型的名称：

```
{#enum Abc_VerbLevel as AbcVerbLevel {underscoreToCase} #}

```

还有另外两种变换（可以与 `underscoreToCase` 结合使用：`upcaseFirstLetter` 和 `downcaseFirstLetter`，尽管我不确定后者何时会导致有效的 Haskell 代码。

*Pointer.* 与指定在 `Foreign.C.Types` 中的 C 原语不同，Haskell 需要告知如何将指针类型（`foo*`）映射到 Haskell 类型。考虑某些结构体：

```
struct foobar {
  int foo;
  int bar;
}

```

完全有可能在 Haskell 代码库中存在 `data Foobar = Foobar Int Int`，在这种情况下，我们希望 `Ptr Foobar` 表示原始 C 代码中的 `struct foobar*`。c2hs 无法直接推导出这些信息，因此我们向其提供这些信息：

```
{#pointer *foobar as FoobarPtr -> Foobar #}

```

这生成了以下代码：

```
type FoobarPtr = Ptr Foobar

```

但更重要的是，允许 c2hs 在为 FFI 绑定编写的签名中放置更具体的类型（我们将在本系列的下一篇文章中看到）。

一些主题的变种：

+   如果你想表示一个不会进行马歇尔处理的不透明指针，你可以选择空数据声明：

    ```
    data Foobar
    {#pointer *foobar as FoobarPtr -> Foobar #}

    ```

    或者你可以让 c2hs 使用新类型技巧生成代码：

    ```
    {#pointer *foobar as FoobarPtr newtype #}

    ```

    我更喜欢空数据声明，因为在这种情况下不需要包装和解包新类型：新类型将生成：

    ```
    newtype FoobarPtr = FoobarPtr (Ptr FoobarPtr)

    ```

    如果代码期望 `Ptr a`，则需要将其解包。

+   如果你不喜欢 `FoobarPtr` 这个名称，而只想显式地说 `Ptr Foobar`，你可以告诉 c2hs 不要发出类型定义，使用 `nocode`：

    ```
    {#pointer *foobar -> Foobar nocode #}

    ```

+   如果没有指定 Haskell 名称映射，它将简单地使用 C 名称：

    ```
    -- if it was struct Foobar...
    {#pointer *Foobar #}

    ```

+   如果你想引用 C 中已经是指针的 typedef，只需省略星号：

    ```
    typedef struct Foobar*   FoobarPtr
    {#pointer FoobarPtr #}

    ```

+   c2hs 也支持有限的声明指针为 foreign 或 stable，并相应地生成代码。我没有在这方面使用过，除了一个情况，发现指针的生成绑定不够灵活。效果可能有所不同。

*导入.* 包含多个头文件的 C 库可能会有一些头文件包含其他头文件以获取重要的类型定义。如果你组织你的 Haskell 模块类似地，你需要模仿这些包含：这可以通过 import 来实现。

```
{#import Foobar.Internal.Common #}

```

特别是，这会设置来自其他模块的 `pointer` 映射，并生成通常的 `import` 语句。

*上下文（可选）.* 上下文有两个所谓的目的。第一个是指定文件中 FFI 声明应链接的库；然而，在 Cabal 中，这实际上没有任何作用——所以你仍然需要将库添加到 `Extra-libraries`。第二个是通过为你引用的每个 C 标识符添加隐式前缀来节省击键次数，假设原始的 C 代码被命名空间为 `gtk_` 或类似的。我个人喜欢不需要将我的导入限定到更低级别的 API，并喜欢 C 前缀的视觉区分，所以我倾向于省略这一点。一些指令允许你在局部改变前缀，特别是 `enum`。

*下次.* [使用 get 和 set 进行马歇尔处理](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/)。
