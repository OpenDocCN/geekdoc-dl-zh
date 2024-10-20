<!--yml

category: 未分类

date: 2024-07-01 18:18:14

-->

# Call and fun: marshalling redux : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/`](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)

这是 [c2hs 的六部分介绍之一](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/)。最终我们谈论的是 c2hs 的主要用途：从 Haskell 调用 C 函数。由于 c2hs 对 C 头文件有了解，因此可以自动生成 FFI 导入的工作。`call` 钩子只是告诉 c2hs 生成 FFI 导入，而 `fun` 钩子则生成另一个执行 marshalling 的 Haskell 函数。

*Call.* 调用的格式非常简单，因为像 `get` 和 `set` 一样，它意味着可以与其他 Haskell 代码交错使用。如果我想从 `readline/readline.h` 调用 `readline` 函数，只需 `{#call readline #}` 即可；c2hs 将会生成正确签名的 FFI 导入，并将调用指令转换为 FFI 导入的名称。

当然，`readline` 不会回调到 Haskell，所以我们可以添加 `unsafe`：`{#call unsafe readline #}`。如果你确信 C 函数没有副作用，可以添加 `pure`：`{#call pure unsafe sin #}`。如果多次调用同一个函数并使用相同的 FFI 声明，它们的标志需要保持一致。

默认情况下，`cid` 将精确用于确定 FFI 导入的名称；如果它不是函数的有效 Haskell 标识符（即大写），或者 C 函数名称会与其他名称冲突，则需要指定 FFI 将导入为什么。常见的约定包括给函数添加前缀 `c_`，或者使用 `^` 来进行 c2hs 的大写转换。`{#call FooBar_Baz as ^ #}` 将转换为 `fooBarBaz`（并带有适当的 FFI 声明）。

*Fun.* 因为 FFI 声明的签名都是 C 类型，而 Haskell 程序往往不使用这些类型，并且因为频繁进行 C 类型的转换，有一点自动化来帮助你处理 `fun` 指令。与 `call` 不同，它是作为一个定义独立存在的，而不是嵌入在代码中的。请注意，*不* 必须使用 `fun`；例如 gtk2hs 就没有使用它，但很多人发现它很有用。

`fun` 的开始与 `call` 类似：首先指定它是纯的和/或不安全的，指定 C 标识符，然后指定 Haskell 名称。由于大多数代码将引用 Haskell 名称，通常最好指定 `^` 来保持一致的命名约定。

在这里，我们需要指定 *所需 Haskell 函数的最终类型*，以及 *如何从这些类型转换为 C 类型（即 marshalling 函数）*。c2hs 教程在这个主题上有一些内容，所以我们将采取更多的示例导向的方法。

*基本 C 类型。* 整数、浮点数和布尔（通常在幕后是整数）基本的 C 类型非常普遍，如果没有指定，c2hs 将自动使用`cIntConv`、`cFloatConv`和`cFromBool`/`cToBool`函数进行编组。这些函数可以双向工作。这个指令：

```
{#fun pure sinf as ^
  { `Float' } -> `Float' #}

```

生成：

```
sinf :: Float -> Float
sinf a1 =
  let {a1' = cFloatConv a1} in
  let {res = sinf'_ a1'} in
  let {res' = cFloatConv res} in
  (res')

```

您可以看到，会添加一堆（丑陋的）生成代码以在参数上运行编组函数，将其传递给 FFI，然后在结果上调用另一个编组函数。惯用的 Haskell 可能是这样的：

```
sinf = cFloatConv . sinf'_ . cFloatConv

```

如果您想要为编组函数使用不同的名称，可以在参数类型之前指定它（“in”编组器），或者在结果之后指定它（“out”编组器），如下所示：

```
{#fun pure sinf as ^
  { myFloatConv `Float` } -> `Float` myFloatConv

```

而您只需在生成的 Haskell 中替换相关函数调用。

*String 参数。* 字符串也在 c2hs 的心中占有特殊地位；处理以 null 结尾的字符串和需要显式长度信息的字符串都很容易。考虑这两个函数原型：

```
void print_null_str(char *str);
void print_explicit_str(char *str, int length);

```

我们可以编写以下 c2hs 指令：

```
{#fun print_null_str as ^ { `String' } -> `()' }
{#fun print_explicit_str as ^ { `String'& } -> `()' }

```

并且它们将自动使用`withCString*`和`withCStringLen*`进行编组。

这里发生了几件有趣的事情。我们使用`()`（Haskell 中的空类型）来表示空返回类型。此外，`print_explicit_str`中的 String 参数有一个附加的和号；这意味着编组器应该产生一个参数元组，这些参数将作为两个单独的参数传递给函数。确实，`withCStringLen`的结果是`(Ptr CChar, Int)`，而 c2hs 使用略有不同的变体`withCStringLenIntConv`，它将`Int`转换为`CInt`。（请注意，如果您需要更复杂的多参数排序，`fun`并不适合您。）

但也许最有趣的是附加到输入编组器上的`*`，它有两个效果。首先，它表明输入编组函数是 IO 单子，例如，`withCString`的类型是`String -> (CString -> IO a) -> IO a`。但更重要的是，它指示了一个遵循括号资源模式“with”的函数。我们没有使用`String -> CString`，因为如果我们不稍后释放`CString`，这可能导致内存泄漏！然后生成的代码是：

```
printNullStr :: String -> IO ()
printNullStr a1 =
  withCString a1 $ \a1' ->
  printNullStr'_ a1' >>= \res ->
  return ()

printExplicitStr :: String -> IO ()
printExplicitStr a1 =
  withCStringLenIntConv a1 $ \(a1'1, a1'2) ->
  printExplicitStr'_ a1'1  a1'2 >>= \res ->
  return ()

```

使用悬挂 lambda 保持布局一致。

*编组结构参数。* 尽管 c2hs 文档声称如果您在 C 中有以下情况，那么会有一个默认的编组器：

```
struct my_struct { int b; int c; };
void frob_struct(struct my_struct *);

```

并在 Haskell 中：

```
data MyStruct = MyStruct Int Int
instance Storable MyStruct where ...
{#pointer *my_struct as MyStructPtr -> MyStruct #}

```

因此，您应该能够写出：

```
{#fun frob_struct as ^ { `MyStruct' } -> `()' #}

```

其中，输入编组器是`with*`。不幸的是，我从未能让它起作用；此外，c2hs 认为`with`是一个保留字，所以您需要重命名它才能使用它。

```
withT = with
{#fun copy_struct as ^ { withT* `MyStruct' } -> `()' #}

```

*不透明指针参数。* 当您不想在 Haskell 中对指针执行任何花哨的操作时，可以简单地指定指针是参数并使用`id`作为编组器。在前面的例子中，`copy_struct`也可以另外定义为：

```
{#fun copy_struct as ^ { id `MyStructPtr' } -> `()' #}

```

一个约定是，如果只处理不透明指针，可以省略指针类型的名称中的 `Ptr`。

*输出编组器的输入参数。* C 代码中的一个常见模式是使用指针参数允许函数返回多个结果。例如，`strtol` 的签名如下：

```
long int strtol(const char *nptr, char **endptr, int base);

```

`endptr` 指向一个指针，该指针将设置为我们解析的 `nptr` 字符串部分的结尾处的指针。如果我们不关心它，可以将 `endptr = NULL` 设置为 `NULL`。

显然，我们不希望我们的 Haskell 函数这样做，并且我们有更简单的方法使用元组返回多个结果，所以 c2hs 有一个关于输入参数的输出编组器的概念。它还有一个“虚假”输入参数的概念，用户不必传递，以防我们的函数完全负责分配指向函数的指针的内存。

这是编写 `strtol` 的 `fun` 钩子的第一个尝试：

```
{#fun strtol as ^ {id `Ptr CChar', id `Ptr (Ptr CChar)', `Int'} -> `Int` #}

```

我们避开了默认的字符串编组，因为否则 `endptr` 不会给我们非常有趣的信息。这个版本是原始内容的简单复制。

为了改进这一点，我们认为 `Ptr (Ptr CChar)` 是返回 `Ptr CChar` 的一种方式。因此，在函数运行后，我们应该 `peek`（解引用指针）并返回结果：

```
{#fun strtol as ^ {id `Ptr CChar', withT* `Ptr CChar' peek*, `Int'} -> `Int' #}

```

`peek` 在 IO 中，所以它需要星号，但对于我们的编组器来说，它并不会导致任何复杂的括号使用。现在，这个函数的 Haskell 返回类型不是 `Int`；它是 `(Int, Ptr CChar)`。

```
strtol :: Ptr CChar -> Ptr CChar -> Int -> IO (Int, Ptr CChar)
strtol a1 a2 a3 =
  let {a1' = id a1} in
  withT a2 $ \a2' ->
  let {a3' = cIntConv a3} in
  strtol'_ a1' a2' a3' >>= \res ->
  peek a2'>>= \a2'' ->
  let {res' = cIntConv res} in
  return (res', a2'')

```

由于我们要覆盖指针的原始内容，强制用户向我们传递它并没有多大意义。我们可以在我们的输入编组器后缀上加`-`，以表明它不是真正的 Haskell 参数，并改用`alloca`代替：

```
{#fun strtol as ^ {id `Ptr CChar', alloca- `Ptr CChar' peek*, `Int'} -> `Int' #}

```

请注意，我们去掉了 `*`；这是一种或另一种方式。现在我们有一个可用的函数：

```
strtol :: Ptr CChar -> Int -> IO (Int, Ptr CChar)
strtol a1 a3 =
  let {a1' = id a1} in
  alloca $ \a2' ->
  let {a3' = cIntConv a3} in
  strtol'_ a1' a2' a3' >>= \res ->
  peek a2'>>= \a2'' ->
  let {res' = cIntConv res} in
  return (res', a2'')

```

或者，在习惯用法的 Haskell 中：

```
strtol nptr base = alloca $ \endptr -> do
  result <- strtol'_ nptr endptr (cIntconv base)
  end <- peek endptr
  return (result, end)

```

*错误处理。* 还有一个功能片段我们尚未讨论，即一个输出编组器上的 `-` 标志，导致 Haskell 忽略结果。单独使用时通常没有用，但与 `*`（表示操作在 IO 中）结合使用时，可用于附加检查错误条件并在情况成立时抛出异常的函数。请记住，`()` 的默认输出编组器是 `void-`，忽略函数的输出结果。
