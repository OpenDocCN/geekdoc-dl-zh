<!--yml

分类：未分类

日期：2024-07-01 18:17:03

-->

# AST 中的向后兼容故事：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/12/a-tale-of-backwards-compatibility-in-asts/`](http://blog.ezyang.com/2016/12/a-tale-of-backwards-compatibility-in-asts/)

那些推崇向后兼容价值的人常常声称，向后兼容仅仅是永远不 *移除* 事物的问题。但是，任何发布涉及数据结构的 API 的人都知道，事实并非如此简单。我想描述一下我在 Cabal 文件格式上最近正在处理的一个向后兼容问题的思考过程。像往常一样，我对您可能有的任何见解和评论都很感兴趣。

**现状。** 在 Cabal 文件中，`build-depends` 字段用于声明对其他包的依赖关系。格式是一个逗号分隔的包名称和版本约束的列表，例如 `base >= 4.2 && < 4.3`。抽象地说，我们将其表示为 `Dependency` 的列表：

```
data Dependency = Dependency PackageName VersionRange

```

在 `build-depends` 中的一项的效果是双重的：首先，它指定了一个版本约束，依赖解决器在选择包的版本时会考虑这一约束；其次，它将该包的模块引入作用域，以便可以使用这些模块。

**扩展。** 我们在 Cabal 中添加了对 "内部库" 的支持，允许你在单个包中指定多个库。例如，假设你正在编写一个库，但是有一些内部函数你想暴露给测试套件而不是一般公众。你可以将这些函数放在一个内部库中，该库被公共库和测试套件依赖，但不对外部包可用。

想了解更多动机，请参阅原始的 [功能请求](https://github.com/haskell/cabal/issues/269)，但出于本博客文章的目的，我们关注如何指定对其中一个内部库的依赖问题。

**尝试 #1：保留旧语法。** 我对内部库的新语法的第一个想法是保留 `build-depends` 的语法 *不变*。要引用名为 `foo` 的内部库，你只需写 `build-depends: foo`；一个内部库会遮蔽同名的任何外部包。

向后兼容？绝对不是。请记住，`build-depends` 中条目的最初解释是 *包* 名称和版本范围。因此，如果你的代码假设 `build-depends` 中的每个条目实际上是一个外部包，当指定一个依赖于内部库时，它会以意想不到的方式中断。这正是 cabal-install 的依赖解决器所发生的，需要更新以过滤掉对应于内部库的依赖关系。

有人可能会认为，如果使用了新功能，旧代码破坏是可以接受的。但是，对以这种方式重载包名称存在更大的哲学上的反对意见：如果... 实际上并不是一个包名称，那就不要称之为包名称！

**尝试＃2：一种新的语法。** 受到这种哲学关注的启发，以及您无法同时引用名为`foo`的内部库和名为`foo`的外部包的问题，我们引入了一种新的句法形式：要引用包`pkg`中的内部库`foo`，我们写`build-depends: pkg:foo`。

由于有一个新的句法形式，我们的内部 AST 也必须更改以处理这种新形式。显而易见的做法是引入一种新类型的依赖：

```
data BuildDependency =
  BuildDependency PackageName
                  (Maybe UnqualComponentName)
                  VersionRange

```

并声明`build-depends`的内容是`BuildDependency`的列表。

当涉及数据表示的更改时，这是一个“最佳情况”，因为我们可以轻松编写一个函数`BuildDependency -> Dependency`。因此，假设我们用于描述库构建信息的数据结构看起来像这样：

```
data BuildInfo = BuildInfo {
    targetBuildDepends :: [Dependency],
    -- other fields
  }

```

我们可以通过将`targetBuildDepends`转换为一个函数来保持向后兼容性，该函数读取新的扩展字段并将其转换为旧形式：

```
data BuildInfo = BuildInfo {
    targetBuildDepends2 :: [BuildDependency],
    -- other fields
  }

targetBuildDepends :: BuildInfo -> [Dependency]
targetBuildDepends = map buildDependencyToDependency
                   . targetBuildDepends2

```

关键是，这利用了 Haskell 中记录选择器看起来像函数的事实，因此我们可以用函数替换选择器而不影响下游代码。

不幸的是，这实际上并不是真的。Haskell 还支持*记录更新*，让用户可以按照以下方式覆盖字段：`bi { targetBuildDepends = new_deps }`。如果我们查看 Hackage，实际上有大约十几个使用`targetBuildDepends`的方式。因此，如果我们想维持向后兼容性，就不能删除这个字段。不幸的是，Haskell 不支持重载记录更新的含义（也许这里要学到的教训是你永远不应该导出记录选择器：而是导出一些镜头）。

可能，在平衡中，破坏十几个软件包是一个公平的代价来支付这样的变化。但让我们假设我们坚决要保持 BC。

**尝试＃3：保留两个字段。** 保持旧代码正常工作的一种简单方法是保留两个字段：

```
data BuildInfo = BuildInfo {
    targetBuildDepends  :: [Dependency],
    targetBuildDepends2 :: [BuildDependency],
    -- other fields
  }

```

我们引入了一个新的不变量，即`targetBuildDepends bi == map buildDependencyToDependency (targetBuildDepends2 bi)`。看到问题了吗？任何更新`targetBuildDepends`的旧代码可能不知道要更新`targetBuildDepends2`，破坏不变量，可能导致一些非常令人困惑的错误。呃。

**尝试＃4：做一些数学。** 上面表示的问题是冗余的，这意味着我们必须添加不变量来“减少”类型下可接受值的空间。通常，我们喜欢“紧密”的类型，因此，正如 Yaron Minsky 所说，我们“使非法状态不可表示”。

为了更仔细地思考这个问题，让我们将其转化为数学形式。我们有一个`Old`类型（同构于`[(PN, VR)]`）和一个`New`类型（同构于`[(PN, Maybe CN, VR)]`）。`Old`是`New`的子空间，因此我们有一个众所周知的注入`inj :: Old -> New`。

当用户更新`targetBuildDepends`时，他们会应用一个函数`f :: Old -> Old`。在使我们的系统向后兼容时，我们隐式地定义了一个新函数`g :: New -> New`，它是`f`的扩展（即`inj . f == g . inj`）：这个函数告诉我们在新系统中对旧更新的*语义*是什么。一旦我们有了这个函数，我们就试图将`New`分解为`(Old, T)`，使得将`f`应用于`(Old, T)`的第一个分量会给你一个新值，这个新值等同于将`g`应用于`New`的结果。

因为在 Haskell 中，`f`是一个不透明的函数，我们实际上无法实现许多“常识性”的扩展。例如，我们可能希望`f`更新所有`parsec`出现为`parsec-new`，相应的`g`也做同样的更新。但是我们无法区分一个更新`parsec`的`f`和一个删除`parsec`依赖，然后添加`parsec-new`依赖的`f`。在双向编程世界中，这就是[基于状态和基于操作的方法之间的区别](https://www.cis.upenn.edu/~bcpierce/papers/lenses-etapsslides.pdf)。

如果`f`只能添加依赖项，我们真的只能做一些合理的事情，比如这样写：

```
data BuildInfo = BuildInfo {
    targetBuildDepends :: [Dependency],
    targetSubLibDepends :: [(PackageName, UnqualComponentName)],
    targetExcludeLibDepends :: [PackageName],
    -- other fields
  }

```

从这里到`BuildDependency`的转换大致如下：

1.  对于`targetBuildDepends`中的每个`Dependency pn vr`，如果包名称在`targetExcludeLibDepends`中未提及，我们有`BuildDependency pn Nothing vr`。

1.  对于`targetSubLibDepends`中的每个`(pn, cn)`，如果存在一个匹配的`Dependency pn vr`（即包名称匹配），我们有`BuildDependency pn (Just cn) vr`。

暂时退一步，*这真的是我们想写的代码*吗？如果修改不是单调的，我们将陷入麻烦；如果有人读取`targetBuildDepends`然后将其写入一个全新的`BuildInfo`，我们将陷入麻烦。真的值得为了实现如此小的、容易出错的向后兼容性而费这么大的劲吗？

**结论。** 我仍然不确定我要采取什么样的方法来处理这个特定的扩展，但似乎有几个教训：

1.  记录对于向后兼容性来说是不好的，因为没有办法重载记录更新与自定义的新更新。更新的镜头会更好。

1.  记录更新对于向后兼容性来说是不好的，因为它将我们置于*双向编程*的领域，要求我们将旧世界的更新反映到新世界中。如果我们的记录是只读的，生活会轻松得多。另一方面，如果有人设计了一种明确考虑向后兼容性的编程语言，双向编程最好能够出现在你的工具箱中。

1.  向后兼容性可能在治愈中更糟。你是希望你的软件在编译时出问题，因为确实需要考虑这个新情况，还是希望所有东西都继续编译，但如果新功能被使用，会以微妙的方式破坏？

你怎么看？我不会自称是向后兼容性问题的专家，非常希望看到你的参与，无论是关于我应该采取哪种方法，还是关于编程语言与向后兼容性交互的一般想法。
