<!--yml

category: 未分类

date: 2024-07-01 18:17:03

-->

# 背包和 PVP：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/12/backpack-and-the-pvp/`](http://blog.ezyang.com/2016/12/backpack-and-the-pvp/)

在 [PVP](http://pvp.haskell.org/) 中，如果向一个模块添加函数，则增加次要版本号；如果从一个模块中移除函数，则增加主要版本号。直观地说，这是因为添加函数是向后兼容的更改，而删除函数是破坏性的更改；更正式地说，如果新接口是旧接口的*子类型*，则只需要增加次要版本号。

Backpack 给混合添加了一个新的复杂性：签名。向签名添加/删除函数的 PVP 政策应该是什么？如果我们将具有必需签名的包解释为一个*函数*，理论告诉我们答案：签名是[逆变](http://blog.ezyang.com/2014/11/tomatoes-are-a-subtype-of-vegetables/)的，因此添加必需函数是破坏性的（增加主要版本号），而**删除**必需函数是向后兼容的（增加次要版本号）。

然而，故事并没有结束。签名可以*重复使用*，即一个包可以定义一个签名，然后另一个包可以重用该签名：

```
unit sigs where
  signature A where
    x :: Bool
unit p where
  dependency sigs[A=<A>]
  module B where
    import A
    z = x

```

在上面的例子中，我们将一个签名放在 sigs 单元中，p 通过对 sigs 声明依赖项来使用它。B 可以访问 sigs 中由 A 定义的所有声明。

但这里有一些非常奇怪的地方：如果 sigs 曾经删除了 x 的声明，p 将会中断（x 将不再在作用域内）。在这种情况下，上述的 PVP 规则是错误的：p 必须始终对 sigs 声明一个精确的版本边界，因为任何添加或删除都将是破坏性的更改。

所以我们处于这种奇怪的情况中：

1.  如果我们包含一个依赖项和一个签名，但我们从未使用过该签名的任何声明，我们可以对依赖项指定一个宽松的版本边界，允许它从签名中删除声明（使签名更容易实现）。

1.  然而，如果我们导入签名并使用其中的任何内容，我们必须指定一个精确的边界，因为现在删除操作将是破坏性的更改。

我认为不应该期望 Backpack 的最终用户能够自行正确地理解这一点，因此 GHC（在这个 [提议的补丁集](https://phabricator.haskell.org/D2906) 中）试图通过向仅来自可能已被指定为宽松边界的包的声明附加此类警告来帮助用户。

```
foo.bkp:9:11: warning: [-Wdeprecations]
    In the use of ‘x’ (imported from A):
    "Inherited requirements from non-signature libraries
    (libraries with modules) should not be used, as this
    mode of use is not compatible with PVP-style version
    bounds.  Instead, copy the declaration to the local
    hsig file or move the signature to a library of its
    own and add that library as a dependency."

```

**更新。** 在发布这篇文章后，我们最终删除了这个错误，因为它在与 PVP 兼容的情况下触发了。（详细信息：如果一个模块重新导出了一个来自签名的实体，那么来自该模块的实体使用将会触发错误，这是由于过时通知的工作方式。）

当然，GHC 对边界一无所知，所以我们使用的启发式方法是，如果一个包不暴露任何模块，则认为它是一个*签名包*，具有精确的边界。像这样的包只通过导入其签名才有用，所以我们从不对这种情况发出警告。我们保守地假设暴露模块的包可能受到 PVP 风格的边界约束，因此在这种情况下会发出警告，例如：

```
unit q where
  signature A where
    x :: Bool
  module M where -- Module!
unit p where
  dependency q[A=<A>]
  module B where
    import A
    z = x

```

正如警告所示，可以通过在 p 中明确指定`x :: Bool`来修复这个错误，这样，即使 q 移除其要求，也不会导致代码破坏：

```
unit q where
  signature A where
    x :: Bool
  module M where -- Module!
unit p where
  dependency q[A=<A>]
  signature A where
    x :: Bool
  module B where
    import A
    z = x

```

或者将签名放入自己的新库中（就像原始示例中的情况一样）。

这个解决方案并不完美，因为仍然有一些方法可以使你以 PVP 不兼容的方式依赖继承的签名。最明显的是与类型相关的情况。在下面的代码中，我们依赖于 q 的签名强制 T 类型等于 Bool 的事实：

```
unit q where
  signature A where
    type T = Bool
    x :: T
  module Q where
unit p where
  dependency q[A=<A>]
  signature A where
    data T
    x :: T
  module P where
    import A
    y = x :: Bool

```

原则上，q 可以放宽对 T 的要求，允许其实现为任何形式（而不仅仅是 Bool 的同义词），但这一变更将破坏 P 中对 x 的使用。不幸的是，在这种情况下并没有简单的方法来发出警告。

也许一个更有原则的方法是禁止来自非签名包的签名导入。然而，在我看来，这样做会使 Backpack 模型变得更加复杂，而这并没有很好的理由（毕竟，总有一天我们会用签名增强版本号，那将是辉煌的，对吧？）

**总结一下。** 如果你想重用来自签名包的签名，请在该包上指定一个*精确的*版本边界。如果你使用的组件是参数化的签名，*不要*导入和使用这些签名的声明；如果你这样做，GHC 会警告你。
