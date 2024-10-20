<!--yml

category: 未分类

date: 2024-07-01 18:18:09

-->

# 会话类型、子类型和依赖类型：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/09/session-types-subtyping-and-dependent-types/`](http://blog.ezyang.com/2010/09/session-types-subtyping-and-dependent-types/)

在我研究会话类型编码时，我注意到了一些有趣的事情：即会话类型在捕捉协议控制流程时，实际上正在实现某种强烈让人联想到依赖类型的东西。

任何合理的会话类型编码都需要能够表示选择：在 Simon Gay 的论文中，这是 `T-Case` 规则，在 Neubauer 和 Thiemann 的工作中，这是 `ALT` 运算符，在 Pucella 和 Tov 的实现中，这是 `:+:` 类型运算符，以及 `offer`、`sel1` 和 `sel2` 函数。通常会指出，二进制交替方案在用户界面上较之名称为基础的交替方案要差，但后者实现起来更为困难。

这些论文的作者真正要求的是支持某种看起来像依赖类型的东西。当您尝试为现有协议编写会话类型编码时，这一点变得更加明显。考虑来自 Google 的 SPDY 的以下小片段：

> 一旦流创建，就可以用来发送任意数量的数据。通常这意味着将在流上发送一系列数据帧，直到设置了包含 FLAG_FIN 标志的帧为止。FLAG_FIN 可以在 SYN_STREAM、SYN_REPLY 或 DATA 帧上设置。一旦发送了 FLAG_FIN，流就被视为半关闭。

数据帧的格式为：

```
+----------------------------------+
|C|       Stream-ID (31bits)       |
+----------------------------------+
| Flags (8)  |  Length (24 bits)   |
+----------------------------------+
|               Data               |
+----------------------------------+

```

而`offer`是通过网络传输单个比特来实现的，在这里，控制流是否关闭的关键比特深藏在数据之中。因此，如果我甚至想*考虑*编写会话类型编码，我必须使用带有额外幻影类型的数据定义，而不是显而易见的类型：

```
data DataFrame fin = DataFrame StreamId FlagFin Data

```

我不得不将 `FlagFin` 从普通术语提升为适合于 `fin` 穴位的类型，这种做法明显具有依赖类型的味道。幸运的是，依赖类型的需求被事实上会话类型将立即在类型上进行案例分割所回避，考虑到它是真的情况和它是假的情况。我们在编译时不知道值实际上将是什么，但事实证明我们并不在乎！如果我们小心地只允许 `fin` 在 `FlagFin` 实际上为 `True` 时才能作为记录中的字段，我们甚至不需要将 `FlagFin` 作为记录中的一个字段。

当人们说你可以在不使用依赖类型的情况下玩弄类型技巧时，我相信他们在指的是这一点。将编译时已知的值推入类型是一个明显的例子（Peano 整数，有人？），但在这种情况下，我们通过处理所有可能的情况，将编译时未知的值推入类型！

啊呀，在 Haskell 中实际做这件事情相当麻烦。考虑一些现实世界中的代数数据类型，一个简化版本的 SPDY 协议，它只允许同时处理一条流：

```
data ControlFrame = InvalidControlFrame
                  | SynStream FlagFin FlagUnidirectional Priority NameValueBlock
                  | SynReply FlagFin NameValueBlock
                  | RstStream StatusCode
                  | Settings FlagSettingsClearPreviouslyPersistedSettings IdValuePairs
                  | NoOp
                  | Ping Word32
                  | Headers NameValueBlock
                  | WindowUpdate DeltaWindowSize

```

每个构造函数都需要转换为一个类型，`FlagFin` 也一样，但事实证明其他数据对会话类型不重要。因此，我们最终为每个构造函数编写了一个数据声明，而将它们有效地串联起来的好方法并不存在：

```
data RstStream
data SynStream fin uni = SynStream Priority NameValueBlock
data SynReply fin = SynReply NameValueBlock
...

```

我们在这里寻找的线索是子类型化，具体来说是更为奇特的和类型的子类型化（与产品类型的子类型化相对应，一般称为记录子类型化）。另一种思考方式是，我们的类型现在表示了可能出现在变量中的一组有限的可能项：随着程序的发展，越来越多的项可能出现在这个变量中，我们需要进行案例分割，以减少可能性，使其更易管理。

啊呀，我听说子类型化会大大增加推断的复杂性。哎，这是我考虑到的尽头。毫无疑问，肯定有一篇论文存在于某处，我应该读一读，以澄清这一点。你觉得呢？
