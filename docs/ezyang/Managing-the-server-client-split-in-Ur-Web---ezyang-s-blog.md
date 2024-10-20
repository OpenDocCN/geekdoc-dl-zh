<!--yml

类别：未分类

日期：2024-07-01 18:17:29

-->

# 管理 Ur/Web 中的服务器/客户端分离：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/07/managing-the-server-client-split-in-ur-web/`](http://blog.ezyang.com/2012/07/managing-the-server-client-split-in-ur-web/)

Web 应用程序开发的圣杯是一种*单一语言*，既可以在服务器端运行，也可以在客户端运行。其原因是多方面的：单一语言促进了组件的重用，不再需要在两种语言中重新实现，并且允许服务器和客户端之间进行更轻松的通信。明确努力处理服务器和客户端的 Web 框架包括[Meteor](http://www.meteor.com/)、[Ur/Web](http://www.impredicative.com/ur/)、[Opa](http://opalang.org/)和[Google Web Toolkit](https://developers.google.com/web-toolkit/overview)。

任何希望构建这样一个系统的人面临的最大实施困难之一是存在多个运行时：服务器运行时和浏览器运行时，每个都有相应不同的原语和可用的 API。此外，我们可能希望某些代码仅存在于服务器上，而不发送到客户端。当某个语言特性可以在两个运行时上实现时，我们保持客户端和服务器不可区分的假象；当无法时，这种假象就破灭了。

因此，为了支持在这种集成语言中运行时特定的 FFI 调用，必须回答以下问题：

1.  代码何时发送到客户端，何时保留在服务器上？这些信息必须对用户进行公开（而不是作为“实现细节”保留）。

1.  如何强制在服务器上执行？

1.  如何强制在客户端执行？

在这篇博文中，我将讨论[Ur/Web](http://www.impredicative.com/ur/)如何解决这些问题。答案相当简单（如果有幸可以推广到其他类似系统），但如果你把编译器视为黑匣子，那么找到它们就相当困难。

### 1\. 客户端/服务器分割

解决客户端/服务器分割问题的一个明显解决方案是标记入口点（例如主函数或 onClick 处理程序）为从服务器（main）或客户端（onClick）开始，然后进行可达性分析以标记所有其他函数。因此，在下面的 Ur/Web 代码中，`txn : transaction unit` 将在这里在服务器上执行：

```
fun main () =
  txn;
  return <xml><body>Done!</body></xml>

```

在此处它将在客户端执行：

```
fun main () =
  return <xml><body><a onclick={txn}>Click me!</a></body></xml>

```

当给定像这样的片段时：

```
fun foo body =
  r <- txn body;
  return <xml>{r}</xml>

```

无法知道`txn`是否在客户端或服务器端需要执行，除非分析所有调用方并检查它们是客户端还是服务器端。像这样的情况对于强制服务器端或客户端行为至关重要。

### 2\. 强制服务器端

假设我们希望强制`txn`在服务器端执行。如果我们已经在服务器上，那就没有更多的事情要做了。但是，如果我们在客户端，我们需要向服务器发起 RPC 调用。在 Ur/Web 中，这很容易实现：

```
fun fooClient body =
  r <- rpc (txn body);
  return <xml>{r}</xml>

```

然而，由于`rpc`在 Ur/Web 中仅限于客户端功能，我们无法再将此功能用于服务器端计算。这种选择的一个后果是，它迫使我们明确何时发生 RPC，这对于理解和安全性来说是个好消息。

### 3\. 强制客户端执行

假设我们希望强制`txn`在客户端执行。这很棘手：如果我们已经在客户端，我们可以像往常一样继续，但如果我们在服务器端执行，*在客户端执行某些代码*是什么意思呢？

一种解释是这样的：因为我们正在构建一些要显示给客户端的 HTML，当客户端实际显示 HTML 时，应该运行`txn`。最近 Ur/Web 添加了`active`标签，实现了这种效果：

```
fun fooServer body =
  return <xml><active code={txn body} /></xml>

```

`code`属性的行为与`onclick`和其他类似属性类似，因为它定义了一个入口点，当在浏览器中显示时自动运行。它仍然是一个事件处理程序，因为如果有人调用了`fooServer`，但随后没有使用 HTML，`txn`就不会被调用：`active`可以被视为一种延迟执行。

如果我们真的希望客户端立即执行某些代码，我们最好的选择是将`active`标签插入到与活动`dyn`元素挂钩的`source`中：

```
fun fooServer body source =
  set source <xml><active code={txn body} /></xml>;
  return <xml>???</xml>

```

但在这种情况下，实际上不可能询问客户端计算的结果（服务器不允许阻塞！）这种移动代码传递方法甚至可以异步完成，使用通道：

```
fun fooServerAsync body channel =
  send channel <xml><active code={txn body} /></xml>;
  return <xml>???</xml>

```

### 4\. 与优化器的交互

HTML 事件处理程序中的代码（例如`onclick={...}`和`active code={...}`）可以有自由变量，这些自由变量绑定到它们的词法作用域中的变量，这些变量可能是从服务器计算的。因此，在这种情况下，您可以期望`foo: int -> xbody`在服务器上执行：

```
fun main n =
  let val x = foo n
  in return <xml><body><active code={txn; return x} /></body></xml>

```

然而，Ur/Web 的优化器太聪明了：由于`foo`是纯的，因此引用透明，它总是可以安全地内联（特别是当只有一个使用点时）：

```
fun main n =
  return <xml><body><active code={txn; return (foo n)} /></body></xml>

```

这样写，清楚地表明`foo`是从客户端运行的。因此，如果`foo`是一个在客户端未实现的服务器 FFI 调用，那么一个无辜的转换就可能破坏您的代码。

令人困扰的结论是变量替换*可能使有效的程序无效*。当然，在急切评估、不纯的语言中，变量替换是无效的。但我们可能期望在像 Ur/Web 这样的纯语言中是真的。无论如何，我们可以通过在我们的`urp`文件中将`foo`标记为`benignEffectful`来教会 Ur/Web 不内联。

### 5\. 结论

总的来说，在编写 Ur/Web 应用程序时，以下是一些有用的指导原则：

1.  始终在你的`urp`文件中用`serverOnly`和`clientOnly`标记仅限服务器或仅限客户端的标识符。Ur/Web 通常会适当处理单向 FFI 函数，但如果你的代码利用了只在一侧实现的语言特性（例如服务器端的闭包），请确保适当标记这些函数。

1.  使用`rpc`从客户端到服务器传输，使用`active`从服务器到客户端传输。由于“`rpc`必须引用命名函数”的不变性，Ur/Web 应用程序的一般结构将是服务器代码块内部嵌入的客户端代码。

1.  如果你对生成包含纯服务器计算数据的客户端代码感兴趣，请确保计算该数据的函数被标记为`benignEffectful`。

1.  一般来说，不必担心服务器/客户端分离！Ur/Web 会在需要移动东西时提醒你，但大部分情况下，事情应该顺利进行。

最后关于在共享服务器/客户端模型中的安全性问题：Ur/Web 如何确保输入验证最终在服务器端而不是客户端？这相当简单：程序中唯一关心输入验证的部分是涉及持久数据的部分，而所有这些函数都是仅限服务器的。因此，任何传递给这些函数的用户数据必定通过顶级页面处理程序或`rpc`，这使得确保验证在“正确”的一侧非常简单。如果你使用的是正确构造的数据结构，那么验证就会自动完成！
