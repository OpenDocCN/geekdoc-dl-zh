<!--yml

类别：未分类

日期：2024 年 07 月 01 日 18:17:21

-->

# AST 类型问题：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/05/the-ast-typing-problem/`](http://blog.ezyang.com/2013/05/the-ast-typing-problem/)

这篇[《Lambda the Ultimate》帖子（2010 年）](http://lambda-the-ultimate.org/node/4170)描述了编译器编写者面临的一个相当普遍的问题：如何向 AST 添加“额外信息”（例如类型）？（帖子本身将问题分为三个组成部分：将信息添加到数据类型中，使用信息来指导节点的构建，使用信息来指导节点的销毁，但我只对如何定义数据类型感兴趣。）在这篇帖子中，我想总结解决这个问题的方法，这些方法在这篇帖子中被描述，并且看看一些真实世界的编译器是如何做的。运行示例 lambda 演算如下：

```
data Exp = Num Int
         | Bool Bool
         | Var Var
         | If Exp Exp Exp
         | Lambda Var Exp
         | App Exp Exp
data Type = TyInt | TyBool | TyArrow Type Type

```

### 单独的 IR，其中节点带有类型装饰

低技术解决方案：如果您需要一个包含更多信息的新版本 IR，只需定义一个新的 IR 类型，其中每个节点也可以携带信息。使这些定义更简洁的一个技巧是创建一个相互递归的数据结构。[[1]](http://lambda-the-ultimate.org/node/4170#comment-63834)

```
type TExp = (TExp', Type)
data TExp' = TNum Int
           | TBool Bool
           | TVar Var
           | TIf TExp TExp TExp
           | TLambda Var TExp
           | TApp TExp TExp

```

尽管（或许正因为）它的简单性，这种方法在许多编译器中非常受欢迎，特别是在 ML 社区中。一些例子包括 OCaml（parsetree/typedtree）、MLton（AST/CoreML）和 Ikarus Scheme。部分原因是从前端语言到类型化语言的转换还伴随着其他一些变化，当定义一个新的 AST 时，这些变化也可以结合在一起。

### 可空字段

无原则解决方案：使用一个 AST，但有一个可选字段，可以插入信息。[[2]](http://lambda-the-ultimate.org/node/4170#comment-63832)

```
type TExp = (TExp', Maybe Type)
data TExp' = TNum Int
           | TBool Bool
           | TVar Var
           | TIf TExp TExp TExp
           | TLambda Var TExp
           | TApp TExp TExp

```

不再进行进一步评论。

### 显式类型化

虽然与单独的 IR 解决方案密切相关，但明确类型化的 IR 采取的方法是不为每个节点装饰类型，而是安排任何给定节点的类型可以仅使用局部信息快速计算。[[3]](http://lambda-the-ultimate.org/node/4170#comment-63884)

```
data TExp = TNum Int
          | TBool Bool
          | TVar Var
          | TIf TExp TExp TExp
          | TLambda Var Type TExp
          | TApp TExp TExp

```

在这里，`TExp`和`Exp`之间的区别非常微小；`TLambda`用显式类型为绑定器进行了注释。就类型检查而言，这是一个天壤之别：我们不再需要查看 lambda 外部来确定绑定器可能是什么。

强制使您的 IR 明确类型化通常是出于元理论原因一个好主意，因为复杂的类型系统通常没有可判定的推理算法。GHC 的核心 IR、Ur/Web 的核心和 Coq 都以这种方式明确类型化。

### 两级类型

通过延迟递归数据结构的节点连接时机，您可以安排基本函子同时为无类型和类型表示提供服务。[[4]](http://lambda-the-ultimate.org/node/4170#comment-63836)

```
data ExpF a = Num Int
            | Bool Bool
            | Var Var
            | If a a a
            | Lambda Var a
            | App a a
newtype Exp = Exp (ExpF Exp)
newtype TExp = TExp (ExpF TExp, Type)

```

Coq 内核使用这种方法来定义其表达式类型，尽管它不用它来定义一个无类型的变体。

### （惰性）属性语法

我不敢说我太理解这种方法，但它本质上是一种与通常的代数数据类型不同的编程模型，它将树的节点上的属性关联起来。在某种意义上，它可以被视为从 AST 节点到属性的记忆函数。许多编译器确实使用映射，但仅用于顶层声明。[[5]](http://lambda-the-ultimate.org/node/4170#comment-63903)
