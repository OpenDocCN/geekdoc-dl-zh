<!--yml

类别：未分类

日期：2024-07-01 18:18:21

-->

# fclabels 的非必需指南：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/04/inessential-guide-to-fclabels/`](http://blog.ezyang.com/2010/04/inessential-guide-to-fclabels/)

上次我做了一个[关于数据访问器的非必需指南](http://blog.ezyang.com/2010/04/inessential-guide-to-data-accessor/)，所有人告诉我，“你应该使用 fclabels！”所以这里是伙伴指南，fclabels 的非必需指南。像数据访问器一样，其目标是使记录的访问和编辑变得更加顺畅。然而，它为您提供了一些更有用的抽象。它在您的记录之上使用了模板哈斯克尔，因此与数据访问器不兼容。

*识别*。有三个显著特征：

1.  类型签名中包含`:->`（“哦，看起来有点像函数箭头...但不是？好奇怪！”），

1.  记录包含具有前导下划线的字段（与数据访问器约定使用尾随下划线不同），以及

1.  一个`import Prelude hiding (id, (.), mod)`，用来自`Control.Category`的导入替换它们。

*解释类型*。*标签*由`r :-> a`表示，其中包含一个 getter`r -> a`和一个 setter`a -> r -> r`。在内部，包装的标签仅仅是一个*点*，一个由`r -> a`和`b -> r -> r`组成的结构，要求`a`等于`b`。（稍后我们将看到，点本身非常有用，但不适用于基本功能。）

*访问记录字段*。

```
get fieldname record

```

*设置记录字段*。

```
set fieldname newval record

```

*修改记录字段*。对于`fieldname :: f a :-> a`，`modifier`应具有类型`a -> a`。

```
mod fieldname modifier record

```

*访问、设置和修改子记录字段*。组合使用点运算符`(.)`进行组合，但不能使用 Prelude 中的那个，因为那个只适用于函数。这种组合被视为您正在组合 getter。

```
get (innerField . outerField) record
set (innerField . outerField) newVal record
mod (innerField . outerField) modifier record

```

*访问器胜于应用程序*。您可以使用`fmapL`将访问器提升到应用程序上下文中。如果您的记录实际上是`Maybe r`（您可以将`r :-> a`转换为`Maybe r :-> Maybe a`），这将非常有用。

但等等，还有更多！

*更有趣的视图*。请记住，点是一个 getter 和一个 setter，但它们不必是相同类型的。结合巧妙的应用实例，我们可以使用这一点逐步构建由多个标签组成的标签。结果看起来很像您在关系数据库上创建的视图。配方如下：

1.  有生成类型的构造函数（例如`(,)`，元组构造函数），

1.  对于生成类型的所有访问器（例如`fst`和`snd`），以及

1.  您希望组合在一起的标签（比如`label1`和`label2`）。

使用`for`结合每个生成类型的访问器（2）和要访问该访问器的标签（3），将所有这些生成的点与生成类型的构造函数结合使用，即`<$>`和`<*>`，然后将其放入带有`Label`的标签中：

```
(,) <$> fst `for` label1 <*> snd `for` label2

```

令人惊讶的是，你将无法混淆放置访问器（2）的参数的方向；结果将无法通过类型检查！（详见*后记*中的更详细的论证。）

*与镜头（lenses）更多有趣的事情*。一个函数暗示着方向性：从 a 到 b。但光可以通过镜头双向过滤，因此镜头代表了双向函数。我们可以通过一个标签`f :-> a`通过一个镜头`a :<->: b`来获取一个新的标签`f :-> b`（请记住，与常规函数的组合不够，因为我们需要放入值和取出值）。必须小心你的镜头指向的方向。如果`label :: r :-> a`，`in :: b -> a`和`out :: a -> b`，那么：

```
(out <-> in) `iso` label :: r :-> b
(in <-> out) `osi` label :: r :-> b

```

如果`a != b`，其他方向将无法通过类型检查。

你可以使用`lmap`将一个镜头提升为一个函子（它简单地在两个方向上运行`fmap`）。

*进一步阅读*。[Hackage 文档](http://hackage.haskell.org/package/fclabels)中有大量优秀的例子。

*后记*。以我们原始的例子为例：

```
label1 :: r -> a
label2 :: r -> b
(,) <$> fst `for` label1 <*> snd `for` label2 :: r :-> (a, b)

```

我们考虑我们构建的点的类型，在与应用实例组合之前：

```
fst `for` label1 :: Point Person (a, b) a
snd `for` label2 :: Point Person (a, b) b

```

我们有一个共享的应用函子`Point Person (a, b)`，如果我们将其视为`f`，显然：

```
(,) :: a -> b -> (a, b)
fst `for` label1 :: f a
snd `for` label2 :: f b
(,) <$> fst `for` label1 <*> snd `for` label2 :: f (a, b)

```

这等同于`Point Person (a, b) (a, b)`，这是一个有效的`Label`。

那么`for`在做什么呢？源代码文档说：

> 将部分析构器与标签结合起来，轻松地在隐藏的点数据类型的应用实例中使用 getter 中的协变，setter 中的逆变双函子映射函数。（请参考示例，因为这个函数本身无法单独解释。）

嗯，我要忽略这个建议，因为你已经看过例子了。让我们来解析一下。`for`在 getter `r -> a`中是协变的，在 setter `a -> f -> f`中是逆变的。这些术语来自范畴论，用来描述函子。协变函子是“正常”的函子，而逆变函子是将组合反过来的函子。因此，虽然通常情况下 `fmap f g == f . g`，在逆变世界中 `fmap f g == g . f`：

```
for :: (i -> o) -> (f :-> o) -> Point f i o
for a b = dimap id a (unLabel b)

```

好吧，我们对 getter 并没有做太多有趣的事情，但我们将`a :: (a, b) -> a`（在我们的例子中）映射到 setter `a -> f -> f`上。幸运的是（对于困惑的人来说），协变映射不会类型检查（`(a, b) != (f -> f)`），但是逆变映射会：`(a, b) -> f -> f`，这是一个新的 setter，接受`(a, b)`，正是我们从类型签名中期望的。

因此，`for`设置了我们的 setter 和部分 getter，并且应用实例完成了我们的 getter 设置。
