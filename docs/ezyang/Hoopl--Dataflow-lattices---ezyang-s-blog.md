<!--yml

category: 未分类

date: 2024-07-01 18:17:54

-->

# Hoopl：数据流格 lattice ：ezyang’s 博客

> 来源：[`blog.ezyang.com/2011/04/hoopl-dataflow-lattices/`](http://blog.ezyang.com/2011/04/hoopl-dataflow-lattices/)

数据流优化的本质是*分析*和*转换*，并且毫不奇怪，一旦您定义了中间表示，您与 Hoopl 的大部分工作将涉及在基本块图上定义分析和转换。分析本身可以进一步分为我们正在计算的*数据流事实*的规范化，以及我们在分析过程中如何推导这些数据流事实。在这个[Hoopl 系列的第二部分](http://blog.ezyang.com/2011/04/hoopl-guided-tour-base-system/)中，我们将看看分析背后的基本结构：*数据流格 lattice*。我们讨论使用格的理论原因，并且给出您可以为诸如常量传播和活跃性分析等优化定义的格的示例。

* * *

尽管其听起来复杂的名称，数据流分析与人类程序员在不实际在计算机上运行代码的情况下推理代码的方式非常相似。我们从对系统状态的一些初始信念开始，然后随着我们逐步执行指令，我们会用新信息更新我们的信念。例如，如果我有以下代码：

```
f(x) {
  a = 3;
  b = 4;
  return (x * a + b);
}

```

在函数的最顶部，我对`x`一无所知。当我看到表达式`a = 3`和`b = 4`时，我知道`a`等于`3`，`b`等于`4`。在最小的表达式中，我可以使用常量传播来简化结果为`x * 3 + 4`。确实，在没有控制流的情况下，我们可以将分析简单地看作是逐行执行代码并更新我们的假设，也称为*数据流事实*。我们可以在两个方向上做到这一点：我们刚刚完成的分析是*前向分析*，但我们也可以进行*反向分析*，这在活跃性分析的情况下是如此。

唉，如果事情能这么简单就好了！这里有两个问题：Y 型控制流和循环控制流。

Y 型控制流（称为*连接*，原因显而易见，也因为很快就会变得明显）之所以被命名为这样，是因为有两条明显的执行路径，然后合并成一条。然后我们对程序状态有两种不同的信念，我们需要在继续之前调和这些信念：

```
f(x) {
  if (x) {
    a = 2; // branch A
  } else {
    a = 3; // branch B
  }
  return a;
}

```

在分支 A 内部，我们知道`a`是 2，在分支 B 内部，我们知道`a`是 3，但在此条件之外，我们只能说`a`是 2 或 3。（由于两个可能的值对于常量传播并不是很有用，我们将代替说`a`的值是*top*：没有一个值代表变量的持有值。）结果是，您正在进行分析的任何一组数据流事实必须定义一个`join`操作：

```
data DataflowFactsTry1 a = DF1 { fact_join :: a -> a -> a }

```

反向分析也有类似的情况，当你有一个条件跳转时发生，两个控制流的“未来”再次连接在一起，因此需要进行类似的连接。

循环控制流也有连接，但它们面临更进一步的问题，即我们不知道其中一个传入的代码路径的状态是什么：我们在分析循环体之前无法弄清楚，但要分析循环体，我们需要知道传入状态是什么。这是一个进退两难的局面！解决这个问题的技巧是定义一个*底部*事实，直观地表示可能的最保守数据流事实：当它与其他数据流事实结合时，它是身份。因此，当我们遇到这些循环边时，与其尝试计算边（这是一个进退两难的问题），我们反而输入底部元素，并得到该循环边上事实的近似值。如果这个近似值比底部更好，我们就用新结果替代旧的，并且这个过程重复，直到不再有变化为止：达到了*不动点*。

有兴趣数学的人可能会注意到，我们所定义的看起来非常像*格*：

```
data DataflowLattice a = DataflowLattice
 { fact_name       :: String          -- Documentation
 , fact_bot        :: a               -- Lattice bottom element
 , fact_join       :: JoinFun a       -- Lattice join plus change flag
                                      -- (changes iff result > old fact)
 }
type JoinFun a = Label -> OldFact a -> NewFact a -> (ChangeFlag, a)

```

这里有一点额外的噪音：`Label` 严格用于调试目的（它告诉连接函数连接正在进行的标签），而 `ChangeFlag` 用于优化目的：它让 `fact_join` 在达到不动点时高效地通知`NoChange`。

> *旁注：格.* 在这里，我们回顾了一些关于格的基本术语和直觉。格是一个偏序集，对于所有元素对存在*最小上界*（lub）和*最大下界*（glb）。如果想象一个[哈斯图](http://blog.ezyang.com/2010/12/hussling-haskell-types-into-hasse-diagrams/)，最小上界的存在意味着我可以从两个元素向上沿图追溯，直到找到一个共享元素；最大下界是向下同样的过程。最小上界也称为两个元素的*连接*，最大下界称为*交*。（我更喜欢 lub 和 glb 因为我总是搞混 join 和 meet！）在符号上，最小上界用逻辑或符号或方括号并运算符表示，而最大下界用逻辑与符号或方括号交运算符表示。符号的选择具有暗示性：逻辑符号的重载对应于逻辑命题可以使用一种特殊类型的格——布尔代数——定义其语义，其中 lub 等价于或，glb 等价于且（底部是虚假，顶部是公理）。集合运算符的重载对应于关于幂集构造的通常顺序上的格：lub 是集合并运算，glb 是集合交运算。
> 
> 在 Hoopl 中，我们处理有界格，即存在*顶部*和*底部*元素的格。这些是特殊元素，比所有其他元素大（相应地，小于所有其他元素）。将底部元素与任何其他元素连接是一个无操作：另一个元素是结果（这就是为什么我们使用底部作为初始化值的原因！）将顶部元素与任何其他元素连接会导致顶部元素（因此，如果你达到顶部，你就“卡住”了，可以这么说）。
> 
> 对于一丝不苟的人，严格来说，Hoopl 不需要格：相反，我们需要一个有界半格（因为我们只需要定义连接，而不是相遇）。还有另一个不足之处：Lerner-Grove-Chambers 和 Hoopl 使用底部和连接，但大多数现有的数据流格文献使用顶部和相遇（实质上，将格上下颠倒）。事实上，哪种选择“自然”取决于分析：正如我们将看到的，活性分析自然倾向于使用底部和连接，而常量传播则建议使用顶部和相遇。为了与 Hoopl 保持一致，我们将始终使用底部和连接；只要我们保持一致，格的方向就不重要。

* * *

现在我们将具体示例用于活性分析和常量传播的数据流格。这两个示例展示了要查看的格的良好分布：活性分析是变量名的集合，而常量传播是变量名到可能值的平面格的映射。

活性分析（`Live.hs`）使用非常简单的格，因此它作为设置`DataflowLattice`所涉及的额外仪式的良好入门示例：

```
type Live = S.Set Var
liveLattice :: DataflowLattice Live
liveLattice = DataflowLattice
  { fact_name = "Live variables"
  , fact_bot  = S.empty
  , fact_join = add
  }
    where add _ (OldFact old) (NewFact new) = (ch, j)
            where
              j = new `S.union` old
              ch = changeIf (S.size j > S.size old)

```

类型`Live`是我们数据流事实的类型。这代表的是活跃的变量集合（即，稍后代码将使用的变量）：

```
f() {
  // live: {x, y}
  x = 3;
  y = 4;
  y = x + 2;
  // live: {y}
  return y;
  // live: {}
}

```

记住，活性分析是*反向*分析：我们从过程的底部开始并向上工作：变量的使用意味着它在其上方的所有点都是活跃的。我们用文档、显著元素（底部）和这些事实的操作（连接）填写`DataflowLattice`。`Var`是`Expr.hs`，只是变量的字符串名称。我们的底部元素（用于初始化我们无法立即计算的边缘）是空集，因为在任何过程的底部，所有变量都是死的。

连接是集合并，可以在此示例中清楚地看到：

```
f (x) {
  // live: {a,b,x,r} (union of the two branches,
  // as well as x, due to its usage in the conditional)
  a = 2;
  b = 3;
  if (x) {
    // live: {a,r}
    r = a;
  } else {
    // live: {b,r}
    r = b;
  }
  // live: {r}
  return r;
  // live: {}
}

```

我们还看到一些计算更改`ch`的代码，这是集合大小比较的简单方式，因为并集只会增加集合的大小，而不会减少它。`changeIf`是一个实用函数，将`Bool`转换为`ChangeFlag`。

如果我们有三个变量，这里是格结构的示意图：它只是幂集构造上的通常排序。

* * *

这是常量传播的格子（`ConstProp.hs`）。虽然与活跃集相比稍微复杂一些，但部分复杂性被 Hoopl 提供的一些实用数据类型和函数隐藏了起来。

```
-- ConstFact:
--   Not present in map => bottom
--   PElem v => variable has value v
--   Top     => variable's value is not constant
type ConstFact = Map.Map Var (WithTop Lit)
constLattice :: DataflowLattice ConstFact
constLattice = DataflowLattice
 { fact_name = "Const var value"
 , fact_bot  = Map.empty
 , fact_join = joinMaps (extendJoinDomain constFactAdd) }
 where
   constFactAdd _ (OldFact old) (NewFact new)
       = if new == old then (NoChange, PElem new)
         else               (SomeChange, Top)

```

在这个结构中实际上有两个格子。 “外部” 格子是映射，其中底部元素是空映射，加入是将两个映射合并在一起，使用内部格子合并元素。 “内部” （半）格子是 `WithTop Lit`，由 Hoopl 提供。（可以说内部格子是点逐点地提升到映射中。）我们在这里举例说明了包含布尔变量的内部格子：

关于内部格点，有一点需要强调的是底部和顶部之间的区别。两者都表示一种“不知道变量内容”的状态，但在底部的情况下，变量可能是常量也可能不是常量，而在顶部的情况下，变量肯定不是常量。很容易搞混的是，“底部意味着我们不知道变量的值是什么”，而“顶部意味着变量的值可以是任何东西”。如果我们把这个格子看作是一个集合，其中 `{True}` 表示这个变量的值为真，则 `{True,False}`（底部）表示变量可能是*常量*真或*常量*假，而不是变量可以是真或假。这也意味着我们可以恰当地解释 `{}`（顶部）：对于这个变量来说，没有一个值是常量。（注意，这是倒置的幂集格点！）

在这个例子中有几个有趣的实用函数：`extendJoinDomain` 和 `joinMaps`。`extendJoinDomain` 免去了我们完全编写与顶部的所有交互的麻烦，例如：

```
constFactAddExtended _ (OldFact old) (NewFact new)
   = case (old, new) of
        (Top, _) -> (NoChange, Top)
        (_, Top) -> (SomeChange, Top)
        (PElem old, PElem new) | new == old -> (NoChange, PElem new)
                               | otherwise -> (SomeChange, Top)

```

`joinMaps` 将我们的内部格点提升为映射形式，并且处理了 `ChangeFlag` 的连接（如果新映射中的任何条目在旧映射中不存在，或者加入的条目发生了变化，则输出 `SomeChange`）。

* * *

这就结束了我们关于 Hoopl 和数据流格子的讨论。我们还没有涵盖 Hoopl 提供的所有操作数据流格子的函数；以下是一些进一步查看的模块：

+   `Compiler.Hoopl.Combinators` 定义了 `pairLattice`，它对两个格子进行了乘积构造。它可以用来同时执行多个分析。

+   `Compiler.Hoopl.Pointed` 定义了许多辅助数据结构和函数，用于向现有数据类型添加 `Top` 和 `Bottom`。这就是 `extendJoinDomain` 的来源。

+   `Compiler.Hoopl.Collections` 和 `Compiler.Hoopl.Unique` 定义了在唯一键上的映射和集合（最突出的是标签）。您很可能会在数据流格子中使用这些。

下次，我们将讨论转移函数，这是我们计算数据流事实的机制。

*进一步阅读。* 数据流格被覆盖在《编译器原理、技术与工具》（红龙书）的第 10.11 章中。原始论文是基尔达尔在 1973 年发表的《统一的全局程序优化方法》。有趣的是，红龙书指出：“它并没有被广泛使用，可能是因为系统节省的工作量不及诸如语法分析器生成器等工具。” 我觉得这在传统编译器优化中是正确的，但对于 Lerner-Grove-Chambers 风格的通过程来说可能不是（其中分析和重写是交错进行的）。
