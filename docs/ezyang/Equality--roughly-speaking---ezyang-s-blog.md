<!--yml

category: 未分类

date: 2024-07-01 18:17:15

-->

# 等式，粗略地说：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/01/equality-roughly-speaking/`](http://blog.ezyang.com/2014/01/equality-roughly-speaking/)

## 等式，粗略地说

在《软件基础》中，等式是[以这种方式定义的](http://www.cis.upenn.edu/~bcpierce/sf/Logic.html#lab220)：

> 即使是 Coq 的等式关系也不是内建的。它有（大致）以下归纳定义。
> 
> ```
> Inductive eq0 {X:Type} : X -> X -> Prop :=
>   refl_equal0 : forall x, eq0 x x.
> 
> ```

*为什么是“粗略地说”？* 好吧，事实证明，Coq 对等式的定义略有不同（重新格式化以匹配《软件基础》的呈现）：

```
Inductive eq1 {X:Type} (x:X) : X -> Prop :=
  refl_equal1 : eq1 x x.

```

什么是区别？诀窍在于查看 Coq 为每个这些生成的归纳原理：

```
eq0_ind
   : forall (X : Type) (P : X -> X -> Prop),
     (forall x : X, P x x) -> forall y y0 : X, eq0 y y0 -> P y y0

eq1_ind
   : forall (X : Type) (x : X) (P : X -> Prop),
     P x -> forall y : X, eq1 x y -> P y

```

在我们的同伦类型论阅读小组中，Jeremy 指出这两个原则之间的区别正是路径归纳（eq0）和基于路径归纳（eq1）之间的确切区别。（这在[同伦类型论书](http://homotopytypetheory.org/book/)的第 1.12 节中有涵盖）因此，Coq 使用略微更奇怪的定义，因为它恰好更方便一些。（我确信这是传统知识，但直到现在我才注意到这一点！欲了解更多，请阅读 Dan Licata 的[优秀博文](http://homotopytypetheory.org/2011/04/10/just-kidding-understanding-identity-elimination-in-homotopy-type-theory/)。）
