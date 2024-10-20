<!--yml

category: 未分类

date: 2024-07-01 18:18:23

-->

# Cup of FP with a Java twist : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/04/cup-of-fp-java-twis/`](http://blog.ezyang.com/2010/04/cup-of-fp-java-twis/)

```
zip: List<A>, List<B> -> List<(A, B)>
zip(Nil, Nil) = Nil
zip(_, Nil) = Nil
zip(Nil, _) = Nil
zip(Cons(a, as), Cons(b, bs)) = Cons((a, b), zip(as, bs))

fst: (A, B) -> A
fst((a, _)) = a

last: List<A> -> A
last(Cons(a, Nil)) = a
last(Cons(a, as)) = last(as)

foldl: (B, A -> B), B, List<A> -> B
foldl(_, z, Nil) = z
foldl(f, z, Cons(x, xs)) = foldl(f, f(z, x), xs)

```

天啊，爱德华，你那里有什么？简直像是 Haskell、Java 和 ML 的变种混合体。

它实际上是由 Daniel Jackson 发明的受 ML 启发的伪语言。它被 [MIT course 6.005](http://ocw.mit.edu/OcwWeb/Electrical-Engineering-and-Computer-Science/6-005Fall-2008/CourseHome/index.htm) 用来教授其学生函数编程概念。它没有编译器或正式规范（尽管我听说助教们正在拼命地研究一种类型），但其语法的最显著点在 [第 10 讲（PDF）](http://ocw.mit.edu/NR/rdonlyres/Electrical-Engineering-and-Computer-Science/6-005Fall-2008/5FC036C0-0505-49AE-BCA2-455E89B1AB18/0/MIT6_005f08_lec10.pdf) 中介绍，当他们开始讨论如何构建 SAT 求解器时。

我们的第二份问题集要求我们在这种伪语言中编写一些代码。不幸的是，作为伪语言，您实际上无法运行它...而且我讨厌写我无法运行的代码。但它确实看起来很像 Haskell...只是更啰嗦了一点。我问课程工作人员是否可以用 Haskell 提交问题集，他们告诉我：“不行，因为课程工作人员不懂。但如果它确实与这种语言如您所说的那么接近，您完成后可以将其翻译成这种语言。”

我就是[这样做的](http://github.com/ezyang/haskell-mit6005)。

这个计划实际上是不可能的，没有一个现有的 [Haskell 漂亮打印程序](http://hackage.haskell.org/packages/archive/haskell-src/1.0.1.3/doc/html/Language-Haskell-Pretty.html) 来为我做大部分的脚手架工作。从那里开始，在适当的函数中混合使用 `<>`、`lparen` 和 `comma` 等朋友来渲染数据类型。[漂亮打印组合器太棒了！](http://hackage.haskell.org/packages/archive/pretty/1.0.1.1/doc/html/Text-PrettyPrint-HughesPJ.html)
