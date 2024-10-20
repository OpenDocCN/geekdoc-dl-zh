<!--yml

category: 未分类

date: 2024-07-01 18:17:05

-->

# Backpack 和独立编译：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/09/backpack-and-separate-compilation/`](http://blog.ezyang.com/2016/09/backpack-and-separate-compilation/)

## Backpack 和独立编译

在构建一个支持在多个实现之间参数化代码（即 functors）的模块系统时，你会遇到一个重要的实现问题：如何*编译*这些参数化的代码？在现有的语言实现中，有三种主要的思路：

1.  **独立编译**学派认为，你应该独立编译你的 functors，而不管它们的具体实现。这个学派更重视编译时间而非性能：一旦一个 functor 被构建，你可以自由地替换其参数的实现，而无需重新构建 functor，从而实现快速的编译时间。Pre-Flambda 的 OCaml 就是这种方式。缺点是，无法基于实现的知识对 functor 主体进行优化（除非你可能有一个即时编译器可用）。

1.  **使用时专门化**学派说，你可以通过在已知实现的使用点内联 functors 来获得性能提升。如果 functor 主体不太大，你可以在不需要大幅度改变编译器架构的情况下透明地获得良好的性能效益。[Post-FLambda OCaml](https://blogs.janestreet.com/flambda/) 和 C++ 模板在[Borland 模型](https://gcc.gnu.org/onlinedocs/gcc/Template-Instantiation.html)中都是这样工作的。缺点是，代码必须在每个使用点重新优化，并且可能会存在大量的代码重复（这可以在链接时减少）。

1.  **专门化仓库**学派认为，不断重新编译实例化是愚蠢的：相反，每个实例化的编译代码应该被缓存到某个全局位置；下次需要相同实例时，应该重用它。C++ 中的模板在 Cfront 模型和 Backpack 中都是这样工作的。

仓库视角听起来不错，直到你意识到这需要对编译器的工作方式进行重大的架构更改：大多数编译器*不*尝试将中间结果写入某些共享缓存中，而添加对此的支持可能会非常复杂且容易出错。

Backpack 通过将实例化的缓存工作外包给包管理器来规避这个问题，后者*确实*知道如何缓存中间产品。这种折衷是，Backpack 并没有像某些人希望的那样完全整合到 Haskell 本身中（它*极其*不是第一类）。
