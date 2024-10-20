<!--yml

类别：未分类

日期：2024-07-01 18:17:58

-->

# FFI 绑定的两个小贴士：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/02/two-short-tips-for-ffi-binding/`](http://blog.ezyang.com/2011/02/two-short-tips-for-ffi-binding/)

## FFI 绑定的两个小贴士

主题：[Haskell-cafe] 请审阅我的 Xapian 外部函数接口

谢谢 Oliver！

我还没有时间仔细查看你的绑定，但我有一些初步的想法：

+   你手工编写你的导入。其他几个项目曾经这样做，但当你需要绑定数百个函数并且没有完全正确地执行时，这会很麻烦，然后由于 API 不匹配而导致段错误。考虑使用像 [c2hs](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/) 这样的工具，可以排除这种可能性（并减少你需要编写的代码！）

+   我看到了很多 `unsafePerformIO`，但没有考虑可中断性或线程安全性。使用 Haskell 的人往往希望他们的代码是线程安全和可中断的，所以我们的标准很高;-) 但是，即使看起来是线程安全的 C++ 代码，在底层可能会修改共享内存，所以要仔细检查。

我使用 Sup，因此我日常处理 Xapian。绑定看起来不错。
