<!--yml

类别：未分类

日期：2024-07-01 18:18:14

-->

# Groom：用于 Haskell 的人类可读的 Show：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/groom-human-readable-show-for-haskell/`](http://blog.ezyang.com/2010/07/groom-human-readable-show-for-haskell/)

## Groom：用于 Haskell 的人类可读的 Show

在一个复杂的数据结构上敲击，我发现自己面对一堵巨大的语言困境之墙。

“天哪！”我惊叹道，“GHC 的神灵又一次用没有空白的派生 Show 实例咒骂了我！”我不满地自言自语，并开始匹配括号和方括号，扫描文本页以寻找可能告诉我正在寻找的数据的可辨识特征。

但是，我突然想到：“显示被指定为有效的 Haskell 表达式，不带空白。如果我解析它，然后漂亮地打印出生成的 AST 呢？”

几行代码后（借助`Language.Haskell`的帮助）...

[啊，好多了！](http://hackage.haskell.org/package/groom)

*如何使用它。* 在你的 shell 中：

```
cabal install groom

```

以及在你的程序中：

```
import Text.Groom
main = putStrLn . groom $ yourDataStructureHere

```

*更新。* Gleb 提到了 [ipprint](http://hackage.haskell.org/package/ipprint)，它基本上也是做同样的事情，但还有一个`putStrLn . show`的函数，并且有一些调整后的默认设置，包括知道您终端的大小。

*更新 2。* Don 向我提到了 [pretty-show](http://hackage.haskell.org/package/pretty-show) 这个由 Iavor S. Diatchki 开发的软件包，它也具有类似的功能，并配备了一个可让您离线美化输出的可执行文件！
