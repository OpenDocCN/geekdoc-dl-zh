<!--yml

category: 未分类

date: 2024-07-01 18:18:20

-->

# 艺术。代码。数学。（以及 mit-scheme）：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/04/art-code-math-and-mit-scheme/`](http://blog.ezyang.com/2010/04/art-code-math-and-mit-scheme/)

我今天在排练中，作为第二管簧管演奏圣桑的管风琴交响曲，这已经是第 n 次了，突然想到：我已经听过并演奏了这首音乐足够多的次数，以至于知道整体流程和大部分管弦乐部分，不仅仅是我的部分。因此，当圣歌的呼叫为最后一乐章的管风琴的胜利入场让路，或者当速度开始变化，同时加快和减慢，在曲末时，这并不令人惊讶；几乎是不可避免的。不能有其他方式。

但我们*本来可以*另有他法；圣桑本可以决定移动第二乐章或引入另一个主题或任何其他多种变化。但他创作了这首曲子，唯独这首曲子，这就是被奉为美的东西。

这让我想起了我计算可计算性问题集上的第一个问题，它要求我展示宇宙的一个基本真理（好吧，在数学哲学家的界限内）；不可协商的、不动的、普遍的。或者我写的程序，当然是一个创造性的过程，但通过需求和规范牢固地锚定在具体的领域。那些数学家和程序员需要多么创造性才能设计出优雅的证明和程序，然而他们离艺术家还有多远。

*不合逻辑的推论*。MIT/GNU Scheme 在你运行它时喜欢冒出大量额外的横幅垃圾，即使你实际上并不想使用交互式 REPL，只是运行一些 mit-scheme 代码。事实证明，mit-scheme 的维护者做出了以下决定：

> 过去，我（CPH）对于稳定版本的政策是在发布之前必须更新文档。实际上，这意味着近年来没有稳定版本。从这个版本开始，我们将不再将更新后的文档视为稳定版本的先决条件。

哎，什么？

无论如何，这里有一个名为`--batch-mode`的奇妙未记录选项，可以抑制入口消息。然而，在 7.7.90 版本（Ubuntu Karmic 的默认版本，别试图自己编译；你需要 mit-scheme 来编译 mit-scheme），它并不能抑制“Loading…”消息，所以你需要用以下小技巧来调用 load：

```
# run-scheme LOAD EVAL
#   LOAD - Scheme file to load
#   EVAL - Scheme expression to evaluate
run-scheme() {
    # --batch-mode doesn't work completely on mit-scheme 7.7.90, in
    # particular it fails to suppress Loading... messages.  As a result,
    # we require this elaborate work-around.
    mit-scheme --batch-mode --eval \
        "(begin (set! load/suppress-loading-message? #t) \
                (load \"$1\") $2)" </dev/null
}

```

简而言之，有点令人失望。
