<!--yml

分类：未分类

日期：2024 年 7 月 1 日 18:17:39

-->

# 透明的 xmobar：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/11/transparent-xmobar/`](http://blog.ezyang.com/2011/11/transparent-xmobar/)

## 透明的 xmobar

我应该在做的事情：*研究生院个人陈述。*

我过去五个小时实际上在做的事情：*透明的 xmobar。*

它使用了可怕的“从根 X 窗口获取像素图”的 hack。你可以在这里[获取补丁](https://github.com/ezyang/xmobar/)，但我还没有投入足够的精力来使其成为一个可配置的选项；如果你只是编译了那个分支，你会得到一个 alpha 值为 100/255、着色为黑色的 xmobar。（该算法需要一些工作来正确地泛化到不同的着色；欢迎提出建议！）也许其他人会提供一个更完善的补丁。（也应该鼓动更完整的 XRender 绑定集！）

这在与支持几乎相同的着色和透明行为的 trayer 非常配合得不错。Trayer 在 Oneiric 上也很好，因为它合理地调整了新的电池图标的大小，而 stalonetray 则没有。如果你想知道为什么字体看起来是抗锯齿的，那是因为我[编译时启用了 XFT 支持](http://projects.haskell.org/xmobar/#optional-features)。

（而且是的，显然我的电池容量是 101%。加油！）

*更新。* 功能已经美化并且可以配置。在你的配置文件中调整`alpha`值：0 表示完全透明，255 表示不透明。我已经提交了一个拉取请求。
