<!--yml

category: 未分类

date: 2024-07-01 18:18:24

-->

# 在 Xournal 和 Gimp 中绘图：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/04/diagramming-in-xournal-and-gimp/`](http://blog.ezyang.com/2010/04/diagramming-in-xournal-and-gimp/)

两个人问我如何为我的上一篇文章[You Could Have Invented Zippers](http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/)绘制图表，我想我会稍微详细地分享一下，因为在我找到适合自己的方法之前，这显然是一些实验。

Linux 的绘图软件太糟糕了。Mac OS X 上的人可以使用[OmniGraffle](http://www.omnigroup.com/products/OmniGraffle/)制作令人眼前一亮的美丽图表；我们能做的最好的是一些微不足道的 GraphViz 输出，或者也许如果我们有很多时间，从 Inkscape 中精心制作的 SVG 文件。这对我来说太费时间了。

因此，对我来说，手绘图表！我做的第一件事是打开我信赖的[Xournal](http://xournal.sourceforge.net/)，这是由[Denis Auroux](http://www-math.mit.edu/~auroux/)（我的前多元微积分教授）编写的基于 GTK 的高质量笔记应用程序。然后我开始画图。

实际上，这并不完全正确；到这个时候，我已经花了一些时间用铅笔和纸笔画图，并想出了我想要的布局。因此，当我在平板电脑上时，我脑海中有一个清晰的图像，并仔细地用黑色画出图表。如果我需要图表的多个版本，我会复制粘贴并根据需要调整颜色（电子绘图的一个伟大之处！）。我还会使用荧光笔工具进行着色。完成后，我可能会有几页图表，可能会用到也可能不会。

从那里开始，“File > Export to PDF”，然后在 Gimp 中打开生成的 PDF 文件。有一段时间，我没有意识到可以这样做，而是使用`scrot`来截取屏幕截图。Gimp 会询问你要导入哪些页面；我导入了所有页面。

每一页都放在单独的“层”中（这有点无用，但不会太有害）。然后我裁剪一个逻辑图表，另存为结果（请求 Gimp 合并可见层），然后撤销以返回到全屏状态（并裁剪另一选择）。当我完成一页后，我将其从可见层中删除，然后转到下一页。

当所有工作都完成时，我有一个标记图像的目录。我根据需要使用`convert -resize XX% ORIG NEW`调整它们的大小，然后将它们放入公共文件夹中以供链接。

*附言.* Kevin Riggle 提醒我，在同一图中不要混合绿色和红色，除非我想要困扰我的色盲朋友。Xournal 有一个黑色、蓝色、红色、绿色、灰色、青色、石灰色、粉红色、橙色、黄色和白色的调色板，这有点限制。不过我敢打赌，你可以通过在*src/xo-misc.c*中混淆`predef_colors_rgba`来切换它们的顺序
