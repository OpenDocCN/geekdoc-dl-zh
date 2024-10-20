<!--yml

category: 未分类

date: 2024-07-01 18:18:33

-->

# rxvt-unicode for gnome-terminal refugees : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/01/rxvt-unicode-for-gnome-terminal-refugees/`](http://blog.ezyang.com/2010/01/rxvt-unicode-for-gnome-terminal-refugees/)

当我从 Ubuntu 的默认 Gnome 桌面切换到平铺窗口管理器 [Xmonad](http://xmonad.org/) 时，我保留了 Gnome Terminal，尽管删除了菜单栏和滚动条。我从默认的白色切换到了一个很好的 #2B2B2B 色调（[Sup](http://sup.rubyforge.org/) 最初向我介绍的一种色调）。

然而，随着时间的推移，当我在切换窗口时（在平铺窗口管理器中是一个常见的任务，特别是在屏幕尺寸相对较小时），我对 Gnome Terminal 渲染速度缓慢感到越来越恼火；基本症状是在旧终端离开和新终端绘制过程中屏幕会闪白。在测试了 xterm 并发现在切换窗口时它并*不*会闪白后，我决定寻找一个更快的终端仿真器；在 David Benjamin 的建议下，我最终选择了 rxvt-unicode，也称为 urxvt。

rxvt-unicode 是 X 终端仿真器自豪传统的一部分，因此其设置由 X 窗口管理器管理（而不是 gnome-terminal 使用的 gnome-settings）。您可以使用名为`xrdb`的程序在运行时操纵设置；但我发现将我想要的设置放在 `.Xdefaults` 中大多数时候更加简单，它会在会话启动时自动加载。语法很简单：一个以换行分隔的文件，格式为 `Appname*option: value`。在 rxvt-unicode 的情况下，Appname 是 `URxvt`。

使用了很长时间的 gnome-terminal，我有点不愿意放弃我所喜爱的颜色和行为。因此，这是我的 `.Xdefaults` 文件，带有关于各部分作用的注释：

```
URxvt*background: #2B2B2B
URxvt*foreground: #DEDEDE
URxvt*font: xft:Monospace:pixelsize=12
URxvt*boldFont: xft:Monospace:bold:pixelsize=12
URxvt*saveLines: 12000
URxvt*scrollBar: false
URxvt*scrollstyle: rxvt

```

这些部分都相当容易理解；rxvt-unicode 支持抗锯齿字体，这意味着粗体文本看起来很好（这也是我无法忍受 xterm 的主要原因之一，因为粗体字体在没有抗锯齿的情况下往往会相互混合）。

```
URxvt*perl-ext-common: default,matcher
URxvt*urlLauncher: firefox
URxvt*matcher.button: 1
URxvt*colorUL: #86a2be

```

这些行实现了终端内可点击的 URL。启动器在光标上不会给出任何视觉提示，但我发现下划线和颜色变化已经足够。

```
! black
URxvt*color0  : #2E3436
URxvt*color8  : #555753
! red
URxvt*color1  : #CC0000
URxvt*color9  : #EF2929
! green
URxvt*color2  : #4E9A06
URxvt*color10 : #8AE234
! yellow
URxvt*color3  : #C4A000
URxvt*color11 : #FCE94F
! blue
URxvt*color4  : #3465A4
URxvt*color12 : #729FCF
! magenta
URxvt*color5  : #75507B
URxvt*color13 : #AD7FA8
! cyan
URxvt*color6  : #06989A
URxvt*color14 : #34E2E2
! white
URxvt*color7  : #D3D7CF
URxvt*color15 : #EEEEEC

```

我非常喜欢 gnome-terminal 的颜色方案，比 rxvt 默认的要低调一些。因此，它被采纳了。第一个颜色是“正常”的；第二个颜色是“亮的”。
