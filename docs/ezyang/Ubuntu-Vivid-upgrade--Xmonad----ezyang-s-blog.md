<!--yml

category: 未分类

date: 2024-07-01 18:17:10

-->

# Ubuntu Vivid 升级（Xmonad）：ezyang 的博客

> 来源：[`blog.ezyang.com/2015/05/ubuntu-vivid-upgrade-xmonad/`](http://blog.ezyang.com/2015/05/ubuntu-vivid-upgrade-xmonad/)

## Ubuntu Vivid 升级（Xmonad）

又是半年过去了，又一次 Ubuntu 升级。这次升级基本上很顺利：唯一出了问题的是我的 xbindkeys 绑定了音量和暂停功能，不过这很容易修复。

### 调高和调低音量

如果之前有：

```
#Volume Up
"pactl set-sink-volume 0 -- +5%"
    m:0x10 + c:123
    Mod2 + XF86AudioRaiseVolume

```

这个语法不再适用：你必须在命令中早些放置双破折号，如下所示：

```
#Volume Up
"pactl -- set-sink-volume 0 +5%"
    m:0x10 + c:123
    Mod2 + XF86AudioRaiseVolume

```

调低音量时也是同样的操作。

### 暂停

如果之前有：

```
#Sleep
"dbus-send --system --print-reply --dest="org.freedesktop.UPower" /org/freedesktop/UPower org.freedesktop.UPower.Suspend"
     m:0x10 + c:150
     Mod2 + XF86Sleep

```

UPower 不再处理暂停功能；你必须将命令发送到登录界面：

```
#Sleep
"dbus-send --system --print-reply --dest=org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager.Suspend boolean:true"
    m:0x10 + c:150
    Mod2 + XF86Sleep

```
