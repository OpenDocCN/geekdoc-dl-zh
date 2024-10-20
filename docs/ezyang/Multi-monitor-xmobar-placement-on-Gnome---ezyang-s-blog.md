<!--yml

category: 未分类

date: 2024-07-01 18:17:44

-->

# Gnome 上的多显示器 xmobar 放置：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/multi-monitor-xmobar-placement-on-nome/`](http://blog.ezyang.com/2011/06/multi-monitor-xmobar-placement-on-nome/)

## Gnome 上的多显示器 xmobar 放置

本文描述了如何在多显示器设置中更改 xmobar 出现的监视器。对我来说，这一直是一个烦恼，因为在切换到多显示器后，xmobar 可能会显示在正确的监视器上，但如果之后重新启动 XMonad，它会迁移到我的另一个监视器，让我非常恼火。请注意，监视器不同于 X 屏幕，可以使用 `-x` 命令行直接从 xmobar 配置。

如何让 xmobar 选择使用哪个屏幕？它会选择“主要”监视器，默认情况下是您的 `xrandr` 列表中的第一个条目：

```
Screen 0: minimum 320 x 200, current 2464 x 900, maximum 8192 x 8192
VGA1 connected 1440x900+1024+0 (normal left inverted right x axis y axis) 408mm x 255mm
   1440x900       59.9*+   75.0
   1280x1024      75.0     60.0
   1280x960       60.0
   1152x864       75.0
   1024x768       75.1     70.1     66.0     60.0
   832x624        74.6
   800x600        72.2     75.0     60.3     56.2
   640x480        72.8     75.0     66.7     66.0     60.0
   720x400        70.1
LVDS1 connected 1024x768+0+0 (normal left inverted right x axis y axis) 245mm x 184mm
   1024x768       60.0*+
   800x600        60.3     56.2
   640x480        59.9

```

我们可以使用 `xrandr --output $MONITOR --primary` 命令切换主要监视器。但这种更改不是持久的；您每次添加新监视器时都需要运行此命令。

幸运的是，`gnome-settings-daemon` 实际上记录了它看到的监视器信息，以便正确配置它们。此信息位于 `.config/monitors.xml` 中。

```
<monitors version="1">
  <configuration>
      <clone>no</clone>
      <output name="VGA1">
          <vendor>HSD</vendor>
          <product>0x8991</product>
          <serial>0x01010101</serial>
          <width>1440</width>
          <height>900</height>
          <rate>60</rate>
          <x>0</x>
          <y>0</y>
          <rotation>normal</rotation>
          <reflect_x>no</reflect_x>
          <reflect_y>no</reflect_y>
          <primary>no</primary>
      </output>
      <output name="LVDS1">
          <vendor>LEN</vendor>
          <product>0x4002</product>
          <serial>0x00000000</serial>
          <width>1024</width>
          <height>768</height>
          <rate>60</rate>
          <x>1440</x>
          <y>0</y>
          <rotation>normal</rotation>
          <reflect_x>no</reflect_x>
          <reflect_y>no</reflect_y>
          <primary>no</primary>
      </output>
  </configuration>
</monitors>

```

因此，我们只需在适当的监视器上将 `primary` 调整为 `yes`。

特别感谢 David Benjamin 和 Evan Broder 告诉我如何做到这一点。
