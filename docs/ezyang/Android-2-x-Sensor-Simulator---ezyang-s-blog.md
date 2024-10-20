<!--yml

类别：未分类

日期：2024 年 07 月 01 日 18:17:59

-->

# Android 2.x 传感器模拟器：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/02/android-2-x-sensor-simulator/`](http://blog.ezyang.com/2011/02/android-2-x-sensor-simulator/)

## Android 2.x 传感器模拟器

OpenIntents 有一个很棒的应用程序叫做[SensorSimulator](http://code.google.com/p/openintents/wiki/SensorSimulator)，允许您向 Android 应用程序提供加速度计、方向和温度传感器数据。不幸的是，在较新的 Android 2.x 系列设备上表现不佳。特别是：

+   展示给用户的模拟 API 与真实 API 不同。部分原因是原始代码中复制了 Sensor、SensorEvent 和 SensorEventHandler，以解决 Android 没有这些类的公共构造函数的问题，

+   虽然文档声称“无论何时您未连接到模拟器，您将获得真实设备的传感器数据”，但事实并非如此：所有与真实传感器系统接口的代码都被注释掉了。因此，不仅 API 不兼容，而且在您希望变换测试时，必须从一种方式编辑代码到另一种方式。 （代码在处理实际未测试应用程序的边缘条件时表现也很糟糕。）

对于这种现状，我感到相当不满，决定进行修复。借助 Java 反射的力量（咳咳），我将表示方式切换为真实的 Android 对象（在模拟器未连接时有效地消除了所有开销）。幸运的是，Sensor 和 SensorEvent 是小巧的数据导向类，所以我认为我没有对内部表示造成太大影响，尽管随着未来版本的 Android SDK，代码可能会彻底崩溃。也许我应该建议上游开发人员将它们的构造函数设置为公共的。

您可以在这里获取代码：[Github 上的 SensorSimulator](https://github.com/ezyang/SensorSimulator)。如果发现错误，请告知；我只在 Froyo（Android 2.2）上进行了测试。
