<!--yml

类别：未分类

date: 2024-07-01 18:16:53

-->

# [Surface Book 2](https://blog.ezyang.com/2019/03/microsoft-surface-book-2/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2019/03/microsoft-surface-book-2/`](http://blog.ezyang.com/2019/03/microsoft-surface-book-2/)

## Surface Book 2

长期阅读我的人可能知道，过去十年我一直在使用 ThinkPad X61T。在我的第二台机器的铰链出现问题后，我决定终于是时候换一台新的笔记本了。我对一款特定的型号情有独钟，这源自上次 Haskell 实现者工作坊上 Simon Peyton Jones 向我展示他的新笔记本：Microsoft Surface Book 2。它符合我对笔记本的主要要求：它是可转换为平板模式的笔记本，并配备数字笔。这支笔虽然不是 Wacom 品牌的，但有一个橡皮擦端，并可以磁性吸附在笔记本上（没有笔的外壳，但我认为对于现代硬件来说，这种限制是无法满足的）。此外，还有一个[Linux 爱好者社区](https://github.com/jakeday/linux-surface/)关注这款设备，这让我觉得我更有可能让 Linux 正常工作。所以几周前，我决定尝试一下，花了三千美元买了一台自己的 Surface Book 2。事实证明效果不错，但典型的 Linux 风格，没有少费一点功夫。

### 快速评估

The good:

1.  我已经成功使所有“重要”功能正常工作。这包括 Xournal 与 XInput 笔以及休眠功能（尽管存在一些注意事项）。

1.  对于其他随机功能的 Linux 支持令我惊喜不小：我成功安装了 CUDA 和驱动（用于 PyTorch 开发），能够裸机启动我的 Linux 分区以及在 Windows 的 VM 中启动，甚至可以在进入 Linux 时拆卸屏幕。

1.  键盘很好用；虽然不及经典 Thinkpad 键盘那么好，但它有真正的功能键（不像我在工作中使用的 Macbook Pro）。

1.  两个标准 USB 端口和一个 USB-C 端口意味着我在大多数情况下不需要转接头（不像我的 Macbook Pro，它只有 USB-C 端口）。

The bad:

1.  （更新于 2019 年 3 月 19 日）暂停功能非常慢。尽管 jakeday 的 setup.sh 建议暂停功能不起作用，*某种*功能正在运行，如果我关闭笔记本盖子，笔记本会进入某种低功耗状态。但进入暂停状态的时间相当长，重新启动时间更长，你仍然需要点击过引导加载程序（这让我严重怀疑我们是否真的在暂停）。

1.  当我把它放进背包时，笔记本有时会自动解除休眠状态。我目前的假设是电源按钮被按下了（不像大多数笔记本电脑，电源按钮位于屏幕顶部且无保护）。也许通过一些 ACPI 设置的调整可能会有所帮助，但我还没有仔细研究过。

1.  这是一个高 DPI 屏幕。本质上这并没有问题（而且现在基本上买不到非高 DPI 的笔记本电脑了），但是任何不懂如何处理高 DPI 的软件（VMWare 和 Xournal，我在看你们）看起来都很糟糕。然而，Ubuntu Unity 对高 DPI 的支持自从我上次尝试以来已经好多了；如果我只使用终端和浏览器，一切看起来都还可以。

1.  功能键被硬编码为切换 fn 锁定。这有点让人恼火，因为你必须记住它处于哪个设置，决定是否按住它来获取另一个切换。我也感到失去了专门的页面向上/向下键有些遗憾。

1.  显然，NVIDIA GPU 由于热传感器问题自行降频（大概是因为风扇在主板上而不是 GPU 上，所以驱动程序认为风扇坏了并且进行了降频？模模糊糊的）。

1.  扬声器…还行。不是很好，只是还行。

1.  微软选择了一些定制的 Surface Book 2 充电器，有点可惜。

### Linux 设置

我进行了最新的 Ubuntu LTS（18.04）的标准安装，与 Windows 双启动（1TB 硬盘真心好用！），然后安装了 jakeday 的 [自定义 Linux 内核和驱动程序。](https://github.com/jakeday/linux-surface) 关于这个过程的一些说明：

+   我花了一段时间琢磨为什么我不能安装 Linux 双启动。一些搜索建议问题是因为 Windows 没有真正关机；它只是休眠了（为了快速启动）。我没能禁用这个功能，所以我只是在 Windows 内部调整了分区大小，然后在那个分区上安装了 Linux。

+   不要忘记为 Linux 分配一个专用的交换分区；如果没有它，你将无法休眠。

+   Surface Book 2 启用了安全启动。你必须按照 [SIGNING.md](https://github.com/jakeday/linux-surface/blob/master/SIGNING.md) 中的说明来获取已签名的内核。

+   生成带有签名的内核的一个后果是，如果你同时安装了未签名和签名的内核，`update-initramfs -u` 将为你的 *未签名* 内核更新 initrd，这意味着你不会看到你的更改，除非你复制 initrd！这让我对下一步感到非常困惑…

+   如果你想为你闪亮的 NVIDIA GPU 使用 NVIDIA 驱动程序，你需要拉黑 nouveau。网上有很多指导，但我可以亲自保证 [remingtonlang 的指导](https://github.com/jakeday/linux-surface/issues/264#issuecomment-427452156)。确保你在更新正确的 initrd；参见我上面的要点。修复了这个问题后，CUDA 安装程序的标准调用让我开始运行 `nvidia-smi`。请注意，我手动使用了 [这里的指南](https://askubuntu.com/questions/1023036/how-to-install-nvidia-driver-with-secure-boot-enabled)，因为我已经生成了一个私钥，所以看起来很蠢再生成另一个，因为 NVIDIA 的安装程序要求我这样做。

+   安装 NVIDIA 驱动程序后，你必须小心相反的问题：Xorg 决定在 NVIDIA 卡上进行所有渲染！当这种情况发生时的常见症状是 Linux 的鼠标输入非常卡顿。如果你有工作的 `nvidia-smi`，你也可以看到 Xorg 在你的 GPU 上作为一个正在运行的进程。无论如何，这是不好的：你不想使用独立显卡来进行普通的桌面渲染；你需要集成的。我发现取消注释 `/etc/X11/xorg.conf.d` 中的 Intel 配置示例可以解决这个问题：

    ```
    Section "Device"
        Identifier  "Intel Graphics"
        Driver      "intel"
    EndSection

    ```

    但这在 VMWare 上并不太友好；后面会详细讨论。

+   声音在我升级到 Linux 5.0.1 之前无法正常工作（声音太小，或者右侧扬声器不工作）。

+   在[我的 Xournal 分支](https://github.com/ezyang/xournal)上启用 XInput 后，它在我重新启动 Xournal 后才开始起作用。橡皮擦在开箱即用时就可以使用。

+   别忘了创建交换分区（Ubuntu 默认安装程序没有提示我创建交换分区，可能是因为我是作为双引导安装的）；否则，休眠功能将无法使用。

+   有时候，从休眠状态唤醒后，网络无法正常工作。幸运的是，可以通过手动重新加载 WiFi 内核模块来解决这个问题：`modprobe mwifiex_pcie` 和 `systemctl restart NetworkManager.service`。关于[此问题的更多讨论。](https://github.com/jakeday/linux-surface/issues/431)

+   有时候，从休眠/挂起状态唤醒时，我会看到一个大的温度计图标。重新启动后，它会消失，但我的休眠/挂起功能却消失了。令人困惑！我不知道为什么会这样发生。

### 通过虚拟机引导

生活的悲哀之处在于，Windows 平板的体验要比 Linux 的体验好得多——以至于许多人只会安装 Windows，然后从虚拟机（或 Windows Subsystem for Linux）中引导 Linux。对我来说这是行不通的：必须从裸金属引导 Linux 才能获得最佳的笔输入体验。不过，为什么不也让从运行在 Windows 上的 VMWare 引导 Linux 分区成为可能呢？这种设置是被[VMWare 明确支持的](https://www.vmware.com/support/ws5/doc/disks_dualmult_ws.html)，但实际上需要几天的折腾才能让它真正起作用。

+   首先，你需要 VMWare Workstation Pro 来配置一个能够访问原始磁盘的虚拟机（尽管生成的虚拟机映像可以在免费的 VMWare Player 中运行）。你可以注册获取 30 天的试用期来配置它，然后从此使用 Player，如果你喜欢的话。在设置磁盘时，VMWare 将提供原始磁盘作为选项；选择它并选择你的机器上的 Linux 分区。

+   设置这个系统的主要挑战在于，Surface Book 2 上的标准 Linux 安装没有传统的 Linux 引导分区；相反，它有一个 EFI 分区。最显著的是，这个分区在 Windows 启动时被永久挂载，因此你无法重新挂载它用于你的虚拟机。你的常规分区没有引导加载程序，这就是为什么当你启动虚拟机时，你被强制进入通过 PXE 进行网络启动的原因。我最终采取的解决方法是制作一个新的虚拟磁盘（支持 vmdk 格式），并在上面安装引导分区（你实际上不需要任何内核或 initrd，因为它们存在于你的根文件系统中；只有 `/boot/efi` 是从 EFI 分区挂载的）。当然，你必须实际设置这个引导分区；我做的方法是在救援 CD 中 chroot 到我的分区，然后运行 `grub-install /dev/sda1`。在折腾过程中，我还不小心运行了 `update-grub`，导致我的 Windows 引导选项消失了，但在裸机 Linux 启动时重新运行此命令修复了问题（因为真正的 `/boot/efi` 将被挂载，因此 Grub 将找到 Windows 引导选项）。

+   一些关于双启动的文档是针对 VMWare Fusion 的。这是特定于 OS X 的，因此与微软 Surface Book 2 无关。

+   获取一个可启动的 Linux CD（我个人使用[SystemRescueCd](http://www.system-rescue-cd.org/)）来帮助调试安装过程中的问题。

+   确保你的 `/etc/fstab` 中的所有条目对应于真实的磁盘，否则你的 Ubuntu 启动过程将花费一段时间等待永远不会显示的磁盘。我在 `/boot/efi` 挂载上遇到了这个问题，因为挂载是基于 UUID 的；我通过将挂载改为基于 LABEL 并相应地为我的 vmdk 标记标签来“修复”它（我想也可以尝试更改我的 vmdk 的 UUID，但我找不到在 Windows 上合理的操作说明）。注意，卷实际上不必成功挂载（我的没有，因为我忘记将其格式化为 vfat）；它只需存在，以便系统不会等待查看它是否在稍后的时间点连接。

+   我真的不明白 Unity 如何决定提供缩放选项，但尽管在裸机启动时提供放大选项，但在虚拟机下运行时却不可用。通过将分辨率设置为 1680 x 1050，我得到了一个相对合适大小的显示（只有轻微模糊）。在 VMWare 中我启用了“拉伸模式”。

+   你能否登录你的账户取决于你的 X11 配置；如果你像我一样取消了 Intel 的配置，我发现这会导致我的登录失败（你可以通过再次注释掉来解决）。如何使两者都正常工作？别问我，我还在摸索中。

### 窗口管理器

我还没有开始设置 xmonad；这在很大程度上是因为 Unity 似乎只支持一种非常基础的平铺方式：Windows-left 和 Windows-right 会将窗口移动到显示器的左侧或右侧的一半，而 Windows-up 则会使窗口全屏。也许我还会尝试在 18.04 上设置 xmonad，但现在不用与 trayer 为标准图标而斗争的感觉很好。

### 接下来做什么

我改善 Surface Book 2 上 Linux 状态的两个主要优先事项：

1.  重写 Xournal 以支持 hDPI（这有多难呢，哈哈）

1.  弄清楚如何让挂起/休眠更可靠

除此之外，我对这台新笔记本非常满意。特别是我的邮件客户端（仍然是 sup）运行得快得多；以前搜索新邮件会很慢，但在这台笔记本上它们像闪电般流入。这只是说明从 1.6GHz 处理器升级到 4.2GHz 处理器有多大的提升啊 :3
