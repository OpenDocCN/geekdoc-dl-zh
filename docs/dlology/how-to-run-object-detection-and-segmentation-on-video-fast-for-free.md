# 如何免费快速运行视频中的对象检测和分割

> 原文：<https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零九个月前

([评论](/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/#disqus_thread))

![Mask R-CNN](img/f859a1333186ff4cd6306a58178b98fd.png)

TL；博士读完这篇文章后，你将学习如何运行艺术状态的对象检测和视频文件分割**快速**。即使是在集成显卡、旧 CPU、只有 2G 内存的旧笔记本电脑上。

这就是问题所在。这只有在你有互联网连接和谷歌 Gmail 账户的情况下才有效。既然你在读这篇文章，很有可能你已经合格了。

帖子中的所有代码都完全运行在云中。感谢谷歌的合作实验室又名谷歌实验室！

我将免费向您展示如何在具有服务器级 CPU、> 10 GB RAM 和强大 GPU 的 Colab 上运行我们的代码！是的，你没听错。

## 使用启用 GPU 的 Google Colab

Colab 的建立是为了促进机器学习专业人员更无缝地相互协作。我已经为这个帖子分享了我的 [Python 笔记本，点击打开。](https://drive.google.com/file/d/11yXcMidH2rmnvy5GxFAr0M_0mABr1M_-/view?usp=sharing)

如果您还没有登录，请登录右上角的 Google Gmail 帐户。它会让你在屏幕上方用 Colab 打开它。然后你要做一个拷贝，这样你就可以编辑它。

![colab copy](img/01b35da6f392e1dbafe2b05726883447.png)

现在你应该可以点击“**运行时**菜单按钮来选择 Python 的版本以及是否使用 GPU/CPU 来加速计算。

![runtime](img/b53ef5f1a20373bb1bab4b4b043cc0a0.png)

环境都准备好了。这么简单不是吗？没有安装 Cuda 和 cudnn 让 GPU 在本地机器上工作的麻烦。

运行此代码以确认 TensorFlow 可以看到 GPU。

它输出:

```py
Found GPU at: /device:GPU:0
```

太好了，我们可以走了！

如果你很好奇你用的是什么 GPU 型号。是一款英伟达特斯拉 K80，内存 24G。相当厉害。

运行这段代码自己找出答案。

您稍后会看到 24G 显存确实有所帮助。它可以一次处理更多的帧，以加速视频处理。

## 屏蔽 R-CNN 演示

该演示基于 [Mask R-CNN GitHub repo](https://github.com/matterport/Mask_RCNN/) 。是 [Mask R-CNN](https://arxiv.org/abs/1703.06870) 在 Keras+TensorFlow 上的实现。它不仅为检测到的对象生成边界框，而且还在对象区域上生成遮罩。

![mask-sh-expo](img/90c6aad46f975643dfce016a11249012.png)

### 安装依赖项并运行演示

在我们运行演示之前，需要安装一些依赖项。Colab 允许你通过 **pip** 安装 Python 包，通过 **apt-get** 安装一般的 Linux 包/库。

以防你还不知道。您当前的 Google Colab 实例运行在 Ubuntu 虚拟机上。您可以在 Linux 机器上运行几乎所有的 Linux 命令。

屏蔽 R-CNN 依赖于 **pycocotools** ，我们正在用下面的单元格安装。

它从 GitHub 克隆了 coco 存储库。安装生成依赖项。最后，构建并安装 coco API 库。

所有这些都发生在云虚拟机中，而且速度相当快。

我们现在准备将 Mask_RCNN repo 从 GitHub 和 cd 克隆到目录中。

请注意我们是如何使用 Python 脚本而不是运行 shell“CD”命令来更改目录的，因为我们是在当前笔记本中运行 Python 的。

现在，您应该能够像在本地机器上一样在 colab 上运行 Mask R-CNN 演示了。因此，请继续在您的 Colab 笔记本上运行它。

到目前为止，这些样本图像来自 GitHub repo。但是如何用自定义图像进行预测呢？

### [](#predict-with-custom-images)用自定义图像预测

要上传一张图片到 Colab notebook，我想到的有三个选项。

1.使用像 [imgbb](https://imgbb.com/) 这样的免费图像托管提供商。

2 。创建一个 GitHub repo，然后从 colab 下载图片链接。

通过这两个选项中的任何一个上传图像后，您将获得一个到图像的链接，可以使用 Linux **wget** 命令将其下载到您的 colab VM。它将一个图像下载到。/images 文件夹。

```py
!wget https://preview.ibb.co/cubifS/sh_expo.jpg -P ./images
```

如果你只是想上传 1 或 2 张图片，而不在乎互联网上的其他人也能看到它的链接，前两个选项将是理想的。

3.使用 Google Drive

如果您有私人图像/视频/其他文件要上传到 colab，该选项是理想的。

运行此块来验证虚拟机以连接到您的 Google Drive。

在运行期间，它将要求两个验证代码。

然后执行此单元将驱动器挂载到目录' **drive** '

您现在可以访问目录中的 Google drive 内容。/驱动

```py
!ls drive/
```

希望到目前为止你玩得开心，为什么不在一个视频文件上试试呢？

## 处理视频

处理一个视频文件需要三个步骤。

1.视频到图像帧。

2.处理图像

3.将处理后的图像转换为输出视频。

在我们之前的演示中，我们要求模型一次只处理一幅图像，就像在`IMAGES_PER_GPU`中配置的那样。

如果我们要一帧一帧地处理整个视频，那将需要很长时间。因此，我们将利用 GPU 来并行处理多个帧。

Mask R-CNN 的流水线计算量相当大，占用大量 GPU 内存。我发现 24G 内存的 Colab 上的 Tesla K80 GPU 可以安全地一次处理 3 张图像。如果超出这个范围，笔记本电脑可能会在处理视频的过程中崩溃。

所以在下面的代码中，我们将`batch_size`设置为 3，并使用**【cv2】**库在用模型处理它们之前一次暂存 3 个图像。

运行这段代码后，您现在应该将所有处理过的图像文件放在一个文件夹 `./videos/save` 中。

下一步很简单，我们需要从这些图像中生成新的视频。我们将使用 **cv2** 的**视频作者**来完成这个任务。

但是你要确定两件事:

1.这些帧需要按照从原始视频中提取的相同方式进行排序。(如果您喜欢这样看视频，也可以倒着看)

2 。帧速率匹配原始视频。您可以使用下面的代码来检查视频的帧速率，或者直接打开 file 属性。

最后，这里是从处理过的图像帧生成视频的代码。

如果您已经完成了这一步，经过处理的视频现在应该可以下载到您的本地机器上了。

![video clip](img/ac6f6541d425accc55621d952cfe59e5.png)

免费免费尝试你最喜欢的视频剪辑。可能在重建视频时有意降低帧速率，以慢动作观看。

## 总结和进一步阅读

在文章中，我们介绍了如何使用 GPU 加速在 Google Colab 上运行您的模型。

您已经学习了如何对视频进行对象检测和分割。由于 Colab 上强大的 GPU，使得并行处理多个帧以加快处理速度成为可能。

### 进一步阅读

如果您想了解更多关于对象检测和分割算法背后的技术，这里是 Mask R-CNN 的原始[论文，详细介绍了该模型。](https://arxiv.org/pdf/1703.06870.pdf)

或者，如果您刚刚开始使用异议检测，请查看我的[对象检测/定位指南系列](https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/)了解许多模型之间共享的基本知识。

这里再来  [本帖的 Python 笔记本](https://drive.google.com/file/d/11yXcMidH2rmnvy5GxFAr0M_0mABr1M_-/view?usp=sharing)和 [GitHub repo](https://github.com/Tony607/colab-mask-rcnn) 为您提供方便。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/&text=How%20to%20run%20Object%20Detection%20and%20Segmentation%20on%20a%20Video%20Fast%20for%20Free) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)

*   [←你能相信一个 Keras 模型能区分非洲象和亚洲象吗？](/blog/can-you-trust-keras-to-tell-african-from-asian-elephant/)
*   [如何处理 Keras 中的消失/爆炸渐变→](/blog/how-to-deal-with-vanishingexploding-gradients-in-keras/)