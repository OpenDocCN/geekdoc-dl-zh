# 基于深度可分卷积的简单语音关键词检测

> 原文：<https://www.dlology.com/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零六个月前

([评论](/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/#disqus_thread))

![voice-command](img/fcc0c87b96bcd2d4c755071876ef9059.png)

关键字检测或语音命令可以被视为语音识别系统的最小版本。如果我们能够制造出精确的模型，同时消耗足够小的内存和计算空间，甚至可以在裸机(没有操作系统)的微控制器上实时运行，那会怎么样？如果这成为现实，想象一下哪些传统的消费电子设备会因为启用了永远在线语音命令而变得更加智能。

在这篇文章中，我们将采取第一步来建立和训练这样一个深度学习模型，在有限的内存和计算资源的情况下进行关键词检测。

## 关键词检测系统

与通常基于云并且可以识别几乎任何口语单词的完整语音识别系统相比，另一方面，关键字检测可以检测预定义的关键字，如“Alexa”、“Ok Google”、“嘿 Siri”等。哪个是“一直开”。关键词的检测触发了特定的动作，例如激活全面语音识别系统。在一些其他的用例中，这样的关键字可以用于激活启用语音的灯泡。

关键词检测系统由两个基本部分组成。

1.  特征提取器，用于将音频剪辑从时域波形转换为频域语音特征。
2.  基于神经网络的分类器，用于处理频域特征并预测所有预定义关键字加上“未知”单词和“无声”的可能性。

![keyword-pipline](img/9aa810d6115acb36b926b350687fffdc.png)

我们的系统采用[梅尔倒谱系数](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) 或 MFCCs 作为特征提取器来获取音频的 2D‘指纹’。由于神经网络的输入是像 2D 音频指纹这样的图像，横轴表示时间，纵轴表示频率系数，因此选择基于卷积的模型似乎是一个自然的选择。

## 深度可分卷积

标准卷积运算的问题可能仍然需要太多来自微控制器的存储器和计算资源，考虑到甚至一些最高性能的微控制器只有大约 320KB 的 SRAM 和大约 1MB 的闪存。满足约束条件同时仍然保持高精度的一种方法是应用深度方向可分离卷积来代替传统的卷积神经网络。

它首先在 Xception ImageNet 模型中引入，然后被 MobileNet 和 ShuffleNet 等其他一些模型采用，旨在降低模型复杂性，以部署在智能手机、无人机和机器人等资源受限的目标上。

深度方向可分离卷积神经网络在于首先执行**深度方向空间卷积**，其分别作用于每个输入通道，随后是**点方向卷积**(即，1x1 卷积)，其混合所得的输出通道。直观上，可分卷积可以理解为将卷积核分解成两个更小的核的一种方式。

一个标准的卷积运算一步就把输入过滤合并成一组新的输出。与传统的卷积运算相比，深度方向可分离卷积将其分为两层，一层用于滤波，另一层用于合并。这种因子分解具有显著减少计算和模型大小的效果。 深度方向可分卷积在参数数量和运算上都更高效，这使得更深更宽的架构甚至在资源受限的器件中也成为可能。 

![ds_cnn](img/75e88fc002ec9ae46bd8e3baf358b73b.png)

下一节我们将通过 TensorFlow 用深度方向可分离的 CNN 架构实现该模型。 

## 构建模型

第一步是将原始音频波形转换为 MFCC 特征，可以像这样在 TensorFlow 中完成。

如果输入音频和特征提取器有以下参数，

*   输入音频采样率:16000 赫兹
*   输入音频剪辑长度:1000 毫秒(长)
*   谱图窗口大小:40 毫秒(长)
*   谱图窗口步幅:20 毫秒
*   MFCC 系数计数:10 (F)

那么张量`self.mfcc_`的 形状将为(None，T，F)，其中帧数:T = (L-l) / s +1 = (1000 - 40) / 20 + 1 = 49。`self.mfcc_`然后成为深度学习模型的`fingerprint_input`。

基于 MobileNet 的实现，我们采用了一个深度方向可分离的 CNN，完整的实现可以在我的 [GitHub](https://github.com/Tony607/Keyword-detection) 上获得。

在最后使用一个全连接层之后的平均池来提供全局交互并减少最终层中的参数总数。

## 模型表现如何？

预先训练好的模型已经准备好供您使用，包括标准的 CNN、DS_CNN(深度方向可分离卷积)和各种其他模型架构。对于每个体系结构，将搜索各种超参数，如内核大小/步幅，并分别训练不同规模的模型，以便您可以权衡更小、更快的模型，以在精度稍低的资源受限设备上运行。

![](img/f9f61fa9c5bb0f9f819ce53c3a9c01aa.png)

用深度方向可分离卷积构建的模型比具有相似运算次数的 DNN 模型获得了更好的精度，但是对内存的需求却降低了 10 倍。

注意，表中显示的所需内存是在将浮点权重量化为 8 位定点后的，我将在以后的帖子中解释这一点。

通过已训练的 DS_CNN 模型运行音频文件，并获得顶部预测，

## 结论和进一步阅读

在这篇文章中，我们探讨了如何实现一个简单而强大的关键字检测模型，该模型有可能在微控制器等资源受限的设备上运行。

一些你可能会觉得有用的相关资源。

1.  TensorFlow 教程- [简单音频识别](https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
2.  GitHub 中的 TensorFlow [语音命令示例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)代码
3.  博客- [微控制器关键词识别](https://community.arm.com/processors/b/blog/posts/high-accuracy-keyword-spotting-on-cortex-m-processors)
4.  Keras 中的深度方向可分离卷积神经网络 [可分离卷积 2D](https://keras.io/layers/convolutional/#separableconv2d)
5.  Paper - [例外:深度可分卷积深度学习](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
6.  论文- [MobileNets:用于移动视觉应用的高效卷积神经网络](https://arxiv.org/pdf/1704.04861.pdf)

在以后的帖子中，我将解释如何应用模型权重量化过程来减小模型大小，并向您展示如何在微控制器上运行模型。

查看我的 [GitHub repo](https://github.com/Tony607/Keyword-detection) 和更多信息，包括培训和测试模型。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/&text=Simple%20Speech%20Keyword%20Detecting%20with%20Depthwise%20Separable%20Convolutions) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/)

*   [←使用 TensorFlow Hub DELF 模块轻松识别地标图像](/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/)
*   [如何用 CMSIS-NN 在微控制器上运行深度学习模型(第一部分)→](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/)