# 使用 Keras 从网络摄像头视频中轻松实时预测性别年龄

> 原文：<https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/>

###### 发布者:[程维](/blog/author/Chengwei/) 4 年 12 个月前

([评论](/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/#disqus_thread))

![crowd faces](img/4880d541d58745d0d79d786de70e68ae.png)

你曾经有过猜测另一个人年龄的情况吗？也许这个简单的神经网络模型可以为您完成这项工作。

你马上要运行的演示将从网络摄像头获取一个实时视频流，并用年龄和性别标记它找到的每张脸。猜猜看，放置一个这样的网络摄像头会有多酷，比如说在你的前门放置一个这样的摄像头，以获得所有访客的年龄/性别统计数据。

![age gender demo](img/e0de70083d1425abdddb3cb8711c8f27.png)

我在装有 Python 3.5 的 Windows PC 上运行了这个模型。也可以在其他操作系统上运行。

# 它是如何工作的

让我们大致了解一下它是如何工作的。

![pipeline](img/a97bbbdeb77349df5642b18e918622d0.png)

首先，照片是由`**cv2**`模块从网络摄像头实时拍摄的。

其次，我们将图像转换为灰度，并使用`cv2`模块的`CascadeClassifier`类来检测图像中的人脸

由`detectMultiScale`方法返回的变量 faces 是一个检测到的面部坐标列表[x，y，w，h]。

在知道了人脸的坐标之后，我们需要在输入神经网络模型之前裁剪这些人脸。

我们在面部区域增加了 40%的余量，这样整个头部都包括在内了。

然后，我们准备将这些裁剪过的面输入到模型中，这就像调用`predict`方法一样简单。

对于年龄预测，模型的输出是与年龄概率范围从 0~100 相关联的 101 个值的列表，所有 101 个值加起来是 1(或者我们称之为 softmax)。因此，我们将每个值与相关的年龄相乘，并将它们相加，得到最终的预测年龄。

最后但同样重要的是，我们绘制结果并渲染图像。

性别预测是一项二元分类任务。该模型输出 0~1 之间的值，其中该值越高，该模型认为该人脸是男性的置信度越高。

我的完整源代码以及下载预训练模型权重的链接可在 [my GitHub repo](https://github.com/Tony607/Keras_age_gender) 中找到。

# 更深入

对于那些不满意的演示，并有更多的了解模型是如何建立和培训。这部分是给你的。

数据集来自[IMD b-WIKI–500k+带有年龄和性别标签的人脸图像](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)。在把每张图片输入模型之前，我们做了和上面一样的预处理步骤，检测人脸并添加边缘。

神经网络的特征提取部分使用 WideResNet 架构，是宽残差网络的缩写。它利用卷积神经网络(简称 ConvNets)的能力来学习人脸的特征。从不太抽象的特征如棱角到更抽象的特征如眼睛和嘴巴。

WideResNet 架构的独特之处在于作者降低了原始残差网络的深度并增加了其宽度，因此其训练速度提高了数倍。链接到[论文](https://arxiv.org/abs/1605.07146)这里。

# 进一步阅读

模型的可能性是无限的，这真的取决于你把什么数据输入其中。假设你有很多照片被贴上了吸引力的标签，你可以教模型从网络摄像头的直播流中辨别一个人的性感程度。

这是一个相关项目的列表，为那些好奇的人准备的数据集。

[tensor flow 中的年龄/性别检测](https://github.com/dpressel/rude-carnie)

[IMD b-WIKI–50 万张以上带有年龄和性别标签的人脸图像](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

数据:[性别和年龄分类未过滤的人脸](https://www.openu.ac.il/home/hassner/Adience/data.html#agegender)
Github:[keras-vgg face](https://github.com/rcmalli/keras-vggface)

[自拍:一种理解自拍之美的方法](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/)

结束

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/&text=Easy%20Real%20time%20gender%20age%20prediction%20from%20webcam%20video%20with%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)

*   [←如何教会 AI 向在线卖家建议产品价格](/blog/how-to-teach-ai-to-suggest-product-prices-to-online-sellers/)
*   [用预训练的 VGGFace2 模型进行活体人脸识别→](/blog/live-face-identification-with-pre-trained-vggface2-model/)