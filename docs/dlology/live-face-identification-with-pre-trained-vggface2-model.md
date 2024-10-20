# 利用预训练的 VGGFace2 模型进行实时人脸识别

> 原文：<https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 11 个月前

([评论](/blog/live-face-identification-with-pre-trained-vggface2-model/#disqus_thread))

![face identification](img/94fb14491883ee7c926201d193553f22.png)

人脸识别的一个挑战是当你想在现有的列表中添加一个新的人时。你会用大量这个新人的脸和其他人的脸来重新训练你的网络吗？如果我们建立一个分类模型，模型如何对一个未知的人脸进行分类？

在这个演示中，我们通过计算两张人脸的相似度来解决这个问题，一张人脸在我们的数据库中，一张人脸图像在网络摄像头上拍摄。

VGGFace 模型将一张脸“编码”成 2048 个数字的表示。

然后，我们计算两个“编码”人脸之间的欧几里德距离。如果他们是同一个人，距离值将会很低，如果他们来自两个不同的人，距离值将会很高。

在人脸识别期间，如果该值低于阈值，我们将预测这两张照片是同一个人。

该模型本身基于 RESNET50 架构，该架构在处理图像数据方面很流行。

我们先来看看演示。

![face_identification](img/90b514a7fc2853bc903f9daf70285a65.png)

演示源代码包含两个文件。第一个文件将预先计算“编码”人脸的特征，并将结果与人名一起保存。

第二个将是现场演示，从网络摄像头捕捉图像帧，并识别是否有任何已知的面孔。

让我们开始吧。

## 预计算面特征

向模型中添加新人的一个标准方法是调用**一次性学习**。在一次性学习问题中，你必须只从一个例子中学习来再次识别这个人。

我可能有风险，因为这张照片可能光线不好，或者脸部的姿势很糟糕。因此，我的方法是从一个短视频剪辑中提取人脸，只包含这个人，并通过平均每个图像的所有计算特征来计算“平均特征”。

你可以找到我的完整源代码 [precompute_features.py](https://github.com/Tony607/Keras_face_identification_realtime/blob/master/precompute_features.py) 。但这是让奇迹发生的重要部分。

我们为每个人准备了一个或多个视频文件。`FaceExtractor`的` extract_faces`方法取一个视频文件，逐帧读取。

对于每一帧，它裁剪面部区域，然后将面部保存到图像文件中的`save_folder`。

在`extract_faces`方法中，我们调用 VGGFace 特征提取器来生成这样的人脸特征，

我们对所有人的视频都这样做。然后，我们提取这些面部图像的特征，并计算每个人的“平均面部特征”。然后将它保存到演示部分的文件中。

## 现场演示部分

因为我们已经预先计算了现场演示部分中每个人的面部特征。它只需要加载我们刚刚保存的特征文件。

提取面部，计算特征，将它们与我们预先计算的特征进行比较，看是否有匹配的。如果我们找到任何匹配的脸，我们在框架覆盖图中画出这个人的名字。

下面的方法从网络摄像头图像中的人脸计算出特征，并与我们已知的每个人脸的特征进行比较

如果这个人的面部特征离我们所有已知的面部特征“很远”，我们显示“？”在最终的图像覆盖上签名，表明这是一张未知的脸。

下面展示了一个演示，

![face identification unknown](img/af840adf355c2f88d9e36cb177bddc08.png)

## 总结和进一步阅读

我在这个演示中只包括了 3 个人。可以想象，随着人数的增加，这个模型很可能会混淆两张相似的脸。

如果发生这种情况，你可以考虑探索如 [Coursera 课程](https://www.coursera.org/learn/convolutional-neural-networks/lecture/bjhmj/siamese-network)所示的三联体缺失的暹罗网络。

FaceNet 就是一个很好的例子。

对于那些感兴趣的人。完整的源代码列在 [my GitHub repo](https://github.com/Tony607/Keras_face_identification_realtime) 中。尽情享受吧！

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/&text=Live%20Face%20Identification%20with%20pre-trained%20VGGFace2%20model) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/)

*   [←使用 Keras 从网络摄像头视频中轻松实时预测性别年龄](/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)
*   [教老狗新招——训练面部识别模型理解面部情绪→](/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/)