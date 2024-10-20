# 如何借助英特尔显卡更快地运行 TensorFlow 对象检测模型

> 原文：<https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/>

###### 发帖人:[程维](/blog/author/Chengwei/) 3 年 10 个月前

([评论](/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/#disqus_thread))

![chip](img/4394052992b6d23600ac08c6d5350832.png)

在本教程中，我将向您展示如何使用 OpenVINO toolkit 在英特尔显卡上运行定制训练的 TensorFlow 对象检测模型的推理，与 TensorFlow CPU 后端相比，速度至少快 2 倍。我的基准测试还显示，与采用 GTX1070 卡的 TensorFlow GPU 后端相比，该解决方案仅慢 22%。

如果你是 OpenVINO toolkit 新手，建议看一下[之前的教程](https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/)关于如何用 OpenVINO 转换一个 Keras 图像分类模型，加快推理速度。这一次，我们将进一步使用对象检测模型。

## 先决条件

要将 TensorFlow 冻结对象检测图形转换为 OpenVINO 中间表示(IR)文件，您需要准备好这两个文件，

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/&text=How%20to%20run%20TensorFlow%20object%20detection%20model%20faster%20with%20Intel%20Graphics) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/)

*   [←如何免费训练一个简单的物体检测模型](/blog/how-to-train-an-object-detection-model-easy-for-free/)
*   [如何在 Jupyter 笔记本中运行 tensor board→](/blog/how-to-run-tensorboard-in-jupyter-notebook/)