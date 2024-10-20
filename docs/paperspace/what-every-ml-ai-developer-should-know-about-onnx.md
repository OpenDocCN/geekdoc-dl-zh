# 关于 ONNX，每个 ML/AI 开发者都应该知道的

> 原文：<https://blog.paperspace.com/what-every-ml-ai-developer-should-know-about-onnx/>

## [开放神经网络交换格式(ONNYX)](https://medium.com/r/?url=https%3A%2F%2Fonnx.ai%2F) 是交换深度学习模型的新标准。它承诺使深度学习模型可移植，从而防止供应商锁定。让我们看看为什么这对现代 ML/AI 开发人员很重要。

* * *

The end result of a trained deep learning algorithm is a model file that efficiently represents the relationship between input data and output predictions. A neural network is one of the most powerful ways to generate these predictive models but can be difficult to build in to production systems. Most often, these models exist in a data format such as a `.pth` file or an HD5 file. Oftentimes you want these models to be portable so that you can deploy them in environments that might be different than where you initially trained the model.

### ONNX 概述

在高层次上，ONNX 被设计成允许框架互操作性。有许多各种语言的优秀机器学习库——py torch、TensorFlow、MXNet 和 Caffe 只是近年来非常受欢迎的几个，但还有许多其他库。
这个想法是你可以用一个工具栈训练一个模型，然后用另一个工具栈部署它进行推理和预测。为了确保这种互操作性，您必须以`model.onnx`格式导出您的模型，这是 protobuf 文件中模型的序列化表示。目前 ONNX 中有对 PyTorch、CNTK、MXNet 和 Caffe2 的原生支持，但也有针对 TensorFlow 和 CoreML 的转换器。

### ONNX 在实践中

让我们假设您想要训练一个模型来预测您冰箱中的一种食品是否仍然可以食用。你决定运行一组超过保质期的不同阶段的食物照片，并将其传递给卷积神经网络(CNN)，该网络查看食物的图像并训练它预测食物是否仍然可以食用。

一旦你训练好了你的模型，你就想把它部署到一个新的 iOS 应用程序上，这样任何人都可以使用你预先训练好的模型来检查他们自己的食物的安全性。你最初使用 PyTorch 训练你的模型，但是 iOS 期望使用用于应用程序内部的 CoreML。ONNX 是模型的中间表示，让您可以轻松地从一个环境转移到下一个环境。

使用 PyTorch 你通常会使用`torch.save(the_model.state_dict(), PATH)`导出你的模型，导出到 ONNX 交换格式只是多了一行:
`torch.onnx.export(model, dummy_input, 'SplitModel.proto', verbose=True)`

使用 ONNX-CoreML 这样的工具，您现在可以轻松地将预训练的模型转换为文件，然后导入 XCode 并与您的应用程序无缝集成。作为一个工作示例，请查看 Stefano Attardi 关于从头到尾构建 ML 驱动的 iOS 应用程序的这篇[精彩文章](https://medium.com/r/?url=https%3A%2F%2Fattardi.org%2Fpytorch-and-coreml)。

### 结论

随着越来越多的深度学习框架出现，工作流变得更加先进，对可移植性的需求比以往任何时候都更加重要。ONNX 是一个强大的开放标准，用于防止框架锁定，并确保您开发的模型在长期内是可用的。