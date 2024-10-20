# 神经网络的对抗性攻击:探索快速梯度符号法

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/adversarial-attacks-on-neural-networks-exploring-the-fast-gradient-sign-method>

自从他们发明以来，[神经网络](https://web.archive.org/web/20221207124346/https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414)一直是机器学习算法的精华。他们推动了人工智能领域的大部分突破。

神经网络已被证明在执行高度复杂的任务方面是稳健的，即使人类也发现这些任务非常具有挑战性。

它们不可思议的健壮性能超越最初的目的吗？这就是我们在这篇文章中试图找到的。

就我个人而言，我从未想到会与 AI 相交的一个领域是**安全。**事实证明，这是神经网络失败的少数领域之一。

我们将尝试一种非常流行的攻击方法，即[快速梯度符号方法](https://web.archive.org/web/20221207124346/https://arxiv.org/abs/1412.6572)，来演示神经网络的安全漏洞。但首先，让我们来探讨不同类别的攻击。

敌对攻击

## 根据您作为攻击者对您想要欺骗的模型的了解，有几种类型的攻击。最流行的两种是**白盒攻击**和**黑盒攻击。**

这两种攻击类别的总体目标都是欺骗神经网络做出错误的预测。他们通过在网络的输入端添加难以察觉的噪声来做到这一点。

这两种攻击的不同之处在于您能够访问模型的整个架构。使用白盒攻击，您可以完全访问架构(权重)，以及模型的输入和输出。

黑盒攻击对模型的控制较低，因为您只能访问模型的输入和输出。

在执行这些攻击时，您可能会想到一些目标:

**错误分类，**您只想让模型做出错误的预测，而不担心预测的类别，

*   **源/目标错误分类，**您的目的是向图像添加噪声，以推动模型预测特定类别。
*   快速梯度符号方法(FGSM)结合了白盒方法和误分类目标。它欺骗神经网络模型做出错误的预测。

让我们看看 FGSM 是如何工作的。

快速梯度符号法解释

这个名字看起来很难理解，但是 FGSM 攻击非常简单。它依次包括三个步骤:

## 计算正向传播后的损耗，

计算相对于图像像素的梯度，

1.  在计算出的梯度方向上稍微推动图像的像素，使上面计算出的损失最大化。
2.  第一步，计算前向传播后的损失，在机器学习项目中很常见。我们使用负似然损失函数来估计我们的模型的预测与实际类别的接近程度。
3.  **不常见的，是** **计算的渐变与** **对图像像素的尊重**。当涉及到训练神经网络时，梯度是你如何确定将你的权重微移至*减少*损失值的方向。

代替这样做，这里我们在梯度的方向上调整输入图像像素以使损失值最大化。

当训练神经网络时，确定调整网络深处的特定权重(即，损失函数相对于该特定权重的梯度)的方向的最流行的方法是将梯度从起点(输出部分)反向传播到权重。

同样的概念也适用于此。我们将梯度从输出层反向传播到输入图像。

在神经网络训练中，为了微调权重以降低损失值，我们使用这个简单的等式:

***new _ weights = old _ weights–learning _ rate * gradients***

同样，我们将相同的概念应用于 FGSM，但我们希望最大化损失，因此我们根据下面的等式微调图像的像素值:

***新像素=旧像素+ε*渐变***

在上面的图像中，我们看到两个箭头代表两种不同的方式来调整梯度，以达到一个目标。你可能已经猜对了，左手边的方程是训练神经网络的基本方程。自然，计算出的梯度指向损耗最大的方向。**神经网络训练方程**中的负号确保梯度指向相反的方向——使损失最小化的方向。右手边的方程不是这种情况，它是愚弄神经网络的方程。因为我们想使损失最大化，所以可以说，我们应用了自然形式的梯度。

这两个方程有许多不同之处。最重要的是加减法。通过使用等式 2 的结构，我们在与最小化损失的方向相反的方向上推动像素。通过这样做，我们告诉我们的模型只做一件事——做出错误的预测！

在上图中， **x** 代表我们希望模型错误预测的输入图像。图像的第二部分表示损失函数相对于输入图像的梯度。

请记住，梯度只是一个方向张量(它给出了关于向哪个方向移动的信息)。为了加强轻推效果，我们用一个非常小的值ε(图中为 0.007)乘以梯度。然后我们把结果加到输入图像上，就这样！

由此产生的图像下的令人担忧的表达可以简单地这样表达:

***输入 _ 图像 _ 像素+ε*损失函数相对于输入 _ 图像 _ 像素的梯度***

至此，让我们总结一下到目前为止所讲的内容，接下来我们将进行一些编码。为了欺骗神经网络做出错误的预测，我们:

通过我们的神经网络向前传播我们的图像，

计算损失，

*   将梯度反向传播到图像，
*   向最大化损失值的方向微移图像的像素。
*   通过这样做，我们告诉神经网络对远离正确类别的图像进行预测。
*   我们需要注意的一点是，噪声在最终图像上的显著程度取决于**ε–**值越大，噪声越明显。

增加ε也增加了网络做出错误预测的可能性。

代码

在本教程中，我们将使用 TensorFlow 来构建我们的整个管道。我们将把重点放在代码最重要的部分，而不是与数据处理相关的部分。
首先，我们将加载 TensorFlow 最先进的模型之一， **mobileNet V2 模型:**

将模型的可训练属性设置为 *false* 意味着我们的模型不能被训练，因此任何改变模型参数以欺骗模型的努力都将失败。

## 我们可能希望将图像可视化，以了解不同的**ε**值如何影响预测以及图像的性质。简单的代码片段就能解决这个问题。

接下来，我们加载我们的图像，通过我们的模型运行它，并获得相对于图像的损失梯度。

```py
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False
```

打印 signed_grad 显示 1 的张量。一些带有正号，另一些带有负号，这表明梯度仅在图像上实施方向效果。绘制它揭示了下面的图像。

现在我们有了渐变，我们可以在与渐变方向相反的方向上轻推图像像素。

```py
def display_images(image, info):
  _, label, prob = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} n {} : {:.2f}% prob.format(info,label,prob*100))
  plt.show() 
```

换句话说，我们在最大化损失的方向上轻推图像像素。我们将对不同的ε值运行攻击，其中**ε= 0**表示没有运行攻击。

```py
loss_object = tf.keras.losses.Categorical_Crossentropy()
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)

    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

   gradient = tape.gradient(loss, input_image)

   signed_grad = tf.sign(gradient)
   return signed_grad
```

我们得到以下结果:

注意到上面三幅图中的图案了吗？随着ε值的增加，噪声变得更加明显，并且错误预测的可信度增加。

我们成功地愚弄了一个最先进的模型，让它做出了错误的预测，而没有改变这个模型的任何东西。

```py
epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations 
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])

```

让我们在这里做一个小实验来证实我们上面讨论的一个概念。我们不是将ε和梯度的张量乘法结果(image + eps * signed_grad)加到图像上，而是将图像的像素向损失最大化的方向推，而是执行减法，将图像的像素向损失最小化的方向推(image–EPS * signed _ grad)。

结果是:

通过在使损失最小化的梯度方向上轻推图像像素，我们增加了模型做出正确预测的信心，这比没有这个实验要多得多。从 41.82%的信心到 97.89%。

后续步骤

```py
epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
for i, eps in enumerate(epsilons):
  adv_x = image - eps*perturbations 
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])
```

自从 FGSM 发明以来，又创造了其他几种攻击角度不同的方法。你可以在这里查看这些攻击:[攻击调查](https://web.archive.org/web/20221207124346/https://arxiv.org/abs/1810.00069)。

你也可以尝试不同的模型和不同的图像。你也可以从头开始建立自己的模型，也可以尝试不同的ε值。

就这些了，谢谢你的阅读！

## Next steps

Since the invention of the FGSM, several other methods were created with different angles of attacks. You can check out some of these attacks here: [Survey on Attacks](https://web.archive.org/web/20221207124346/https://arxiv.org/abs/1810.00069). 

You can also try out different models and a different image. You may also build your own model from scratch and also try out different values of epsilon.

That’s it for now, thank you for reading!