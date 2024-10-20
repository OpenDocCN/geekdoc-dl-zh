# 如何在 TensorFlow / Keras 中构建轻量级图像分类器

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-build-a-light-weight-image-classifier-in-tensorflow-keras>

计算机视觉是一个快速发展的领域，正在取得巨大的进步，但仍然有许多挑战需要计算机视觉工程师来解决。

首先，他们的终端模型需要健壮和准确。

其次，最终的解决方案应该足够快，并且在理想情况下，达到接近实时的性能。

最后，模型应该占用尽可能少的计算和内存资源。

幸运的是，有许多最先进的算法可供选择。有些在精确度方面是最好的，有些是最快的，有些是难以置信的紧凑。军火库确实很丰富，计算机视觉工程师有很多选项可以考虑。

## 计算机视觉解决的任务

### 图像分类

![Image Classifier](img/af0dad6b52173e4fd4493c218ffac563.png)

*[Source](https://web.archive.org/web/20221201161302/https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/): 9 Applications of Deep Learning for Computer Vision by Jason Brownlee*

在本文中，我们将着重于创建图像分类器。图像分类是一项基本任务，但仍然是计算机视觉工程师能够处理的最重要的任务之一。

对图像进行分类意味着确定类别。类别的数量受限于您想要区分的图像类型的数量。例如，您可能希望根据车辆类型对图像进行分类。类别的可能选项有:自行车、汽车、公共汽车和卡车。

或者，您可能想要一个更详细的集合，并将高级类分解成低级子类。用这种方法，你的类列表可能包括运动自行车，直升机，踏板车，和全地形自行车。汽车类别将进一步分为掀背车，皮卡，紧凑型车，运动跑车，跨界车和面包车。对于公共汽车和卡车也可以进行类似的分解。

最终的类别集由计算机视觉工程师确定，该工程师非常了解问题领域，并且熟悉数据集和可用的注释。

### 目标检测

如果您处理的图像有多个关联的类，该怎么办？继续我们之前的例子，一张图片可能包含一辆公共汽车和一辆摩托车。你肯定不想错过任何一个对象，想两个都抓住。这里，对象检测开始发挥作用。

在计算机视觉中，检测一个对象意味着定位它并给它分配一个类别。简单来说，我们希望在图像上找到一个对象并识别它。如您所见，对象检测包含图像分类部分，因为我们在对象被定位后进行分类。它强调了一个事实，即图像分类是计算机视觉的核心，因此需要认真学习。

### 语义分割

![Semantic segmentation](img/3000a559687570ade55af892c9e91a18.png)

*[Source](https://web.archive.org/web/20221201161302/https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/): Computer Vision Tutorial: A Step-by-Step Introduction to Image Segmentation Techniques (Part 1) by PULKIT SHARMA*

最后，让我们简单讨论一下语义分割，对图像的每个像素进行分类。如果一个像素属于一个特定的对象，那么这个像素对于这个对象被分类为阳性。通过将该对象的每个像素分类为阳性来分割该对象。这再次强调了分类的重要性。

当然，计算机视觉中还有很多其他的任务我们今天不会去碰，但是相信我，图像分类是最核心的一个。

## 什么使得图像分类成为可能

在我们继续学习如何创建轻量级分类器的实用指南之前，我决定暂时停下来，回到问题的理论部分。

这个决定是经过深思熟虑的，来自我的日常观察，越来越多的计算机视觉工程师倾向于在非常高的水平上处理问题。通常，这意味着一个简单的预训练模型导入，将准确性设置为一个度量，并启动训练作业。由于在创建最常见的机器学习框架(scikit-learn、PyTorch 或 Tensorflow)方面所做的大量工作，如今这已经成为可能。

这些框架中的这些进步是否意味着没有必要深究算法背后的数学？绝对不行！了解使图像分类成为可能的基础知识是一种超能力，你可以，而且肯定应该得到。

当一切顺利时，您可能会觉得没有必要超越基本的模型导入。但是，每当你面临困难的时候，理解算法就会很好地为你服务。

也不需要成为策划人。对于深度学习，你需要掌握的只是卷积神经网络的概念。一旦你学会了这一点，你就可以解决在训练机器学习模型时可能遇到的任何问题。相信我，如果你在机器学习的职业道路上，会有很多这样的情况。

***注*** *:在这篇文章里我就不细说 CNN 了，我想说明一下其他的事情。但是，我会分享一些我在网上找到的资源，我认为它们对理解 CNN 非常有用。您可以在参考资料部分找到它们。*

## 最佳卷积神经网络综述

![Convolutional Neural Network](img/1aac57e362a5d10e930853c1b815d5e4.png)

*[Source](https://web.archive.org/web/20221201161302/https://developersbreach.com/convolution-neural-network-deep-learning/): Convolutional Neural Network | Deep Learning by Swapna K E*

当 CNN 的概念清晰时，你可能会想知道现在哪种 CNN 表现最好。这就是我们在这一节要讨论的内容。

尽管第一个 CNN 是在 20 世纪 80 年代推出的，但真正的突破发生在 21 世纪初，图形处理单元(GPU)的出现。

为了了解进展有多快，有必要看一下在一年一度的名为 ImageNet 大规模视觉识别挑战赛(ILSVRC)的比赛中取得的错误率统计数据。以下是错误率的历史变化情况:

![Error rate history](img/08b5d00fe4362cd21be16c911256fd5e.png)

*Error rate history on ImageNet (showing best results per team and up to 10 entries per year) | [Source](https://web.archive.org/web/20221201161302/https://en.wikipedia.org/wiki/ImageNet#ImageNet_Challenge)*

从 2016 年到现在，关于 CNN 已经有了很多进展。

今天值得我们关注的一个架构是由谷歌人工智能的研究人员在 2019 年 5 月推出的。谷歌人工智能博客上发布了一篇名为“ [EfficientNet:通过 AutoML 和模型缩放提高准确性和效率](https://web.archive.org/web/20221201161302/https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)”的文章。研究小组发明了一种全新的 CNN 架构，称为 EfficientNet，这让每个计算机视觉工程师都大吃一惊。

结果证明它非常好，在所有方面都超过了以前所有最先进的架构:准确性、速度和净尺寸。相当令人印象深刻！

今天，我想向您展示利用 Google 的最新发明是多么简单，并以一种简单的方式将其应用于您的分类问题，只要您在 Tensorflow / Keras 框架中工作。

## 机器学习中的 TensorFlow 和 Keras 框架

框架在每个信息技术领域都是必不可少的。机器学习也不例外。在 ML 市场上有几个成熟的玩家帮助我们简化整体的编程体验。PyTorch、scikit-learn、TensorFlow/Keras、MXNet 和 Caffe 只是值得一提的几个。

今天，我想重点谈谈 TensorFlow/Keras。毫不奇怪，这两个是机器学习领域中最受欢迎的框架。这主要是因为 TensorFlow 和 Keras 都提供了丰富的开发能力。两个框架非常相似。无需深究细节，您应该知道的关键要点是，前者(Keras)只是 TensorFlow 框架的包装器。

关于卷积神经网络，Keras 让我们使用机器学习世界中最新的 CNN 架构来导入和构建我们的模型。查看官方文档页面，在这里你可以找到 Keras 中完整的预训练模型库，以便进行微调。

## 图像分类器创建:真实项目示例

### 项目描述

好了，让我们利用 Keras 中可用的预训练模型，解决一个现实生活中的计算机视觉问题。我们将要进行的项目旨在解决一个图像方向问题。我们需要创建一个可以对输入图像的方向进行分类的模型。输入图像有四种方向选项:

*   正常，
*   逆时针旋转 90 度，
*   逆时针旋转 180 度，
*   逆时针旋转 270 度。

给定输入图像的四个方向，我们可以得出结论，模型应该能够区分四个类别。

![Input image orientation](img/96121d3c601901a13316825f55cd41d1.png)

*Input image orientation options displayed*

该模型不应该检测任何图像的方向。如上所述，模型将处理的图像集仅限于一种类型。

该数据集包含大约 11，000 幅图像，所有图像都经过检查并确认处于正常方向。

### 数据生成器创建

![Data generator creation](img/0da8cbc2f91796a921f48368f9b071dc.png)

*[Source](https://web.archive.org/web/20221201161302/https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c): Keras data generators and how to use them by Ilya Michlin*

因为数据集中的所有图像都处于正常方向，所以我们需要在将它们输入到神经网络之前对它们进行旋转，以确保每个类都被表示出来。为此，我们使用一个定制的图像生成器。

自定义生成器的工作方式如下:

*   在给定路径的情况下，从数据集中读取单个图像；
*   图像旋转到四个方向之一。旋转方向是随机选择的。以相等的概率对每个方向进行采样，得到四个输入类的平衡；
*   使用一组预定义的增强方法来增强旋转的图像；
*   使用传递的预处理函数对旋转和增强的图像进行预处理；
*   堆叠旋转和增强的图像以创建一批给定大小的图像；
*   当这一批形成时，它被输出并输入神经网络。

以下是项目中使用的自定义数据生成器的完整代码:

```py
class DataGenerator(Sequence):

    """
    Generates rotated images for the net that detects orientation
    """

    def __init__(self,
                 data_folder,
                 target_samples,
                 preprocessing_f,
                 input_size,
                 batch_size,
                 shuffle,
                 aug):
        """
        Initialization

        :data_folder: path to folder with images (all images: both train and valid)
        :target_samples: an array of basenames for images to use within generator (e.g.: only those for train)
        :preprocessing_f: input preprocessing function
        :input_size: (typle, (width, height) format) image size to be fed into the neural net
        :batch_size: (int) batch size at each iteration
        :shuffle: True to shuffle indices after each epoch
        :aug: True to augment input images
        """

        self.data_folder = data_folder
        self.target_samples = target_samples
        self.preprocessing_f = preprocessing_f
        self.input_size = input_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.on_epoch_end()
    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return: nuber of batches per epoch
        """
        return math.ceil(len(self.target_samples) / self.batch_size)
    def __getitem__(self, index):
        """
        Generates a batch of data (X and Y)
        """

        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        images_bn4batch = [self.target_samples[i] for i in indices]
        path2images4batch = [os.path.join(self.data_folder, im_bn) for im_bn in images_bn4batch]

        images4batch_bgr = [cv2.imread(path2image) for path2image in path2images4batch]
        images4batch_rgb = [cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB) for bgr_im in images4batch_bgr]

        if self.aug:
            angle4rotation = 2
            images4batch_aug = [self.__data_augmentation(im, angle4rotation) for im in images4batch_rgb]
        else:
            images4batch_aug = images4batch_rgb

        rotated_images, labels = self.__data_generation(images4batch_aug)
        images4batch_resized = [cv2.resize(rotated_im, self.input_size) for rotated_im in rotated_images]

        if self.preprocessing_f:
            prep_images4batch = [self.preprocessing_f(resized_im) for resized_im in images4batch_resized]
        else:
            prep_images4batch = images4batch_resized

        images4yielding = np.array(prep_images4batch)

        return images4yielding, labels
    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """

        self.indices = np.arange(len(self.target_samples))
        if self.shuffle:
            np.random.shuffle(self.indices) 
    def __data_generation(self, images):
        """
        Applies random image rotation and geterates labels.
        Labels map: counter clockwise direction! 0 = 0, 90 = 1, 180 = 2, 270 = 3

        :return: rotated_images, labels
        """

        labels = np.random.choice([0,1,2,3], size=len(images), p=[0.25] * 4)
        rotated_images = [np.rot90(im, angle) for im, angle in zip(images, labels)]

        return rotated_images, labels
    def __data_augmentation(self, image, max_rot_angle = 2):
        """
        Applies data augmentation

        :max_rot_angle: maximum angle that can be selected for image rotation
        :return: augmented_images
        """

        rotation_options = np.arange(-1 * max_rot_angle, max_rot_angle + 1, 1)
        angle4rotation = np.random.choice(rotation_options, 1)

        sometimes = lambda aug: iaa.Sometimes(0.5)

        seq = iaa.Sequential(
            [
            iaa.OneOf(
                [iaa.Add((-15,15), per_channel=False),
                 iaa.Multiply((0.8, 1.2)),
                 iaa.MultiplyHueAndSaturation((0.8,1.1))
            ]),

            iaa.OneOf(
                [iaa.AdditiveGaussianNoise(loc=0, scale=(0.02, 0.05*255), per_channel=0.5),
                 iaa.AdditiveLaplaceNoise(loc=0,scale=(0.02, 0.05*255), per_channel=0.5),
                 iaa.AdditivePoissonNoise(lam=(8,16), per_channel=0.5),
            ]),

            iaa.OneOf(
                [iaa.Dropout(p=0.005,per_channel=False),
                 iaa.Pepper(p=0.005),
                 iaa.Salt(p=0.01)
            ]),

            sometimes(
                iaa.FrequencyNoiseAlpha(
                    exponent=(-1,2),
                    first=iaa.Multiply((0.9, 1.1), per_channel=False),
                    second=iaa.ContrastNormalization((0.8,1.2)))
            ),

            iaa.OneOf([
                iaa.GaussianBlur((0.5, 1)), 
                iaa.AverageBlur(k=(3, 5)), 
                iaa.MotionBlur(k=(5, 7), angle=(0, 359)), 
                iaa.MedianBlur(k=(5, 7)), 
            ]),

            sometimes(iaa.JpegCompression((60,80))),

            iaa.OneOf(
                [iaa.GammaContrast((0.7,1.3)),
                 iaa.GammaContrast((0.7,1.3),per_channel=True),
                 iaa.SigmoidContrast(gain=(5,8)),
                 iaa.LogContrast((0.6,1)),
                 iaa.LinearContrast((0.6,1.4))
            ]),

            sometimes(
                iaa.Affine(rotate = angle4rotation, mode = 'edge')
            )
        ])

        img_aug = seq(images = [image])[0]

        return img_aug
```

应该仔细考虑图像增强。例如，我们的项目假设没有几何变换。由于两个原因，作物、翻耕和翻地应排除在可能的选项之外:

*   这种增强会影响输入图像的方向，结果，我们可能会得到无效的模型预测。
*   这种变换并不反映模型将来要处理的图像。情况并非总是如此，但对于这个特殊的项目，模型将接收没有几何变化的图像。这就是为什么在训练模型时不需要综合应用这些变化。

应该应用主要在像素级工作的其他增强类型。它意味着细微的变化，例如，颜色和对比度。模糊，混合和汇集也适用。要了解更多关于各种增强技术的信息，请看一下 imgaug GitHub 页面。

值得一提的是，该项目将初始化两个数据生成器:

*   第一个用于生成训练数据，
*   第二个将用于产生验证数据。

定型集和验证集之间的数据拆分比例为 8:2。所有数据的 80 %将专用于训练模型，20 %将用于模型评估。

### 神经网络架构设计

我们将通过引入一个主干来开始设计我们未来的图像分类器。Keras 让我们可以访问[它的模型动物园](https://web.archive.org/web/20221201161302/https://www.tensorflow.org/api_docs/python/tf/keras/applications)，有多个 CNN 可供输入。我们的目标是创建一个轻量级的分类器，所以我们肯定应该考虑 EfficientNet，它非常高效和准确。

要导入 EfficientNet，首先你必须决定使用哪种深度。EfficientNet 有 8 个深度级别，从 B0(基线)开始，到最深的 B7 结束。EfficientNet-B7 在 ImageNet 数据集上达到了 84.4%的前 1 名/ 97.1%的前 5 名准确率。

![EfficientNet comparison](img/29c00db0b3134e7acbac712a99de5631.png)

*EfficientNet compared to other popular CNN architectures.
Performance metrics shown on ImageNet dataset.*

要选择合适的深度级别，您应该始终考虑问题的复杂性。从我的个人经验来看，选择 B1 或 B2 是一个很好的起点，因为它们足以解决大多数现实生活中的计算机视觉问题。如果你想了解更多关于 EfficientNet 的架构设计，可以考虑阅读由 [Armughan Shahid](https://web.archive.org/web/20221201161302/https://medium.com/@marmughanshahid?source=post_page-----3fde32aef8ff--------------------------------) 撰写的[这篇文章](https://web.archive.org/web/20221201161302/https://towardsdatascience.com/efficientnet-scaling-of-convolutional-neural-networks-done-right-3fde32aef8ff)。

我们的项目并不复杂。这就是为什么 B1 是高效网络主干的理想深度。让我们导入 EfficientNet B1 并初始化这个类的一个对象。

```py
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input

backbone = EfficientNetB1(include_top = False,
                          input_shape = (128, 128, 3),
                          pooling = 'avg')
```

查看用于初始化的参数集。有很多这样的例子，但我只想重点介绍其中的几个:

*   include_top 是一个布尔值，它指定是否在网络顶部包括完全连接的层。对于一个自定义的图像分类器，我们需要创建我们自己的网络的顶部来反映我们拥有的类的数量；
*   input_shape 是一个指示输入图像尺寸的元组:(图像高度、图像宽度、通道数量)。对于 B1，我决定使用相当小的输入图像大小—(128，128，3)，但请记住，您的卷积神经网络越深，图像大小就越高。你肯定不希望从 B1 移到 B3，并保持输入图像大小不变；
*   池化指定用于特征提取的池化模式。池有助于我们对要素地图中的要素进行下采样。选择的“平均”用于平均池模式。

因为我们没有包括顶部，所以我们必须明确地定义它。特别是，我们应该确定:

*   顶部完全连接的层数；
*   每个全连接层中的神经元数量；
*   每层之后使用的激活函数；
*   用于使人工神经网络更快、更稳定且不容易过度拟合的方法(例如:正则化、规范化等)；
*   反映我们试图解决的分类问题的最后一层设计。

这是我为这个项目选择的顶部架构:

```py
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Softmax
from tensorflow.keras.models import Sequential

n_classes = 4
dense_count = 256

model = Sequential()
model.add(backbone)

model.add(Dense(dense_count))
model.add(LeakyReLU())
model.add(BatchNormalization())

model.add(Dense(n_classes))
model.add(Softmax())
```

如您所见，该模型是使用 Sequence 类实例设计的。第一步，添加模型主干。主干仍然是我们导入并初始化的 EfficientNet CNN 实例。

模型的顶部有两个完全连接的(密集)层。第一个有 256 个神经元和一个 LeakyReLU 激活功能。批量标准化通过图层输入的标准化来确保速度和稳定性。

第二层的神经元数量等于类的数量。Softmax 激活用于进行预测。

这是顶部的基本设计，我用在我开始的大多数基线模型上。作为一个生活黑客，我建议用没有中间全连接层的顶部来训练你的模型，只有用于预测的最终全连接层。

起初，这似乎是一个坏主意，因为它可能会由于模型容量减少而导致模型性能下降。令人惊讶的是，我的个人经验证明 EfficientNet 并非如此。在顶部删除中间层不会导致性能下降。此外，在 10 个案例中有 7 个案例中，它在减小最终模型尺寸的同时带来了更好的性能。很鼓舞人心，不是吗？试一试，你会发现 EfficientNet 有多酷。

### 模特培训

到目前为止，您应该已经有了一个生成器和一个模型设计来继续进行。现在，让我们训练模型并得到结果。

查看如何使用[Neptune+tensor flow/Keras integration](https://web.archive.org/web/20221201161302/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)跟踪模型训练元数据。

我通常使用两阶段方法进行模型训练。当我不想因为迁移学习而引入新的数据集而大幅改变原始权重时，这对于微调尤其方便。随着高学习率的建立，它将导致第一模型层的权重发生显著变化，第一模型层负责简单的低级特征提取，并且已经在 ImageNet 数据集上进行了良好的训练。

在两阶段方法中，我们冻结了几乎整个主干，只留下最后几层可以训练。通过这样的冻结，模型被训练几个时期(如果我的训练数据集足够大，我通常不超过 5-10 个时期)。当这些最后层的权重被训练并且进一步的训练没有提高模型性能时，我们然后解冻整个主干，给模型一个机会对先前冻结的层中的权重进行轻微的改变，因此，在保持训练过程稳定的同时获得更好的结果。

下面是两阶段方法的代码:

**1。模型被冻结**

我们模型的主干有 7 个模块。对于第一阶段，前四个块是冻结的，因此这些块中的所有层都是不可训练的。第五，第六和第七块，以及模型的顶部没有被冻结，将在第一阶段进行训练。下面是在代码中如何执行模型冻结:

```py
block_to_unfreeze_from = 5
trainable_flag = False

for layer in model.layers[0].layers:
    if layer.name.find('bn') != -1:
        layer.trainable = True
    else:
        layer.trainable = trainable_flag

    if layer.name.find(f'block{block_to_unfreeze_from}') != -1:
        trainable_flag = True
        layer.trainable = trainable_flag

for layer in model.layers[0].layers:
    print (layer.name, layer.trainable) 
```

为了检查冻结的结果，使用一个简单的打印语句来检查每一层的可训练性。

**2。模型已编译**

```py
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
```

第一阶段，我建议编一个学习率稍微高一点的模型。例如，1e-3 是一个很好的选择。当在最后一层使用 Softmax 激活函数时，模型产生稀疏输出。这就是为什么稀疏度量和损失函数用于编译:它们可以正确地处理我们的模型产生的稀疏输出。

单词“稀疏”仅仅意味着模型输出一组概率:每个类一个概率。总而言之，所有这些概率总是等于 1。为了得出关于预测类别的结论，应该选择具有最高概率的索引。索引号是模型预测的类别。

如果您想知道稀疏和非稀疏指标有何不同，这里有一个准确性示例:

*   categorical _ accuracy 检查最大真值的*索引*是否等于最大预测值的*索引*。
*   sparse _ categorical _ accuracy 检查最大真值是否等于最大预测值的索引。

**3。使用标准启动培训作业。Tensorflow / Keras 中的拟合方法:**

```py
logdir = os.path.join(dir4saving, 'logs')
os.makedirs(logdir, exist_ok=True)

tbCallBack = keras.callbacks.TensorBoard(log_dir = logdir,
                                         histogram_freq = 0,
                                         write_graph = False,
                                         write_images = False)

first_stage_n = 15

model.fit_generator(generator = train_generator,
                    steps_per_epoch = training_steps_per_epoch,
                    epochs = first_stage_n,
                    validation_data = validation_generator,
                    validation_steps = validation_steps_per_epoch,
                    callbacks=[tbCallBack],
                    use_multiprocessing = True,
                    workers = 16
                   )
```

*   training_steps_per_epoch 和 validation_steps_per_epoch 只是两个整数，计算如下:

–training _ steps _ per _ epoch = int(len(train _ set)/batch _ train)
–validation _ steps _ per _ epoch = int(len(validation _ set)/batch _ validation

*   tbCallBack 是 Tensorboard 的回调
*   first_stage_n 是第一阶段训练的时期数
*   use_multiprocessing 和 workers 是设置多处理和要使用的 CPU 内核数量的两个参数

到第一阶段结束时，模型性能已经达到了一个良好的水平:主要度量(稀疏分类准确率)达到了 80%。现在，让我们解冻之前冻结的块，重新编译模型，并启动第二阶段的培训工作。

**4。解冻模型，为第二个训练阶段做准备:**

以下是解冻是如何完成的:

```py
for layer in model.layers:
    layer.trainable = True

for layer in model.layers[0].layers:
    layer.trainable = True
```

**5。模型被重新编译以应用块冻结中的变化**

编译类似于我们在第一阶段所做的。唯一的区别是学习率值的设置。第二阶段，应该减少。对于这个项目，我将其从 1e-3 降低到 1e-4。

**6。使用标准启动第二阶段培训。Tensorflow / Keras 中的拟合方法**

同样，除了使用的回调次数之外，培训工作启动与我们在第一阶段的情况没有太大的不同。我们有很多这样的机会。我强烈推荐阅读[这篇文章](https://web.archive.org/web/20221201161302/https://blog.paperspace.com/tensorflow-callbacks/)来熟悉 TensorFlow / Keras 中可用的选项。

在第二阶段结束时，模型性能很可能达到了 99.97 %左右——还不错！整个培训使用单个 GeForce RTX 2080 GPU，耗时约 4-5 小时。最终的模型检查点确实非常轻量级，只占用 85 mb 内存。

## 结论

我们已经完成了一个真实的计算机视觉项目，在 TensorFlow / Keras 中创建了一个图像分类器。我们作为主干网使用的 CNN 是来自谷歌的尖端高效网络架构。我们最终得到的模型非常轻。

为了比较，我用其他 CNN 做了很多骨干的实验。我得到的最好结果是 ResNet 50。使用这种架构，最终的模型大小约为 550 mb，达到的最高精度为 95.1 %。即使我们决定更深入地使用 EfficientNet，从 B-0 迁移到 B1、B2 甚至 B3，与 ResNet 相比，它们都要轻得多。

请记住，您可以考虑去掉模型的顶部，得到一个更轻的架构，它的性能也相当不错。在没有顶部部件和 ResNet B-0 的情况下，我能够实现 99.12 %的准确性，并且最终的模型大小非常小:只有 39 mb。为效率网热烈鼓掌！

### 参考

1.  [斯坦福大学 CS231n 课程](https://web.archive.org/web/20221201161302/http://cs231n.stanford.edu/)，讲解视觉识别的卷积神经网络；
2.  [卷积神经网络课程](https://web.archive.org/web/20221201161302/https://www.coursera.org/learn/convolutional-neural-networks)由吴恩达教授，他是一位著名的机器学习讲师，知道如何用简单的术语解释复杂的概念；
3.  来自谷歌的关于图像分类的 ML 实习，这是一个单页的网站，提供了你开始与 CNN 合作时需要知道的最基本的事情。对于那些没有时间，但仍然需要深入主题的人来说，这是一个很好的信息来源。