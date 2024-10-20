# Python 中的数据扩充:您需要知道的一切

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/data-augmentation-in-python>

在机器学习( **ML** )中，模型没有很好地从训练数据泛化到看不见的数据的情况被称为**过拟合**。你可能知道，这是应用机器学习中最棘手的障碍之一。

解决这个问题的第一步是真正知道你的模型**过度拟合*。*** 这就是正确的[交叉验证](/web/20220928201835/https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)的用武之地。

识别问题后，您可以通过应用正则化或使用更多数据进行训练来防止问题发生。尽管如此，有时您可能没有额外的数据添加到您的初始数据集。获取和标记额外的数据点也可能是错误的途径。当然，在很多情况下，会有更好的效果，但在工作方面，往往费时费钱。

这就是 [**数据增强**](https://web.archive.org/web/20220928201835/https://www.techopedia.com/definition/28033/data-augmentation) ( **DA** )的用武之地。

## 在本文中，我们将涵盖:

## 什么是数据增强？

**数据扩充**是一种技术，可用于通过从现有数据创建修改后的数据来人为扩大训练集的规模。如果您想要防止**过度拟合**，或者初始数据集太小而无法训练，或者甚至您想要从您的模型中获得更好的性能，那么使用 **DA** 是一个很好的实践。

让我们明确一点，**数据扩充**不仅仅是用来防止**过拟合**。一般来说，拥有大型数据集对于 **ML** 和**深度学习** ( **DL** )模型的性能都至关重要。然而，我们可以通过增加现有的数据来提高模型的性能。这意味着**数据扩充**也有利于增强模型的性能。

一般情况下，在建立 **DL** 模型时，经常使用 **DA** 。这就是为什么在整篇文章中，我们将主要讨论用各种 **DL** 框架来执行**数据扩充**。不过，你应该记住，你也可以为 **ML** 问题扩充数据。

您可以增加:

1.  声音的
2.  文本
3.  形象
4.  任何其他类型的数据

我们将重点放在图像增强，因为这些是最受欢迎的。尽管如此，增加其他类型的数据也同样有效和容易。这就是为什么最好记住一些可以用来扩充数据的常用技术。

### 数据扩充技术

我们可以对原始数据进行各种修改。例如，对于图像，我们可以使用:

1.  **几何变换**–你可以随意翻转、裁剪、旋转或平移图像，而这只是冰山一角
2.  **颜色空间转换**–改变 RGB 颜色通道，增强任何颜色
3.  **内核过滤器**–锐化或模糊图像
4.  **随机擦除**–删除初始图像的一部分
5.  **混合图像**–基本上，将图像彼此混合。可能违反直觉，但它确实有效

对于文本，有:

1.  **单词/句子混排**
2.  **单词替换**–用同义词替换单词
3.  **语法树操作**–使用相同的单词解释句子，使其语法正确
4.  关于 NLP 中的[数据增强的其他描述](/web/20220928201835/https://neptune.ai/blog/data-augmentation-nlp)

对于音频增强，您可以使用:

1.  **噪声注入**
2.  **换挡**
3.  **改变磁带的速度**
4.  还有更多

此外，增强技术的最大优势是你可以一次使用所有这些技术。因此，您可能会从最初的样本中获得大量独特的数据样本。

## 深度学习中的数据增强

上面提到的 [**深度学习，** **数据增强**](https://web.archive.org/web/20220928201835/https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9) 是常见的做法。因此，每个 DL 框架都有自己的增强方法，甚至是一个完整的库。例如，让我们看看如何使用 TensorFlow (TF)和 Keras、PyTorch 和 **MxNet** 中的内置方法来应用图像增强。

### TensorFlow 和 Keras 中的数据扩充

当使用 **TensorFlow** 或 **Keras** 作为我们的 **DL** 框架时，我们可以:

*   使用 **tf.image** 编写我们自己的增强管道或层。
*   使用 **Keras** 预处理层
*   使用**图像数据生成器**

#### Tf .图像

让我们更仔细地看看第一种技术，定义一个可视化图像的函数，然后使用 **tf.image** 将翻转应用于该图像。您可以在下面看到代码和结果。

```py
def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    flipped = tf.image.flip_left_right(image)
    visualize(image, flipped)
```

为了更好的控制，你可以写你自己的增强管道。在大多数情况下，对整个数据集而不是单个图像应用增强是有用的。您可以如下实现它。

```py
import tensorflow_datasets as tfds 

def augment(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5)
  return image, label

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
     with_info=True,
     as_supervised=True,)

train_ds = train_ds
            .shuffle(1000)
            .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
```

当然，这只是冰山一角。 **TensorFlow** API 有大量的增强技术。如果你想了解更多关于这个话题的内容，请查看[官方文件](https://web.archive.org/web/20220928201835/https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en)或[其他文章](https://web.archive.org/web/20220928201835/https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/)。

#### Keras 预处理

如上所述， **Keras** 有多种预处理层，可用于**数据扩充**。您可以按如下方式应用它们。

```py
data_augmentation = tf.keras.Sequential([
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2)])

image = tf.expand_dims(image, 0)
plt.figure(figsize=(10, 10))

for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
```

#### Keras ImageDataGenerator

此外，您可以使用**imagedata generator**(**TF . keras . preprocessing . image . image data generator**)，它可以使用实时 **DA** 生成批量张量图像。

```py
datagen = ImageDataGenerator(rotation_range=90)
datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(img_rows, img_cols, 3))
        pyplot.show()
    break

```

### PyTorch 和 MxNet 中的数据扩充

#### Pytorch 中的变换

**Transforms** library 是 **torchvision** 包的增强部分，该包由流行的数据集、模型架构和用于**计算机视觉**任务的常见图像转换组成。

要安装**转换**你只需要安装**火炬视觉**:

```py
pip3 install torch torchvision

```

**转换**库包含不同的图像转换，可以使用**合成**方法将它们链接在一起。在功能上， **Transforms 实现了多种增强技术**。你可以通过使用 **Compose** 方法来组合它们。只需查看官方文档[就能找到任务的增强功能。](https://web.archive.org/web/20220928201835/https://pytorch.org/docs/stable/torchvision/transforms.html)

此外，还有**torch vision . transforms . functional**模块。它有各种功能转换，可以对转换进行细粒度的控制。如果您正在构建一个更复杂的增强管道，例如，在分段任务的情况下，这可能会非常有用。

除此之外，**变换**并没有什么独特的特性。它主要与 **PyTorch** 一起使用，因为它被认为是一个内置的增强库。

### 更多关于 PYTORCH 闪电

**py torch 变换的示例用法**

让我们看看如何使用**变换**来应用增强。你应该记住，**变换**只适用于 **PIL** 的图像。这就是为什么你要么阅读 PIL 格式的图像，要么对你的增强管道进行必要的转换。

```py
from torchvision import transforms as tr
from torchvision.transfroms import Compose

pipeline = Compose(
             [tr.RandomRotation(degrees = 90),
              tr.RandomRotation(degrees = 270)])

augmented_image = pipeline(img = img)
```

有时你可能想为训练编写一个定制的**数据加载器**。让我们看看如何通过**变换**来应用增强，如果你这样做的话。

```py
from torchvision import transforms
from torchvision.transforms import Compose as C

def aug(p=0.5):
    return C([transforms.RandomHorizontalFlip()], p=p)

class Dataloader(object):
    def __init__(self, train, csv, transform=None):
        ...

    def __getitem__(self, index):
        ...
        img = aug()(**{'image': img})['image']
        return img, target

    def __len__(self):
        return len(self.image_list)

trainset = Dataloader(train=True, csv='/path/to/file/', transform=aug)
```

#### MxNet 中的转换

**Mxnet** 还有一个内置的增强库叫做**Transforms**(**Mxnet . gluon . data . vision . Transforms**)。它非常类似于 **PyTorch 转换**库。几乎没有什么可补充的。如果您想找到关于这个主题的更多信息，请查看上面的**转换**部分。一般用法如下。

**MxNet 转换的示例用法**

```py
color_aug = transforms.RandomColorJitter(
                               brightness=0.5,
                               contrast=0.5,
                               saturation=0.5,
                               hue=0.5)
apply(example_image, color_aug)
```

这些都是很好的例子，但是从我的经验来看，当您使用定制库时，**数据扩充**的真正威力就会显现出来:

*   他们有更广泛的转换方法
*   它们允许您创建自定义增强
*   您可以将一个转换与另一个堆叠在一起。

这就是为什么使用定制的 **DA** 库可能比使用内置库更有效。

## 图像数据增强库

在本节中，我们将讨论以下库:

1.  奥吉
2.  **白蛋白**
3.  **伊姆高格**
4.  **自动增强(DeepAugment)**

我们将查看安装、增强功能、增强过程并行化、自定义增强，并提供一个简单的示例。请记住，我们将侧重于图像增强，因为它是最常用的。

在我们开始之前，我有一些关于在不同的 **DL** 框架中使用定制增强库的一般说明。

一般来说，如果您在训练模型之前执行扩充，所有的库都可以用于所有的框架。

重点是有些库和特定框架有预存的协同，比如**albuminations**和 **Pytorch** 。用这样的对子更方便。不过，如果您需要特定的函数或者您喜欢一个库胜过另一个库，您应该在开始训练模型之前执行 **DA** ，或者编写一个定制的数据加载器和训练过程。

第二个主题是在不同的增强库中使用定制的增强。例如，您想使用自己的 CV2 图像转换，并从**albuminations**库中进行特定的增强。

让我们弄清楚这一点，你可以对任何库这样做，但它可能比你想象的更复杂。一些图书馆在他们的官方文档中有如何做的指南，但是其他的没有。

如果没有向导，你基本上有两种方法:

*   分别应用增强，例如，使用转换操作，然后使用管道。
*   检查一下 Github 仓库，以防有人已经想出如何正确地将定制增强集成到管道中。

好了，现在我们开始吧。

### 奥吉先生

继续谈库， **Augmentor** 是一个 Python 包，它的目标是既是一个**数据扩充**工具，又是一个基本图像预处理函数库。

通过 pip 安装**增强器**非常容易:

```py
pip install Augmentor
```

如果你想从源代码编译这个包，请查阅[官方文档](https://web.archive.org/web/20220928201835/https://augmentor.readthedocs.io/en/master/userguide/install.html)。

一般来说，**增强器**由许多标准图像变换函数的类组成，比如**裁剪**、**旋转**、**翻转**等等。

**增强器**允许用户为每个变换操作选择一个概率参数。此参数控制操作的应用频率。因此，**增强器**允许形成一个增强管道，将大量随机应用的操作链接在一起。

这意味着每次图像通过管道时，都会返回完全不同的图像。根据流水线中的操作数量和概率参数，可以创建非常大量的新图像数据。基本上，这是最好的数据扩充。

使用**增强器**我们可以对图像做什么？**增强器**更侧重于**几何变换**虽然它也有其他增强。**增强器**组件的主要特点是:

1.  **透视倾斜**–从不同的角度观看图像
2.  **弹性变形**–给图像添加变形
3.  **旋转**–简单地说，旋转图像
4.  **剪切**–沿着图像的一边倾斜图像
5.  **裁剪**–裁剪图像
6.  **镜像**–应用不同类型的翻转

**Augmentor** 是一个组织严密的库。您可以将它与各种 **DL** 框架( **TF、Keras、PyTorch、MxNet** )一起使用，因为增强甚至可以在您建立模型之前应用。

此外，**增强器**允许您添加自定义增强。这可能有点棘手，因为它需要[编写](https://web.archive.org/web/20220928201835/https://augmentor.readthedocs.io/en/master/userguide/extend.html)一个新的操作类，但是你可以做到。

不幸的是，Augmentor**在功能上既不非常快也不灵活**。有一些库提供了更多的转换函数，可以更快更有效地执行 **DA** 。这就是为什么**增强器**可能是最不受欢迎的 **DA** 库。

**增强器使用示例**

让我们检查一下**增强器**的简单用法:

1.  我们需要导入它。
2.  我们创建一个空的扩充管道。
3.  在那里添加一些操作
4.  使用**采样**方法获得增强图像。

请注意，当使用**样本**时，您需要指定您想要获得的增强图像的数量。

```py
import Augmentor

p = Augmentor.Pipeline("/path/to/images")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
p.sample(10000)
```

### 白蛋白

**albuminations**是一款计算机视觉工具，旨在执行快速灵活的图像放大。它似乎拥有所有图像增强库中最大的转换函数集。

让我们通过 pip 安装**相册**。如果你想以其他方式做这件事，检查官方文件。

```py
pip install albumentations

```

**albuminations**提供了一个简单的界面来处理不同的计算机视觉任务，如分类、分割、对象检测、姿态估计等。该库为最大速度和性能进行了**优化，拥有大量不同的图像转换操作。**

如果我们谈论的是数据扩充，那么没有什么是**白蛋白**不能做的。说实话，**albuminations**是堆栈最多的库，因为它没有专注于图像转换的某个特定领域。你可以简单地检查官方文档，你会找到一个你需要的操作。

而且，**albuminations**与深度学习框架如 **PyTorch** 和 **Keras** 无缝集成。这个库是 PyTorch 生态系统的一部分，但是你也可以和 TensorFlow 一起使用。因此，**albuminations**是最常用的图像增强库。

另一方面，**albuminations**没有与 **MxNet** 集成，这意味着如果你使用 **MxNet** 作为 **DL** 框架，你应该编写一个定制的数据加载器或者使用另一个增强库。

值得一提的是**albuminations**是一个开源库。如果你愿意，你可以很容易地检查原始的[代码](https://web.archive.org/web/20220928201835/https://github.com/albumentations-team/albumentations#benchmarking-results)。

**白蛋白使用示例**

让我们看看如何使用**相册**来放大图像。您需要使用 **Compose** 方法定义管道(或者您可以使用单个增强)，向其传递一个图像，并获得增强的图像。

```py
import albumentations as A
import cv2

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

image = cv2.imread('/path/to/image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(),
     A.Transpose(),
     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                        rotate_limit=45, p=.75),
     A.Blur(blur_limit=3),
     A.OpticalDistortion(),
     A.GridDistortion(),
     A.HueSaturationValue()])

augmented_image = transform(image=image)['image']
visualize(augmented_image)

```

### 伊姆高格

现在，在阅读了**增强器**和**相册**之后，你可能会认为所有的图像增强库彼此都非常相似。

没错。在许多情况下，每个库的功能是可以互换的。然而，每一个都有自己的主要特点。

ImgAug 也是一个图像增强库。它在功能上与增强器和缓冲区非常相似，但是官方文档中提到的主要特性是 T4 在多个 CPU 内核上执行增强的能力。如果你想这样做，你可能想检查下面的[指南](https://web.archive.org/web/20220928201835/https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb)。

正如你所看到的，这与**增强师专注于几何**变换或 ***融合*** **试图覆盖所有可能的增强**截然不同。

然而， **ImgAug 的**关键特性似乎有点奇怪，因为**增强器**和**缓冲**也可以在多个 CPU 内核上执行。总之 **ImgAug** 支持广泛的增强技术，就像白蛋白一样，并通过细粒度控制实现复杂的增强。

**ImgAug** 可通过 pip 或 [conda](https://web.archive.org/web/20220928201835/https://imgaug.readthedocs.io/en/latest/source/installation.html) 轻松安装。

```py
pip install imgaug

```

**img aug 的使用示例**

与其他图像增强库 ***，* ImgAug** 一样，使用起来也很方便。要定义一个扩充管道，使用**顺序**方法，然后像在其他库中一样简单地堆叠不同的转换操作。

```py
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    		iaa.Crop(px=(0, 16)),
    		iaa.Fliplr(0.5),
    		iaa.GaussianBlur(sigma=(0, 3.0))])

for batch_idx in range(1000):
    		images = load_batch(batch_idx)
    		images_aug = seq(images=images)

```

### 自动增强

另一方面，自动增强更有趣。你可能知道，用**机器学习** ( **ML** )来提高 **ML** 的设计选择，已经到了 **DA** 的空间。

2018 年，谷歌推出了自动增强算法，该算法是**设计的，用于搜索最佳增强**策略。**自动增强帮助提高了最先进的模型在数据集上的性能**，如 **CIFAR-10、CIFAR-100、ImageNet** 等。

然而， **AutoAugment 使用起来很棘手，因为它没有提供控制器模块，这阻止了用户为自己的数据集运行它。这就是为什么只有当我们计划训练的数据集和我们要完成的任务已经有了增强策略时，使用自动增强才是有意义的。**

因此，让我们更仔细地看看 **DeepAugment** ，它比**自动增强**更快、更灵活。 **DeepAugment** 与 **AutoAugment** 除了一般的想法之外没有什么强有力的联系，是由一群爱好者开发的。您可以通过 pip 安装它:

```py
pip install deepaugment

```

知道如何使用 **DeepAugment** 为我们的图像获得最佳的增强策略对我们来说很重要。你可以按照下面的方法来做，或者查看[官方的 Github 库](https://web.archive.org/web/20220928201835/https://github.com/barisozmen/deepaugment)。

请记住，**当您使用优化方法时，您应该指定用于找到最佳增强策略的样本数量**。

```py
from deepaugment.deepaugment import DeepAugment

deepaug = DeepAugment(my_images, my_labels)
best_policies = deepaug.optimize(300)

```

总的来说，**自动增强**和**深度增强**都不常用。不过，如果您不知道哪种增强技术最适合您的数据，运行它们可能会非常有用。你只需要记住，这需要大量的时间，因为要训练多个模型。

值得一提的是，我们没有涵盖所有的自定义图像增强库，但我们已经涵盖了主要的。现在你知道什么库最受欢迎，它们有什么优点和缺点，以及如何使用它们。如果您需要，这些知识将帮助您找到任何其他信息。

## 图像数据增强库的速度比较

您可能已经发现，增强过程在时间和计算上都非常昂贵。

执行 **DA** 所需的时间取决于我们需要转换的数据点的数量、整个扩充流水线的难度，甚至取决于您用来扩充数据的硬件。

让我们进行一些实验，找出最快的增强库。我们将对**增强器**、**白蛋白**、 **ImgAug** 和**转化**进行这些实验。我们将使用来自 Kaggle 的一个[图像数据集](https://web.archive.org/web/20220928201835/https://www.kaggle.com/alxmamaev/flowers-recognition)，它是为花朵识别而设计的，包含超过 4000 张图像。

在我们的第一个实验中，我们将创建一个只包含两个操作的扩充管道。这将是概率为 0.4 的**水平翻转**和概率为 0.8 的**垂直翻转**。让我们将管道应用于数据集中的每个图像，并测量时间。

正如我们所预料的， **Augmentor 的执行速度比其他库**慢。尽管如此，**albumation 和 Transforms 都显示出良好的结果**，因为它们被优化来执行快速增强。

对于我们的第二个实验，我们将**创建一个更复杂的管道，使用各种转换**到**，看看**转换**和**缓冲**是否停留在顶部**。我们将更多的几何变换堆叠成一个流水线。因此，我们将能够使用所有库作为增强器，例如，没有太多的内核过滤操作。

你可以在我为你准备的笔记本中找到完整的流程[。请随意试验和使用它。](https://web.archive.org/web/20220928201835/https://colab.research.google.com/drive/17R6PJRZkwjk7mYUxXCQVtR4o6JaCEVPQ?usp=sharing)

再一次**变换和贴图位于顶部**。

此外，如果我们检查通过 **Neptune** 获得的 CPU 使用图表，我们会发现**缓冲和转换**使用的 CPU 资源都不到 60%。

另一方面，**增强器和 ImgAug** 使用超过 80%。

你可能已经注意到了，**转存和转换都非常快**。这就是它们在现实生活中被广泛使用的原因。

## 最佳实践、提示和技巧

值得一提的是，尽管 **DA** 是一个强大的工具，但你应该小心使用它。在应用增强时，您可能需要遵循一些通用规则:

*   为你的任务选择适当的扩充。让我们想象一下，你正试图检测一张图像上的一张脸。你选择**随机删除**作为增强技术，突然你的模型甚至在训练中表现也不好。这是因为图像上没有人脸，因为它是通过增强技术随机擦除的。同样的事情也适用于声音检测和应用噪声注入到磁带作为一个增强。记住这些案例，在选择 **DA** 技术时要合乎逻辑。

*   不要在一个序列中使用太多的增强。你可能会简单地创建一个全新的观察，与你最初的训练(或测试数据)毫无共同之处
*   在笔记本中显示增强数据(图像和文本),并在开始训练之前聆听转换后的音频样本。当形成一个扩充管道时，很容易出错。这就是为什么复查结果总是更好的原因。

此外，在创建自己的扩充管道之前，检查一下笔记本也是一个很好的做法。你可以在那里找到很多想法。试着为类似的任务找一个笔记本，检查作者是否应用了和你计划的一样的扩充。

## 最后的想法

在本文中，我们已经弄清楚了什么是**数据扩充**，有哪些 **DA** 技术，以及您可以使用哪些库来应用它们。

据我所知，最好的公共图书馆是图书室。这就是为什么如果你正在处理图像并且不使用 **MxNet** 或 **TensorFlow** 作为你的 **DL** 框架，你可能应该使用**albuminations**作为 **DA** 。

希望有了这些信息，你在为下一个机器学习项目设置 **DA** 时不会有任何问题。

## 资源

### 弗拉基米尔·利亚申科

年轻的人工智能爱好者，对医学中的教育技术和计算机视觉充满热情。我想通过帮助其他人学习，探索新的机会，并通过先进的技术跟踪他们的健康状况，让世界变得更美好。

* * *

**阅读下一篇**

## 机器学习中模型评估和选择的最终指南

10 分钟阅读|作者 Samadrita Ghosh |年 7 月 16 日更新

在高层次上，机器学习是统计和计算的结合。机器学习的关键围绕着算法或模型的概念，这些概念实际上是类固醇的统计估计。

然而，根据数据分布的不同，任何给定的模型都有一些限制。它们中没有一个是完全准确的，因为它们只是 ***(即使使用类固醇)*** 。这些限制俗称 ***偏差*** 和 ***方差*** 。

具有高偏差的**模型会由于不太注意训练点而过于简化(例如:在线性回归中，不管数据分布如何，模型将总是假设线性关系)。**

具有高方差的**模型将通过不对其之前未见过的测试点进行概括来将其自身限制于训练数据(例如:max_depth = None 的随机森林)。**

当限制很微妙时，问题就出现了，比如当我们必须在随机森林算法和梯度推进算法之间进行选择，或者在同一决策树算法的两个变体之间进行选择。两者都趋向于具有高方差和低偏差。

这就是模型选择和模型评估发挥作用的地方！

在本文中，我们将讨论:

*   什么是模型选择和模型评估？
*   有效的模型选择方法(重采样和概率方法)
*   流行的模型评估方法
*   重要的机器学习模型权衡

[Continue reading ->](/web/20220928201835/https://neptune.ai/blog/the-ultimate-guide-to-evaluation-and-selection-of-models-in-machine-learning)

* * *