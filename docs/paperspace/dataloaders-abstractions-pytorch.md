# PyTorch 中数据加载器类和抽象的综合指南

> 原文：<https://blog.paperspace.com/dataloaders-abstractions-pytorch/>

在这篇文章中，我们将处理机器学习和深度学习领域最具挑战性的问题之一:加载和处理不同类型的数据。

假设您已经熟悉了 PyTorch 中的神经网络编码，现在您正在使用带有多层感知器的 MNIST 数据集预测一个数字。在这种情况下，您可能使用 torch `DataLoader`类直接加载图像并将其转换为张量。但是现在，在这篇文章中，我们将学习如何超越`DataLoader`类，并遵循在处理各种形式的数据(如 CSV 文件、图像、文本等)时可以使用的最佳实践。以下是我们将要涉及的主题。

*   处理数据集
*   PyTorch 中的数据加载
*   深入查看 MNIST 数据集
*   转换和重新调整数据
*   在 PyTorch 中创建自定义数据集
*   摘要

您可以跟随代码并在 ML Showcase 的[渐变社区笔记本](https://ml-showcase.paperspace.com/projects/working-with-data-in-pytorch)上免费运行它。

## 处理数据集

如果你正在从事一个涉及深度学习的实时项目，通常你的大部分时间都用于处理数据，而不是你要建立的神经网络。这是因为数据就像是你网络的燃料:越合适，结果就越快越准确！你的神经网络表现不佳的一个主要原因可能是由于坏的，或理解不充分的数据。因此，以更直观的方式理解、预处理数据并将其加载到网络中是非常重要的。

在许多情况下，我们在默认或众所周知的数据集(如 MNIST 或 CIFAR)上训练神经网络。在进行这些工作时，我们可以轻松地实现预测和分类类型问题的 90%以上的准确率。原因是，这些数据集组织整齐，易于预处理。但是当你在自己的数据集上工作时，要达到高精度是相当棘手和具有挑战性的。在接下来的章节中，我们将学习如何使用自定义数据集。在此之前，我们将快速浏览一下 PyTorch 库中包含的数据集。

PyTorch 附带了几个内置数据集，所有这些都预加载在类`torch.datasets`中。想起什么了吗？在前面的例子中，当我们对 MNIST 图像进行分类时，我们使用同一个类来下载我们的图像。包里有什么`torch`和`torchvision`？包`torch`由实现神经网络所需的所有核心类和方法组成，而`torchvision`是一个支持包，由流行的数据集、模型架构和计算机视觉的常见图像转换组成。还有一个名为`torchtext`的包，它拥有 PyTorch 自然语言处理的所有基本工具。该包由与文本相关的数据集组成。

这里有一个包含在类`torchvision`和`torchtext`中的数据集的快速概述。

### 火炬视觉中的数据集

**MNIST:** MNIST 是一个数据集，由归一化和中心裁剪的手写图像组成。它有超过 60，000 张训练图像和 10，000 张测试图像。这是学习和实验中最常用的数据集之一。要加载和使用数据集，您可以在安装完`torchvision`包后使用下面的语法导入。

*   **T2`torchvision.datasets.MNIST()`**

**时尚 MNIST:** 该数据集类似于 MNIST，但该数据集包括 t 恤、裤子、包包等服装项目，而不是手写数字。训练和测试样本数分别为 60000 和 10000。下面是 FMNIST 类的位置。

*   **T2`torchvision.datasets.FashionMNIST()`**

**CIFS ar:****CIFS ar 数据集有两个版本，CIFAR10 和 CIFAR100。CIFAR10 由 10 个不同标签的图像组成，而 CIFAR100 有 100 个不同的类别。这些包括常见的图像，如卡车、青蛙、船、汽车、鹿等。建议将此数据集用于构建 CNN。**

*   ****T2`torchvision.datasets.CIFAR10()`****
*   ****T2`torchvision.datasets.CIFAR100()`****

**这个数据集由超过 100，000 个日常物品组成，比如人、瓶子、文具、书籍等。这个图像数据集广泛用于对象检测和图像字幕应用。下面是可以加载 COCO 的位置:**

*   ****T2`torchvision.datasets.CocoCaptions()`****

****EMNIST:** 该数据集是 MNIST 数据集的高级版本。它由图像组成，包括数字和字母。如果您正在处理一个基于从图像中识别文本的问题，这是进行训练的合适数据集。下面是课堂:**

*   ****`torchvision.datasets.EMNIST()`****

****IMAGE-NET:** ImageNet 是用于训练高端神经网络的旗舰数据集之一。它包含超过 120 万张图片，分布在 10，000 个类别中。通常，该数据集被加载到高端硬件系统上，因为单靠 CPU 无法处理如此大规模的数据集。下面是加载 ImageNet 数据集的类:**

*   ****T2`torchvision.datasets.ImageNet()`****

**这些是在 PyTorch 中构建神经网络时最常用的几个数据集。其他一些包括 KMNIST，QMNIST，LSUN，STL10，SVHN，PhotoTour，SBU，Cityscapes，SBD，USPS，Kinetics-400。你可以从 PyTorch 的官方文档中了解更多。**

### **Torchtext 中的数据集**

**如前所述，`torchtext`是一个支持包，由所有自然语言处理的基本工具组成。如果你是 NLP 的新手，它是人工智能的一个子领域，处理和分析大量的自然语言数据(大多与文本相关)。**

**现在让我们来看看几个流行的文本数据集进行实验和工作。**

****IMDB:** 这是一个用于情感分类的数据集，包含 25，000 条用于训练的高度极性电影评论，以及另外 25，000 条用于测试的评论。我们可以通过使用`torchtext`中的以下类来加载这些数据:**

*   ****T2`torchtext.datasets.IMDB()`****

****WikiText2:** 这个语言建模数据集是超过 1 亿个标记的集合。它摘自维基百科，保留了标点符号和实际的字母大小写。它广泛用于涉及长期依赖的应用程序中。该数据可从`torchtext`加载，如下所示:**

*   ****T2`torchtext.datasets.WikiText2()`****

**除了上面两个流行的数据集，在`torchtext`库中还有更多可用的，如 SST、TREC、SNLI、MultiNLI、WikiText-2、WikiText103、PennTreebank、Multi30k 等。**

**到目前为止，我们已经看到了基于一组预定义的图像和文本的数据集。但是如果你有自己的呢？你是怎么装的？现在让我们学习一下`ImageFolder`类，您可以用它来加载您自己的图像数据集。**

### **ImageFolder 类**

**`ImageFolder`是`torchvision`中的一个通用数据加载器类，可以帮助你加载你自己的图像数据集。让我们想象一下，你正在处理一个分类问题，并建立一个神经网络来识别给定图像是苹果还是橙子。要在 PyTorch 中做到这一点，第一步是在默认的文件夹结构中排列图像，如下所示:**

```py
 `root
├── orange
│   ├── orange_image1.png
│   └── orange_image1.png
├── apple
│   └── apple_image1.png
│   └── apple_image2.png
│   └── apple_image3.png` 
```

 **如图所示排列数据集后，可以使用`ImageLoader`类加载所有这些图像。下面是您可以使用的代码片段:

```py
torchvision.datasets.ImageFolder(root, transform)
```

在下一节中，让我们看看如何将数据加载到我们的程序中。

## PyTorch 中的数据加载

数据加载是建立深度学习管道或训练模型的第一步。当数据的复杂性增加时，这项任务变得更具挑战性。在本节中，我们将了解 PyTorch 中的`DataLoader`类，它帮助我们加载和迭代数据集中的元素。该类在`torch.utils.data`模块中作为`DataLoader`提供。`DataLoader`可以导入如下:

```py
from torch.utils.data import DataLoader
```

现在让我们详细讨论一下`DataLoader`类接受的参数，如下所示。

```py
from torch.utils.data import DataLoader

DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 ) 
```

**1。数据集:**`DataLoader`类中的第一个参数是`dataset`。这是我们加载数据的地方。

**2。数据分批:** `batch_size`指一次迭代中使用的训练样本数。通常，我们将数据分为训练集和测试集，并且每个数据集可能有不同的批量大小。

**3。搅乱数据:** `shuffle`是传递给`DataLoader`类的另一个参数。该参数接受一个布尔值(真/假)。如果随机播放设置为`True`，那么所有的样本都被随机播放并批量加载。否则，它们将被一个接一个地发送，没有任何洗牌。

**4。允许多处理:**由于深度学习涉及到用大量数据训练模型，只运行单个进程最终会花费大量时间。在 PyTorch 中，您可以通过使用参数`num_workers`允许多重处理来增加同时运行的进程数量。这也取决于批处理的大小，但是我不会将`num_workers`设置为相同的数字，因为每个工人只装载一个批处理，并且只在它准备好的时候才返回它。

*   **`num_workers=0`** 表示它是在需要时加载数据的主进程。
*   **`num_workers=1`** 表示你只有一个工人，所以可能会慢。

**5。合并数据集:**如果我们想要合并数据集，就使用`collate_fn`参数。此参数是可选的，主要用于从地图样式的数据集中加载批处理。

**6。在 CUDA 张量上加载数据:**您可以使用`pin_memory`参数直接将数据集加载为 CUDA 张量。它是一个可选参数，接受一个布尔值；如果设置为`True`,`DataLoader`类会在返回张量之前将它们复制到 CUDA 固定的内存中。

让我们看一个例子来更好地理解通常的数据加载管道。

## 深入查看 MNIST 数据集

PyTorch 的`torchvision`存储库托管了一些标准数据集，MNIST 是最受欢迎的之一。现在，我们将了解 PyTorch 如何从 pytorch/vision 存储库中加载 MNIST 数据集。让我们首先下载数据集，并将其加载到一个名为`data_train`的变量中。然后我们将打印一个样本图像。

```py
# Import MNIST
from torchvision.datasets import MNIST

# Download and Save MNIST 
data_train = MNIST('~/mnist_data', train=True, download=True)

# Print Data
print(data_train)
print(data_train[12]) 
```

**输出:**

```py
Dataset MNIST Number of datapoints: 60000 Root location: /Users/viharkurama/mnist_data Split: Train (<PIL.Image.Image image mode=L size=28x28 at 0x11164A100>, 3)
```

现在让我们尝试提取元组，其中第一个值对应于图像，第二个值对应于其各自的标签。下面是代码片段:

```py
import matplotlib.pyplot as plt

random_image = data_train[0][0]
random_image_label = data_train[0][1]

# Print the Image using Matplotlib
plt.imshow(random_image)
print("The label of the image is:", random_image_label) 
```

大多数时候，您不会访问带有索引的图像，而是将包含图像的矩阵发送到您的模型。当您需要准备数据批次时(也许，在每次运行之前对它们进行洗牌)，这很方便。现在让我们看看这是如何实时工作的。让我们使用`DataLoader`类来加载数据集，如下所示。

```py
import torch
from torchvision import transforms

data_train = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=True, download=True, 
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=64,
          shuffle=True
          )

for batch_idx, samples in enumerate(data_train):
      print(batch_idx, samples) 
```

这就是我们如何使用`DataLoader`加载一个简单的数据集。然而，我们不能总是依赖于每个数据集的`DataLoader`。我们经常处理包含不对称分辨率图像的大型或不规则数据集，这就是 GPU 发挥重要作用的地方。

### 在 GPU 上加载数据

我们可以启用 GPU 来更快地训练我们的模型。现在让我们看看在加载数据时可以使用的`CUDA` (GPU 对 PyTorch 的支持)的配置。下面是一个示例代码片段:

```py
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True),
  batch_size=batch_size_train, **kwargs)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True),
  batch_size=batch_size, **kwargs) 
```

在上面，我们声明了一个名为`device`的新变量。接下来，我们编写一个简单的`if`条件来检查当前的硬件配置。如果它支持`GPU`，它会将`device`设置为`cuda`，否则它会将其设置为`cpu`。变量`num_workers`表示并行生成批处理的进程数量。对于数据加载，将`pin_memory=True`传递给`DataLoader`类会自动将提取的数据张量放入固定内存，从而使数据更快地传输到支持 CUDA 的 GPU。

在下一节中，我们将学习转换，它定义了加载数据的预处理步骤。

## 转换和重新调整数据

PyTorch 变换定义了简单的图像变换技术，可将整个数据集转换为独特的格式。例如，考虑包含不同分辨率的不同汽车图片的数据集。训练时，训练数据集中的所有图像应该具有相同的分辨率大小。如果我们手动将所有图像转换为所需的输入大小，这将非常耗时，因此我们可以使用 transforms 来代替；通过几行 PyTorch 代码，我们数据集中的所有图像都可以转换成所需的输入大小和分辨率。您也可以使用`transforms`模块调整它们的大小。最常用的几个操作是`transforms.Resize()`调整图像大小、`transforms.CenterCrop()`从中心裁剪图像，以及`transforms.RandomResizedCrop()`随机调整数据集中所有图像的大小。

现在让我们从`torchvision.datasets`加载 CIFAR10 并应用以下转换:

1.  将所有图像的大小调整为 32×32
2.  对图像应用中心裁剪变换
3.  将裁剪的图像转换为张量
4.  标准化图像

首先，我们导入必要的模块，以及来自`torchvision`模块的`transforms`。NumPy 和 Matplotlib 库用于可视化数据集。

```py
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
```

接下来，我们将定义一个名为`transforms`的变量，在这个变量中，我们将按顺序编写所有预处理步骤。我们使用了`Compose`类将所有的转换操作链接在一起。

```py
transform = transforms.Compose([
    # resize
    transforms.Resize(32),
    # center-crop
    transforms.CenterCrop(32),
    # to-tensor
    transforms.ToTensor(),
    # normalize
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]) 
```

*   `resize`:此`Resize`转换将所有图像转换为定义的大小。在这种情况下，我们希望将所有图像的大小调整为 32×32。因此，我们把`32`作为一个参数。
*   `center-crop`:接下来我们使用`CenterCrop`变换来裁剪图像。我们发送的参数也是分辨率/大小，但是因为我们已经将图像的大小调整为`32x32`，所以图像将与这个裁剪居中对齐。这意味着图像将从中心(垂直和水平)裁剪 32 个单位。
*   `to-tensor`:我们使用方法`ToTensor()`将图像转换成`Tensor`数据类型。
*   `normalize`:这将张量中的所有值归一化，使它们位于 0.5 和 1 之间。

在下一步中，在执行了我们刚刚定义的转换之后，我们将使用`trainloader`将`CIFAR`数据集加载到`trainset`中。

```py
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False) 
```

我们从`torchvision.datasets`获取 CIFAR 数据集，将`train`和`download`参数设置为`True`。接下来，我们将转换参数设置为已定义的`transform`变量。`DataLoader` iterable 被初始化，我们将`trainset`作为参数传递给它。`batch_size`被设置为`4`，随机播放为`False`。接下来，我们可以使用下面的代码片段来可视化图像。在 ML Showcase 上查看相应的[渐变社区笔记本](https://ml-showcase.paperspace.com/projects/working-with-data-in-pytorch)来运行代码并查看结果。

```py
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
     img = img / 2 + 0.5
     npimg = img.numpy()
     plt.imshow(np.transpose(npimg, (1, 2, 0)))
     plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()    

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4))) 
```

除了`Resize()`、`CenterCrop()`和`RandomResizedCrop()`之外，还有各种其他的`Transform`等级。让我们看看最常用的。

## 转换类别

1.  PyTorch 中的这个类在任意位置裁剪给定的 PIL 图像。以下是`RandomCrop`接受的论点:

```py
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0)
```

*   `size`:该参数取一个整数，表示期望的随机裁剪输出大小。例如，如果大小设置为 32，输出将是大小为 32×32 的随机裁剪图像。
*   `padding`:这是一个整数自变量，初始设置为`None`。如果设置为整数，它会为图像添加额外的边框。例如，如果填充设置为`4`，则它会以 4 个单位填充左、上、右和下边框。
*   `pad_if_needed`:可选参数，取布尔值。如果它被设置为`True`，那么它会在图像周围填充一个较小的区域，以避免最小的分辨率误差。默认情况下，该参数设置为`False`。
*   `fill`:该常量值初始化所有填充像素的值。默认填充值为`0`。

2.有时，为了让模型在训练时更加健壮，我们会随机翻转图像。类`RandomHorizontalFlip`用于实现这样的结果。它有一个默认参数，`p`，表示图像翻转的概率(在 0 和 1 之间)。默认值为`0.5`。

```py
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

3.`Normalize` : 将图像标准化，以平均值和标准偏差作为参数。该类有四个参数，如下所示:

```py
torchvision.transforms.functional.normalize(tensor, mean, std, inplace=False)
```

*   `tensor`参数接受一个具有三个值的张量:C、H 和 w。它们分别代表通道的数量、高度和宽度。基于给定的参数，输入图像的所有像素值都被归一化。
*   `mean`和`std`参数接受一系列关于每个通道的平均值和标准偏差。
*   `inplace`参数是一个布尔值。如果设置为`True`，所有操作都应就地计算。

4.`ToTensor` : 这个类将 PIL 图像或者一个 NumPy *n* 维数组转换成一个张量。

```py
torchvision.transforms.functional.to_tensor(img)
```

现在，让我们了解加载自定义数据集背后的机制，而不是使用内置数据集。

## 在 PyTorch 中创建自定义数据集

到目前为止，我们已经学习了加载数据集以及预处理数据的各种方法。在本节中，我们将创建一个由数字和文本组成的简单自定义数据集。我们将讨论 PyTorch 中的`Dataset`对象，它有助于处理数字和文本文件，以及如何针对特定任务优化管道。这里的技巧是抽象数据集类中的`__getitem__()`和`__len__()`方法。

*   `__getitem__()`方法通过索引返回数据集中选定的样本。
*   `__len__()`方法返回数据集的总大小。例如，如果您的数据集包含 1，00，000 个样本，那么`len`方法应该返回 1，00，000。

> *注意，此时，数据尚未加载到内存中。*

下面是解释`__getitem__()`和`__len__()`方法实现的抽象视图:

```py
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError 
```

创建自定义数据集并不复杂，但是作为加载数据的典型过程的附加步骤，有必要构建一个接口来获得一个良好的抽象(至少可以说是一个良好的语法糖)。现在，我们将创建一个包含数字及其平方值的新数据集。让我们称我们的数据集为 SquareDataset。它的目的是返回范围`[a,b]`内值的平方。下面是相关代码:

```py
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SquareDataset(Dataset):
     def __init__(self, a=0, b=1):
         super(Dataset, self).__init__()
         assert a <= b
         self.a = a
         self.b = b

     def __len__(self):
         return self.b - self.a + 1

     def __getitem__(self, index):
        assert self.a <= index <= self.b
        return index, index**2

data_train = SquareDataset(a=1,b=64)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
print(len(data_train)) 
```

在上面的代码块中，我们创建了一个名为 SquareDataset 的 Python 类，它继承了 PyTorch 的 Dataset 类。接下来，我们调用了一个`__init__()`构造函数，其中`a`和`b`分别被初始化为`0`和`1`。`super`类用于从继承的`Dataset`类中访问`len`和`get_item`方法。接下来，我们使用`assert`语句来检查`a`是否小于或等于`b`，因为我们想要创建一个数据集，其中的值位于`a`和`b`之间。

然后，我们使用`SquareDataset` 类创建了一个数据集，其中的数据值在 1 到 64 的范围内。我们将它加载到一个名为`data_train`的变量中。最后，`Dataloader`类为存储在`data_train_loader`中的数据创建了一个迭代器，其中`batch_size`被初始化为 64，`shuffle`被设置为`True`。

数据加载器通过采用面向对象的编程概念来利用 Python 的优点。一个很好的练习是使用一些流行的数据集，包括 CelebA、PIMA、COCO、ImageNet、CIFAR-10/100 等，检查各种数据加载器。

## 摘要

在这篇文章中，我们学习了数据加载和抽象。我们从包`torchvision` 和`torchtext`中可用的数据集开始，回顾了几个流行的数据集。然后我们学习了`DataLoader`类，以及它在按照给定的参数组织数据时的重要性。后来，我们通过查看各种可能的技术将 MNIST 数据集调用到我们的工作空间中，对其进行了深入分析。还引入了数据加载器和转换，在 MNIST 的例子中提到了它们的重要性。通过对`RandomCrop`、`RandomHorizontalFlip`、`Normalize`、`ToTensor`和`RandomRotate`类的解释，对转换及其类有了更深入的了解。此后，通过 PyTorch CUDA 的例子解释了 GPU 优于 CPU 的原因。创建一个自定义数据集并不是一项复杂的任务，这个语句已经通过一小段代码得到了证明。您在本教程中学到的概念和基础知识都是使用 PyTorch 的基础。**