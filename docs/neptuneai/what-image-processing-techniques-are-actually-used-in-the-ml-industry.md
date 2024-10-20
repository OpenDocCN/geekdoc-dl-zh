# ML 行业实际使用的图像处理技术有哪些？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/what-image-processing-techniques-are-actually-used-in-the-ml-industry>

处理可用于提高图像质量，或帮助您从中提取有用的信息。它在医学成像等领域很有用，甚至可以用来在图像中隐藏数据。

在这篇文章中，我将告诉你如何在机器学习中应用图像处理，以及你可以使用哪些技术。首先，让我们探索更多真实世界的图像处理示例。

## 现实世界中的图像处理

### 医学成像

在医学上，科学家研究生物体的内部结构和组织，以帮助更快地识别异常。医学成像中使用的图像处理有助于为科学和医学研究生成高质量、清晰的图像，最终帮助医生诊断疾病。

### 安全性

汽车经销公司或商店可以安装安全摄像机来监控该区域，并在窃贼出现时记录下来。但有时，安全摄像头生成的图像需要进行处理，要么将尺寸翻倍，要么增加亮度和对比度，使细节足够清晰，以捕捉其中的重要细节。

### 军事和国防

图像处理在国防领域的一个有趣应用是隐写术。专家可以将信息或图像隐藏在另一个图像中，并在没有任何第三方检测到信息的情况下来回发送信息。

### 常规图像锐化和恢复

这可能是图像处理最广泛的应用。使用 Photoshop 等工具增强和处理图像，或者在 Snapchat 或 Instagram 上使用滤镜，让我们的照片更酷。

如果您需要自动处理大量的图像，手动处理将是一种非常乏味的体验。这就是机器学习算法可以提高图像处理速度的地方，而不会失去我们需要的最终质量。

## ML 行业中使用的图像处理技术

在我继续之前，重要的是要提到图像处理不同于计算机视觉，但人们经常把这两者混淆起来。

图像处理只是计算机视觉的一个方面，两者并不相同。图像处理系统专注于将图像从一种形式转换为另一种形式，而计算机视觉系统帮助计算机理解图像并从中获取含义。

许多计算机视觉系统采用图像处理算法。例如，面部增强应用程序可以使用计算机视觉算法来检测照片中的面部，然后对其应用图像处理技术，如平滑或灰度过滤器。

许多高级图像处理方法利用深度神经网络等机器学习模型来转换各种任务的图像，如应用艺术滤镜，调整图像以获得最佳质量，或增强特定图像细节以最大限度地提高计算机视觉任务的质量。

卷积神经网络(CNN)接受输入图像并对其使用过滤器，以某种方式学习做诸如对象检测、图像分割和分类之类的事情。

除了进行图像处理，最近的机器学习技术使工程师有可能[增加图像数据](https://web.archive.org/web/20221207135524/http://ai.stanford.edu/blog/data-augmentation/)。机器学习模型只和数据集一样好——但是当你没有必要数量的训练数据时，你该怎么办？

我们可以从现有的数据中构建全新的数据集，而不是试图寻找和标记更多的数据集。我们通过应用简单的图像转换技术(水平翻转、色彩空间增强、缩放、随机裁剪)或使用深度学习算法(如特征空间增强和自动编码器、生成对抗网络(GANs)和元学习)来实现这一点。

## 使用 Keras 的示例图像处理任务(带代码示例)

让我们了解如何应用数据扩充来生成图像数据集。我们将拍摄一张狗的图像，对其应用变换，如右移、左移和缩放，以创建图像的全新版本，该图像稍后可用作计算机视觉任务(如对象检测或分类)的训练数据集。

通过 [Neptune + Keras 集成](https://web.archive.org/web/20221207135524/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)和 [Neptune + matplotlib 集成](https://web.archive.org/web/20221207135524/https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/matplotlib)，了解如何跟踪您的模型训练并将所有元数据放在一个地方

### 初始设置

在本教程中，我们将主要依赖四个 Python 包:

1.  Keras: Keras 有一个[图像数据预处理](https://web.archive.org/web/20221207135524/https://keras.io/api/preprocessing/image/#imagedatagenerator-class)类，允许我们无缝地执行数据扩充。
2.  matplotlib:Python 中最流行的数据可视化库之一。它允许我们创建图形和地块，并使它非常容易产生静态光栅或矢量文件，而不需要任何图形用户界面。
3.  Numpy:一个非常有用的库，用于对数组执行数学和逻辑运算。在本教程中，我们将使用它的 expand_dim 类来扩展数组的形状。
4.  Pillow:一个 python 图像库，我们将在本教程中使用它来打开和操作我们的图像文件。

让我们继续安装这些库。

在终端/命令提示符下，键入:

```py
pip3 list

```

查看您已经安装在计算机上的 python 包。然后安装缺失的软件包:

```py
pip3 install numpy matplotlib keras numpy pillow

```

现在我们已经安装了必要的包，让我们继续第 1 步。

**第一步**

创建一个名为`data-aug-sample`的文件夹。在里面，创建一个名为 sample.py 的 python 文件，然后从网上下载一张狗的样本照片，并将其作为 dog.jpg 保存在这个文件夹中。然后像这样导入库:

```py
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator 
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')
plt.imshow(image)
plt.show()

```

**现在，我们的文件夹结构应该是这样的:**

保存该文件，并在您的终端中运行它，如下所示:`python3 sample.py`。

您应该会看到类似这样的内容:

**注意:**如果你在 Jupyter 笔记本上运行本教程，你必须在你的导入之后添加行`%matplotlib inline`来帮助 matplotlib 容易地显示图形。我更喜欢在 Python 文件中运行我所有的机器学习程序，而不是在 Jupyter notebook 中，因为这让我感觉事情在现实中会是什么样子。

**第二步**

现在让我们开始在图像上应用变换操作。

### 旋转

旋转变换在 1 到 359 之间的轴上从右向左旋转图像。在下面的例子中，我们将狗的图像旋转了 90 度。Keras `ImageDataGenerator`类允许我们为此传递一个`rotation_range`参数:

```py
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator 
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)
data_generated = ImageDataGenerator(rotation_range=90)  
it = data_generated.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

运行上面的代码给了我们一个新的狗数据集:

### 翻译

我们可以在图像上应用水平、垂直、向左或向右的移位变换。这种转换对于避免数据中的位置偏差非常有用。例如，在人脸位于图像中心的数据集上训练人脸识别模型会导致位置偏差，使模型在位于左侧或右侧的新人脸上表现非常差。为此，我们将使用`ImageDataGenerator`类的`height_shift_range`和`width_shift_range parameters`。

将**垂直移动**变换应用到我们的狗图像:

```py
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)
data_generator = ImageDataGenerator(height_shift_range=0.5)
it = data_generator.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

我们的结果将如下所示:

将**水平移动**变换应用到我们的狗图像:

```py
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)

it = data_generator.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

我们的结果将如下所示:

### 彩色空间

在这里，我们将应用我们的狗图像的颜色通道空间的转换。这里发生的事情是，它隔离了一个单一的颜色通道(R，G 或 B ),结果是图像的一个明亮或黑暗的版本。通过简单地在`ImageDataGenerator`类中指定`brightness_range`值(通常是一个元组或两个浮点数的列表),我们可以设置亮度偏移值进行选择。

```py
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

结果集是:

### 变焦

顾名思义，我们可以通过简单地传入`ImageDataGenerator`类的`zoom_range`属性，对我们的狗图像应用变换来获得图像的放大/缩小版本。

```py
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(zoom_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

结果集是:

### 轻弹

应用翻转变换允许我们通过`setting vertical_flip=True`或`horizontal_flip=True`水平或垂直改变图像的方向。

```py
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

image = Image.open('dog.jpg')

data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(vertical_flip=True)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()

```

我们的结果集是:

有了我们生成的这个新数据集，我们可以清理它，并消除扭曲的图像或那些包含无意义信息的图像。然后，它可以用于训练对象检测模型或狗分类器。

## 结论

机器学习算法允许你进行大规模的图像处理，并且有很好的细节。我希望你对图像处理如何用于机器学习有所了解，不要忘记图像处理与计算机视觉不一样！