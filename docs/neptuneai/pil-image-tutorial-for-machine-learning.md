# 必不可少的 Pil(枕头)图像教程(针对机器学习的人)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pil-image-tutorial-for-machine-learning>

PIL 代表 Python 图像库。在本文中，我们将看看它的分叉:枕头。PIL 从 2011 年起就没有更新过，所以枕头的情况很明显。

该库支持各种图像格式，包括流行的 JPEG 和 PNG 格式。你会考虑使用枕头的另一个原因是，它很容易使用，而且很受皮托尼斯塔的欢迎。该软件包是大多数处理图像的数据科学家的常用工具。

它还提供了各种图像处理方法，我们将在本文中看到。这些技术非常有用，尤其是在增加计算机视觉问题的训练数据方面。

以下是您可以期望了解到的内容:

*   如何安装
*   枕头的基本概念
*   枕头图像类
*   用 PIL/枕头阅读和书写图像
*   枕头图像操作，例如裁剪和旋转图像
*   使用枕形滤波器增强图像，例如提高图像质量的滤波器
*   在 PIL 处理图像序列(gif)

我们走吧！

## 安装注意事项

枕头可以通过 pip 安装。

```py
pip install Pillow

```

需要注意的是，枕头和 PIL 包不能在同一环境中共存。因此，确保在安装枕头时没有安装 PIL。

现在拿起你的枕头，我们开始吧。

## PIL 形象的基本概念

我们将从理解几个关键的 PIL 形象概念开始。

为了看到这些概念的实际应用，让我们从使用 Pillow 加载这个[图像](https://web.archive.org/web/20220926102324/https://unsplash.com/photos/GvyyGV2uWns)开始。第一步是从 PIL 进口`Image`级。我们将在下一部分详细讨论`Image`类。

### 注意:

我们从 PIL 而不是枕头导入图像类，因为枕头是 PIL 叉子。因此，展望未来，你应该期待从 PIL 进口，而不是枕头。

```py
from PIL import Image
im = Image.open("peacock.png")
```

加载完图像后，让我们开始讨论那些图像概念。

### **波段**

每个图像都有一个或多个条带。使用 Pillow，我们可以在图像中存储一个或几个波段。例如，一幅彩色图像通常分别有红色、蓝色和绿色的“R”、“G”和“B”波段。

下面是我们如何为上面导入的图像获取波段。

```py
im.getbands()
('R', 'G', 'B')

```

### **模式**

模式是指图像中像素的类型和深度。目前支持的一些模式有:

*   l 代表黑色和白色
*   真彩色的 RGB
*   带透明遮罩的真彩色 RGBA
*   彩色视频格式的 YCbCr

下面是我们如何获取上面加载的图像的模式:

```py
im.mode
'RGB'

```

### **尺寸**

我们还可以通过图像属性获得图像的尺寸。

### 注意:

上面加载的图像非常大，所以我缩小了尺寸，以便在本文后面的部分更容易看到。

```py
im.size
(400, 600)

```

### **坐标系**

[枕](https://web.archive.org/web/20220926102324/https://pillow.readthedocs.io/en/stable/)包使用笛卡尔像素坐标系。在本文的后面部分，我们将使用这个概念，所以理解它是至关重要的。在这个系统中:

*   (0，0)是左上角
*   坐标作为元组以(x，y)的形式传递
*   矩形被表示为 4 个元组，首先提供左上角。

## 了解图像类

正如我们前面提到的，在读入我们的[映像](https://web.archive.org/web/20220926102324/https://pillow.readthedocs.io/en/stable/reference/Image.html)之前，我们必须从 PIL 导入`Image`类。这个类包含的函数使我们能够加载图像文件以及创建新的图像。接下来，我们将使用的函数已经作为导入`Image`的结果被导入，除非另有说明。

### **加载并保存图像**

我们已经看到，我们可以使用`Image.open("peacock.jpg")`加载图像，其中`peacock.jpg`是图像位置的路径。

### **从字符串中读取**

为了演示如何使用 Pillow 读入图像字符串，我们将首先通过 [base64](https://web.archive.org/web/20220926102324/https://en.wikipedia.org/wiki/Base64) 将图像转换为字符串。

```py
import base64

with open("peacock.jpg", "rb") as image:
    image_string = base64.b64encode(image.read())
```

我们现在可以解码图像字符串，并使用 PIL 的`Image`类将其作为图像加载。

```py
import io

image = io.BytesIO(base64.b64decode(image_string))
Image.open(image)
```

### **转换成 JPEG**

现在让我们举一个例子，看看如何将一幅图像转换成 JPEG 格式。

### **PIL 保存图像**

图像转换是通过读入图像并以新格式保存来实现的。这里是我们如何将孔雀 PNG 图像转换为 JPEG 格式。

```py
im.save('peacock.jpg')
```

### **创建 JPEG 缩略图**

在某些情况下，我们会对缩小图像的尺寸感兴趣。

例如，为了获得图像的缩略图，可以缩小图像的尺寸。这可以通过定义缩略图的大小并将其传递给`thumbnail`图像函数来实现。

```py
size = 128, 128
im.thumbnail(size)
im.save('thumb.png')
```

## 图像处理

我们已经看到，我们可以操纵图像的各个方面，如大小。让我们更深入一点，看看其他方面，如图像旋转和颜色转换-仅举几个例子。

### **裁剪图像**

为了裁剪一个图像，我们首先定义一个框来指定我们想要裁剪的图像区域。接下来，我们将这个盒子传递给`Image`类的“crop”函数。

```py
im = Image.open('peacock.jpg')
box = (100, 150, 300, 300)
cropped_image = im.crop(box)
cropped_image

```

### **旋转图像**

旋转图像是通过`Image`类的`rotate`函数完成的。

```py
rotated = im.rotate(180)
rotated
```

### **合并图像**

该软件包还允许我们合并两个图像。让我们通过将一个徽标合并到孔雀图像中来说明这一点。我们从进口开始。

```py
logo = Image.open('logo.png')
```

现在让我们定义标志的位置，并合并两个图像。

```py
position = (38, 469)
im.paste(logo, position)
im.save('merged.jpg')
```

此操作将取代原始图像。所以，如果想保留原图，可以做一个拷贝。

```py
image_copy = image.copy()
```

现在，如果您使用 PNG 图像，您可以利用 Pillow 的遮罩功能来消除黑色背景。

```py
im = Image.open("peacock.jpg")
image_copy = im.copy()
position = ((image_copy.width - logo.width), (image_copy.height - logo.height))
image_copy.paste(logo, position,logo)
image_copy
```

### **翻转图像**

现在让我们看看如何翻转上面的图像。这是使用“翻转”方法完成的。一些翻转选项是`FLIP_TOP_BOTTOM`和`FLIP_LEFT_RIGHT`。

```py
im.transpose(Image.FLIP_TOP_BOTTOM)
```

### **PIL 图像到 NumPy 数组**

Pillow 还允许我们将图像转换成 NumPy 数组。将图像转换为 NumPy 数组后，我们可以使用 PIL 读取它。

```py
import numpy as np
im_array = np.array(im)
```

随着图像的转换，我们现在可以使用枕头加载它。这是使用 Pillow 的 Image 类的`fromarray`函数完成的。最后，我们使用 PIL `show`图像功能保存并显示图像。

```py
img = Image.fromarray(im_array, 'RGB')
img.save('image.png')
img.show()
```

### **颜色变换**

我们可以将彩色图像转换成黑白图像，反之亦然。这是通过 convert 函数并传递首选颜色格式来完成的。

```py
im.convert('L')
```

可以用类似的方式转换成彩色。

```py
im.convert('RGBA')
```

### **在图像上绘图**

我们马上就要结束了，在此之前让我们再看几个项目，包括图片上的[画](https://web.archive.org/web/20220926102324/https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html?highlight=Color%20Transformations)。Pillow 允许通过`ImageDraw`模块来完成，因此，我们从导入它开始。

```py
from PIL import  ImageDraw
```

我们将从定义一个大小为 400×400 的空白彩色图像开始。然后我们使用`ImageDraw`来绘制图像。

```py
image = Image.new('RGB', (400, 400))
img_draw = ImageDraw.Draw(image)
```

现在我们可以使用`ImageDraw`对象在图像上绘制一个矩形。我们用白色填充它，给它一个红色的轮廓。使用相同的对象，我们可以在图像上写一些文字，如图所示。

```py
img_draw.rectangle((100, 30, 300, 200), outline='red', fill='white')
img_draw.text((150, 100), 'Neptune AI', fill='red')
image.save('drawing.jpg')
```

## 图像增强

Pillow 还附带了使我们能够执行图像增强的功能。这是一个提高图像原始质量的过程。

我们从导入提供这些功能的模块开始。

```py
from PIL import ImageEnhance
```

例如，我们可以调整图像的锐度:

```py
from PIL import ImageEnhance
enhancer = ImageEnhance.Sharpness(im)
enhancer.enhance(10.0)
```

让我们再举一个例子，我们把图像的亮度加倍。

```py
enhancer = ImageEnhance.Contrast(im)
enhancer.enhance(2)
```

### **过滤器**

我们可以用 Pillow 做的另一件超级酷的事情是给图像添加[滤镜](https://web.archive.org/web/20220926102324/https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html?highlight=Filters#filters)。第一步是导入`ImageFilter`模块。

```py
from PIL import ImageFilter
```

例如，我们可以像这样模糊图像:

```py
from PIL import ImageFilter
im = Image.open("peacock.jpg")

im.filter(ImageFilter.BLUR)

```

其他可用的过滤器包括:

```py
im.filter(ImageFilter.CONTOUR)
```

```py
im.filter(ImageFilter.DETAIL)

```

```py
im.filter(ImageFilter.EDGE_ENHANCE)

```

```py
im.filter(ImageFilter.EMBOSS)
```

```py
im.filter(ImageFilter.FIND_EDGES)
```

## 在 PIL 处理图像序列(gif)

我们也可以加载图像序列，如 GIF 图像。先来导入[图像序列](https://web.archive.org/web/20220926102324/https://pillow.readthedocs.io/en/stable/reference/ImageSequence.html?highlight=image%20sequence)模块。

```py
from PIL import ImageSequence
```

接下来，我们将加载一个 GIF 文件，并将前两帧保存为 PNG 文件。因为帧太多，我们中断了循环。

```py
im = Image.open("embeddings.GIF")

frame_num = 1
for frame in ImageSequence.Iterator(im):
    frame.save("frame%d.png" % frame_num)
    frame_num = frame_num + 1
    if frame_num == 3:
        break
```

让我们来看一个已经保存为 PNG 文件的帧。

## 最后的想法

希望这篇文章能让你对如何在图像处理管道中应用 Pillow 有所了解。

这种用例的一个例子是在[图像分割问题](/web/20220926102324/https://neptune.ai/blog/image-segmentation-in-2020)中使用的图像增强序列。这将帮助您创建更多的图像实例，最终提高图像分割模型的性能。

现在，让我们回顾一下我们讲述的一些内容:

*   安装 PIL/枕头的过程
*   枕头的基本概念
*   使用 PIL/Pillow 中的图像类
*   在 PIL/枕头上阅读和书写图像
*   在 Pillow 中处理图像
*   使用枕头提高图像质量
*   给图像添加滤镜
*   读取图像序列(gif)
*   将枕头图像转换为数字图像

你可以马上开始应用你在这篇文章中学到的技巧。一个很好的应用示例是深度学习模型的图像增强。综合增加训练数据的数量是提高模型性能的最佳(也是最简单)方法之一。

试试吧！

### 德里克·姆维蒂

Derrick Mwiti 是一名数据科学家，他对分享知识充满热情。他是数据科学社区的热心贡献者，例如 Heartbeat、Towards Data Science、Datacamp、Neptune AI、KDnuggets 等博客。他的内容在网上被浏览了超过一百万次。德里克也是一名作家和在线教师。他还培训各种机构并与之合作，以实施数据科学解决方案并提升其员工的技能。你可能想看看他在 Python 课程中完整的数据科学和机器学习训练营。

* * *

**阅读下一篇**

## Python 中的图像处理:你应该知道的算法、工具和方法

9 分钟阅读|作者 Neetika Khandelwal |更新于 2021 年 5 月 27 日

图像定义了世界，每张图像都有自己的故事，它包含了许多在许多方面都有用的重要信息。这些信息可以借助于被称为**图像处理**的技术来获得。

它是计算机视觉的核心部分，在机器人、自动驾驶汽车和物体检测等许多现实世界的例子中起着至关重要的作用。图像处理允许我们一次转换和操作数千幅图像，并从中提取有用的见解。它在几乎每个领域都有广泛的应用。

Python 是为此目的广泛使用的编程语言之一。它惊人的库和工具有助于非常有效地完成图像处理任务。

通过本文，您将了解处理图像并获得所需输出的经典算法、技术和工具。

让我们开始吧！

[Continue reading ->](/web/20220926102324/https://neptune.ai/blog/image-processing-in-python-algorithms-tools-and-methods-you-should-know)

* * *