# 机器学习中使用的 8 大图像处理 Python 库

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/image-processing-python-libraries-for-machine-learning>

据 [IDC](https://web.archive.org/web/20221203093533/https://www.idc.com/) 预测，数字数据将暴涨至 175 zettabytes，而这些数据的巨大部分是图像。数据科学家需要在将这些图像输入任何机器学习模型之前对其进行(预)处理。在有趣的部分开始之前，他们必须做重要的(有时是肮脏的)工作。

为了在不影响结果的情况下高效快速地处理大量数据，数据科学家需要使用图像处理工具来完成机器学习和深度学习任务。

在这篇文章中，我将列出 Python 中最有用的图像处理库，它们在机器学习任务中被大量使用。

## 1.OpenCV

[**OpenCV**](https://web.archive.org/web/20221203093533/https://opencv.org/) 是英特尔在 2000 年开发的开源库。它主要用于计算机视觉任务，如物体检测，人脸检测，人脸识别，图像分割等，但也包含了许多有用的功能，你可能需要在 ML。

### **灰度级**

```py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('goku.jpeg')
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fig.tight_layout()

ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title("Original")

ax[1].imshow(cv.cvtColor(gray_image, cv.COLOR_BGR2RGB))
ax[1].set_title("Grayscale")
plt.show()

```

彩色图像由 3 个颜色通道组成，而灰色图像仅由 1 个颜色通道组成，该通道携带每个像素的强度信息，将图像显示为黑白。

以下代码分隔每个颜色通道:

```py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('goku.jpeg')
b, g, r = cv.split(img)

fig, ax = plt.subplots(1, 3, figsize=(16, 8))
fig.tight_layout()

ax[0].imshow(cv.cvtColor(r, cv.COLOR_BGR2RGB))
ax[0].set_title("Red")

ax[1].imshow(cv.cvtColor(g, cv.COLOR_BGR2RGB))
ax[1].set_title("Green")

ax[2].imshow(cv.cvtColor(b, cv.COLOR_BGR2RGB))
ax[2].set_title("Blue")

```

### **图像翻译**

```py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pics/goku.jpeg")
h, w = image.shape[:2]

half_height, half_width = h//4, w//8
transition_matrix = np.float32([[1, 0, half_width],
                               [0, 1, half_height]])

img_transition = cv.warpAffine(image, transition_matrix, (w, h))

plt.imshow(cv.cvtColor(img_transition, cv.COLOR_BGR2RGB))
plt.title("Translation")
plt.show()

```

上面的代码将图像从一个坐标转换到另一个坐标。

### **图像旋转**

```py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pics/goku.jpeg")

h, w = image.shape[:2]
rotation_matrix = cv.getRotationMatrix2D((w/2,h/2), -180, 0.5)

rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))

plt.imshow(cv.cvtColor(rotated_image, cv.COLOR_BGR2RGB))
plt.title("Rotation")
plt.show()
```

图像绕 X 轴或 Y 轴旋转。

### **缩放和调整大小**

```py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pics/goku.jpeg")

fig, ax = plt.subplots(1, 3, figsize=(16, 8))

image_scaled = cv.resize(image, None, fx=0.15, fy=0.15)
ax[0].imshow(cv.cvtColor(image_scaled, cv.COLOR_BGR2RGB))
ax[0].set_title("Linear Interpolation Scale")

image_scaled_2 = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
ax[1].imshow(cv.cvtColor(image_scaled_2, cv.COLOR_BGR2RGB))
ax[1].set_title("Cubic Interpolation Scale")

image_scaled_3 = cv.resize(image, (200, 400), interpolation=cv.INTER_AREA)
ax[2].imshow(cv.cvtColor(image_scaled_3, cv.COLOR_BGR2RGB))
ax[2].set_title("Skewed Interpolation Scale")

```

图像的缩放是指将图像阵列转换成更低或更高的维度。

这些是可以用 OpenCV 在图像上执行的一些最基本的操作。除此之外，OpenCV 还可以执行诸如**图像分割、人脸检测、物体检测、三维重建、特征提取**等操作。

如果你想看看这些图片是如何使用 **OpenCV** 生成的，那么你可以看看这个 GitHub [**库**](https://web.archive.org/web/20221203093533/https://github.com/Akshay594/OpenCV/tree/master/tutorials) 。

## 2\. Scikit-Image

[**scikit-image**](https://web.archive.org/web/20221203093533/https://scikit-image.org/) 是一个基于 python 的图像处理库，其中一些部分是用[**cy thon**](https://web.archive.org/web/20221203093533/https://cython.org/)([cy thon](https://web.archive.org/web/20221203093533/https://cython.org/)是一种编程语言，它是 Python 编程语言的超集，旨在具有类似 C 编程语言的性能。)才能取得好的成绩。它包括以下算法:

*   分段，
*   几何变换，
*   色彩空间操作，
*   分析，
*   过滤，
*   形态学，
*   特征检测等等

你会发现它对几乎任何计算机视觉任务都很有用。

[**scikit-image**](https://web.archive.org/web/20221203093533/https://scikit-image.org/)使用 [**NumPy**](https://web.archive.org/web/20221203093533/https://numpy.org/) 数组作为图像对象。

#### **活动轮廓**

在计算机视觉中，轮廓模型描述了图像中形状的边界。

> 活动轮廓模型是为基于曲线流、曲率和轮廓的图像分割而定义的，以获得图像中精确的目标区域或片段

下面的代码产生了上面的输出:

```py
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img = data.astronaut()

s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T

cntr = active_contour(gaussian(img, 3),init, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Original Image")

ax[1].imshow(img, cmap=plt.cm.gray)

ax[1].plot(init[:, 0], init[:, 1], '--r', lw=3)
ax[1].plot(cntr[:, 0], cntr[:, 1], '-b', lw=3)
ax[1].set_title("Active Contour Image")

```

## 3.我的天啊

[**Scipy**](https://web.archive.org/web/20221203093533/https://www.scipy.org/) 用于数学和科学计算，但也可以使用子模块 [**scipy.ndimage**](https://web.archive.org/web/20221203093533/https://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage) 执行多维图像处理。它提供了在 n 维 Numpy 数组上操作的函数，最终图像就是这样。

[**Scipy**](https://web.archive.org/web/20221203093533/https://www.scipy.org/) 提供了最常用的图像处理操作如:

*   读取图像
*   图象分割法
*   盘旋
*   人脸检测
*   特征提取等等。

### **用**[](https://web.archive.org/web/20221203093533/https://www.scipy.org/)**模糊图像**

```py
from scipy import misc,ndimage
from matplotlib import pyplot as plt

face = misc.face()
blurred_face = ndimage.gaussian_filter(face, sigma=3)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(face)
ax[0].set_title("Original Image")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(blurred_face)
ax[1].set_title("Blurred Image")
ax[1].set_xticks([])
ax[1].set_yticks([])
```

输出:

在 这里可以找到所有操作 [**。**](https://web.archive.org/web/20221203093533/https://docs.scipy.org/doc/scipy/reference/ndimage.html)

## 4.pillow/GDP

**PIL (Python 图像库)**是一个开源库，用于需要 Python 编程语言的图像处理任务。 **PIL** 可以在图像上执行任务，例如读取、缩放、以不同的图像格式保存。

**PIL** 可用于图像存档、图像处理、图像显示。

### **用 PIL 增强图像**

例如，让我们将下面的图像增强 30%的对比度。

```py
from PIL import Image, ImageFilter

im = Image.open('cat_inpainted.png')

im.show()

from PIL import ImageEnhance
enh = ImageEnhance.Contrast(im)
enh.enhance(1.8).show("30% more contrast")
```

输出:

欲了解更多信息，请点击[这里](https://web.archive.org/web/20221203093533/https://pillow.readthedocs.io/en/stable/reference/Image.html)。

## 5.NumPy

图像本质上是像素值的数组，其中每个像素由 1(灰度)或 3 (RGB)值表示。因此，NumPy 可以轻松执行图像裁剪、遮罩或像素值操作等任务。

例如，从下图中提取红/绿/蓝通道:

我们可以使用 numpy，通过用零替换所有的像素值，一次“惩罚”一个通道。

```py
from PIL import Image
import numpy as np

im = np.array(Image.open('goku.png'))

im_R = im.copy()
im_R[:, :, (1, 2)] = 0
im_G = im.copy()
im_G[:, :, (0, 2)] = 0
im_B = im.copy()
im_B[:, :, (0, 1)] = 0

im_RGB = np.concatenate((im_R, im_G, im_B), axis=1)

pil_img = Image.fromarray(im_RGB)
pil_img.save('goku.jpg')
```

## 6.马霍塔斯

[**Mahotas**](https://web.archive.org/web/20221203093533/https://mahotas.readthedocs.io/) 是另一个为 [**生物图像信息学**](https://web.archive.org/web/20221203093533/http://en.wikipedia.org/wiki/Bioimage_informatics) 设计的图像处理和计算机视觉库。它在 NumPy 数组中读写图像，用 C++实现，具有流畅的 python 接口。

Mahotas 最受欢迎的功能是

让我们看看如何使用 [**Mahotas**](https://web.archive.org/web/20221203093533/https://mahotas.readthedocs.io/) 进行模板匹配，以便**找到 wally。**

下面的代码片段有助于在人群中找到 Wally。

```py
from pylab import imshow, show
import mahotas
import mahotas.demos
import numpy as np

wally = mahotas.demos.load('Wally')
wfloat = wally.astype(float)

r,g,b = wfloat.transpose((2,0,1))
w = wfloat.mean(2)
pattern = np.ones((24,16), float)

for i in range(2):
    pattern[i::4] = -1
    v = mahotas.convolve(r-w, pattern)
    mask = (v == v.max())
    mask = mahotas.dilate(mask, np.ones((48,24)))
    np.subtract(wally, .8*wally * ~mask[:,:,None], out=wally, casting='unsafe')
imshow(wally)
show()
```

## 7.SimpleITK

[ITK](https://web.archive.org/web/20221203093533/https://itk.org/) 或 **Insight 分割和配准工具包**是一个开源平台，广泛用于图像分割和图像配准(一个叠加两个或更多图像的过程)。

### **图像分割**

ITK 使用了 [**CMake**](https://web.archive.org/web/20221203093533/https://en.wikipedia.org/wiki/CMake) 构建环境，库是用 Python 包装的 C++实现的。

你可以检查这个 [Jupyter 笔记本](https://web.archive.org/web/20221203093533/http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/)用于学习和研究。

## 8.Pgmagick

Pgmagick 是 Python 的一个[**GraphicsMagick**](https://web.archive.org/web/20221203093533/http://www.graphicsmagick.org/)**绑定，它提供了对图像执行调整大小、旋转、锐化、渐变图像、绘制文本等操作的工具。**

 **### **模糊图像**

```py
from pgmagick.api import Image

img = Image('leena.jpeg')

img.blur(10, 5)
```

### **图像的缩放**

```py
from pgmagick.api import Image

img = Image('leena.png')

img.scale((150, 100), 'leena_scaled')

```

要了解更多信息，你可以查看 Jupyter 笔记本的精选列表[这里](https://web.archive.org/web/20221203093533/https://github.com/hhatto/pgmagick)。

## 最后的想法

我们已经介绍了用于机器学习的前 8 个图像处理库。希望您现在已经知道哪一个最适合您的项目。祝你好运。🙂****