# 如何在 PyTorch 中从头开始实现 YOLO (v3)对象检测器:第 2 部分

> 原文：<https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/>

图片来源:凯罗尔·马杰克。查看他的 YOLO v3 实时检测视频[这里](https://www.youtube.com/watch?v=8jfscFuP9k)

这是从头开始实现 YOLO v3 检测器教程的第 2 部分。在最后一部分，我解释了 YOLO 是如何工作的，在这一部分，我们将在 PyTorch 中实现 YOLO 使用的层。换句话说，这是我们创建模型的构建模块的部分。

本教程的代码旨在运行在 Python 3.5 和 PyTorch **0.4** 上。在这个 [Github repo](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch) 可以找到它的全部内容。

本教程分为 5 个部分:

1.  第一部分:了解 YOLO 是如何运作的

2.  第 2 部分(这一部分):创建网络体系结构的各层

3.  [第三部分:实现网络的前向传递](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/)

4.  [第 4 部分:目标置信度阈值和非最大值抑制](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/)

5.  [第五部分:设计输入和输出管道](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-5/)

#### 先决条件

*   教程的第一部分/YOLO 如何工作的知识。
*   PyTorch 的基本工作知识，包括如何使用`nn.Module`、`nn.Sequential`和`torch.nn.parameter`类创建定制架构。

我想你以前对 PyTorch 有过一些经验。如果你刚刚开始，我建议你在回到这篇文章之前先尝试一下这个框架。

#### 入门指南

首先创建一个目录，用于存放检测器的代码。

然后，创建一个文件`darknet.py`。 **Darknet 是 YOLO** 底层架构的名字。该文件将包含创建 YOLO 网络的代码。我们将用一个名为`util.py`的文件来补充它，该文件将包含各种辅助函数的代码。将这两个文件保存在检测器文件夹中。您可以使用 git 来跟踪更改。

#### 配置文件

官方代码(用 C 编写)使用一个配置文件来构建网络。 *cfg* 文件逐块描述了网络的布局。如果你来自咖啡馆背景，它相当于用来描述网络的`.protxt`文件。

我们将使用作者发布的官方 *cfg* 文件来构建我们的网络。从[这里](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)下载它，并把它放在探测器目录下一个名为`cfg`的文件夹中。如果您使用的是 Linux，`cd`进入您的网络目录并键入:

```py
mkdir cfg
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg 
```

如果您打开配置文件，您会看到类似这样的内容。

```py
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear 
```

我们看到上面有 4 个街区。其中，3 层描述卷积层，随后是一个*快捷方式*层。一个*快捷方式*层是一个跳过连接，就像 ResNet 中使用的那样。YOLO 使用 5 种类型的图层:

**卷积**

```py
[convolutional]
batch_normalize=1  
filters=64  
size=3  
stride=1  
pad=1  
activation=leaky 
```

**快捷方式**

```py
[shortcut]
from=-3  
activation=linear 
```

一个*快捷方式*层是一个跳过连接，类似于 ResNet 中使用的那个。`from`参数为`-3`，表示快捷层的输出是由*快捷层*向后**加上前一层和第三层的**特征图得到的。

**上采样**

```py
[upsample]
stride=2 
```

使用双线性上采样以因子`stride`对前一层中的特征地图进行上采样。

**路线**

```py
[route]
layers = -4

[route]
layers = -1, 61 
```

*路线*层值得解释一下。它有一个属性`layers`，可以有一个或两个值。

当`layers`属性只有一个值时，它输出由该值索引的图层的要素地图。在我们的示例中，它是-4，因此该图层将从第四层开始从*路线*图层向后输出要素地图。

当`layers`有两个值时，它返回按其值索引的图层的连接要素地图。在我们的示例中，它是-1，61，该图层将输出前一图层(-1)和第 61 个图层的要素地图，沿深度维度连接。

**YOLO**

```py
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1 
```

YOLO 层对应于第 1 部分中描述的检测层。`anchors`描述了 9 个锚，但是仅使用由`mask`标签的属性索引的锚。在这里，`mask`的值是 0，1，2，这意味着使用第一、第二和第三锚。这是有意义的，因为检测层的每个单元预测 3 个盒子。总的来说，我们有 3 个比例的探测层，总共有 9 个锚。

**网**

```py
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1 
```

cfg 中还有另一种称为`net`的块，但我不会称之为层，因为它只描述关于网络输入和训练参数的信息。在 YOLO 的向前传球中没有使用它。然而，它确实为我们提供了像网络输入大小这样的信息，我们用它来调整前向传递中的锚。

#### 解析配置文件

在我们开始之前，在`darknet.py`文件的顶部添加必要的导入。

```py
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
```

我们定义了一个名为`parse_cfg`的函数，它将配置文件的路径作为输入。

```py
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """ 
```

这里的想法是解析 cfg，**将每个块存储为一个字典**。块的属性和它们的值作为键值对存储在字典中。当我们解析 cfg 时，我们不断地将这些字典(在代码中用变量`block`表示)添加到一个列表`blocks`中。我们的函数将返回这个块。

我们首先将 cfg 文件的内容保存在一个字符串列表中。下面的代码对这个列表执行一些预处理。

```py
file = open(cfgfile, 'r')
lines = file.read().split('\n')                        # store the lines in a list
lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
lines = [x for x in lines if x[0] != '#']              # get rid of comments
lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces 
```

然后，我们对结果列表进行循环以获得块。

```py
block = {}
blocks = []

for line in lines:
    if line[0] == "[":               # This marks the start of a new block
        if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
            blocks.append(block)     # add it the blocks list
            block = {}               # re-init the block
        block["type"] = line[1:-1].rstrip()     
    else:
        key,value = line.split("=") 
        block[key.rstrip()] = value.lstrip()
blocks.append(block)

return blocks 
```

#### 创建构建模块

现在我们将使用上面的`parse_cfg`返回的列表为配置文件中的块构建 PyTorch 模块。

我们在列表中有 5 种类型的层(如上所述)。PyTorch 为类型`convolutional`和`upsample`提供了预构建的层。我们将不得不通过扩展`nn.Module`类来为其余的层编写我们自己的模块。

`create_modules`函数获取一个由`parse_cfg`函数返回的列表`blocks`。

```py
def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = [] 
```

在我们迭代块列表之前，我们定义一个变量`net_info`来存储关于网络的信息。

###### nn。模块列表

我们的函数将返回一个`nn.ModuleList`。这个类几乎就像一个包含`nn.Module`对象的普通列表。然而，当我们将`nn.ModuleList`添加为`nn.Module`对象的成员时(即当我们将模块添加到我们的网络中时)，`nn.ModuleList`内的`nn.Module`对象(模块)的所有`parameter`也被添加为`nn.Module`对象的`parameter`(即我们的网络，我们将`nn.ModuleList`添加为其成员)。

当我们定义一个新的卷积层时，我们必须定义它的核的维数。虽然内核的高度和宽度是由 cfg 文件提供的，但内核的深度恰恰是前一层中存在的过滤器的数量(或特征映射的深度)。这意味着我们需要**跟踪应用卷积层的层中的滤波器数量**。我们使用变量`prev_filter`来做这件事。我们将其初始化为 3，因为图像有 3 个对应于 RGB 通道的滤镜。

路径图层从以前的图层中引入(可能串联)要素地图。如果在路径图层的正前方有一个卷积图层，则内核将应用于之前图层的要素地图，准确地说是路径图层带来的要素地图。因此，我们不仅需要跟踪前一层中的过滤器数量，还需要跟踪前一层中的每个的数量。当我们迭代时，我们将每个块的输出过滤器的数量添加到列表`output_filters`中。

现在，我们的想法是遍历块列表，并为每个块创建一个 PyTorch 模块。

```py
 for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list 
```

`nn.Sequential`类用于顺序执行多个`nn.Module`对象。如果您查看 cfg，您会发现一个块可能包含多个层。例如，`convolutional`类型的块除了卷积层之外，还有批量范数层以及泄漏 ReLU 激活层。我们使用`nn.Sequential`将这些层串在一起，这就是`add_module`函数。例如，我们就是这样创建卷积层和上采样层的。

```py
 if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample) 
```

###### 路径层/快捷方式层

接下来，我们编写代码来创建*路线*和*快捷方式*层。

```py
 #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut) 
```

创建路径层的代码值得好好解释一下。首先，我们提取`layers`属性的值，将其转换为整数并存储在一个列表中。

然后，我们有一个新的层称为`EmptyLayer`，顾名思义，这只是一个空层。

```py
route = EmptyLayer() 
```

它被定义为。

```py
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__() 
```

###### 等等，一个空层？

现在，一个空层可能看起来很奇怪，因为它什么也不做。路由层就像任何其他层一样执行操作(向前移动上一层/连接)。在 PyTorch 中，当我们定义一个新的层时，我们子类化`nn.Module`并在`nn.Module`对象的`forward`函数中编写该层执行的操作。

为了设计路由块的层，我们必须构建一个用属性`layers`的值初始化的`nn.Module`对象作为它的成员。然后，我们可以编写代码在`forward`函数中连接/提出特征地图。最后，我们在网络的第`forward`功能中执行这一层。

但是考虑到连接的代码相当短且简单(在特性图上调用`torch.cat`),如上设计一个层将导致不必要的抽象，这只会增加锅炉板代码。相反，我们可以做的是放置一个虚拟层来代替建议的路由层，然后在代表暗网的`nn.Module`对象的`forward`函数中直接执行连接。(如果最后一行对你来说没有太大意义，建议你去看看`nn.Module`类在 PyTorch 中是如何使用的。底部链接)

路径层前面的卷积层将其内核应用于(可能连接)来自前一层的要素地图。以下代码更新了变量`filters`以保存路径图层输出的过滤器数量。

```py
if end < 0:
    #If we are concatenating maps
    filters = output_filters[index + start] + output_filters[index + end]
else:
    filters= output_filters[index + start] 
```

快捷层也利用空层，因为它也执行非常简单的操作(加法)。不需要更新变量`filters`,因为它只是将前一层的特征图添加到后一层的特征图中。

###### YOLO 层

最后，我们编写创建 *YOLO* 层的代码。

```py
 #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection) 
```

我们定义了一个新的层`DetectionLayer`，它包含了用来检测边界框的锚点。

检测层定义为

```py
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors 
```

在循环的最后，我们做一些簿记工作。

```py
 module_list.append(module)
        prev_filters = filters
        output_filters.append(filters) 
```

这就结束了循环体。在函数`create_modules`的末尾，我们返回一个包含`net_info`和`module_list`的元组。

```py
return (net_info, module_list) 
```

#### 测试代码

您可以通过在`darknet.py`的末尾键入以下几行并运行该文件来测试您的代码。

```py
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks)) 
```

您将看到一个很长的列表，(正好包含 106 个条目)，其元素如下所示

```py
.
.

  (9): Sequential(
     (conv_9): Conv2d (128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
     (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
     (leaky_9): LeakyReLU(0.1, inplace)
   )
   (10): Sequential(
     (conv_10): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
     (leaky_10): LeakyReLU(0.1, inplace)
   )
   (11): Sequential(
     (shortcut_11): EmptyLayer(
     )
   )
.
.
. 
```

这部分到此为止。在接下来的[部分](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/)中，我们将组装已经创建的构建模块，以从图像中产生输出。

#### 进一步阅读

1.  [PyTorch 教程](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
2.  [nn。模块，nn。参数类别](http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network)
3.  [nn。ModuleList 和 nn。顺序](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463)

Ayoosh Kathuria 目前在印度国防研究与发展组织实习，致力于改进粒状视频中的物体检测。当他不工作的时候，他不是在睡觉就是在用吉他弹奏平克·弗洛伊德。你可以在 [LinkedIn](https://www.linkedin.com/in/ayoosh-kathuria-44a319132/) 上和他联系，或者在[GitHub](https://github.com/ayooshkathuria)T5 上看看他做了些什么