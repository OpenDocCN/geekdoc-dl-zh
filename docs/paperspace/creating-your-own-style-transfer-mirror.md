# 用渐变和 ml5.js 创建你自己的风格转移镜像

> 原文：<https://blog.paperspace.com/creating-your-own-style-transfer-mirror/>

在这篇文章中，我们将学习如何用 Paperspace 的[渐变](https://www.paperspace.com/gradient)训练一个风格转移网络，并使用 [ml5.js](https://ml5js.org/) 中的模型来创建一个交互式风格转移镜像。这篇文章是一系列博客文章的第二篇，这些文章致力于在 Paperspace 中训练机器学习模型，然后在 [ml5.js](https://ml5js.org/) 中使用它们。你可以在这里阅读关于如何训练 LSTM 网络生成文本的第一篇文章。

## 风格转移

风格转换是以其他图像的风格重新组合图像的技术。 [1](http://genekogan.com/works/style-transfer/)
它最早出现在 2015 年 9 月，当时 Gatys et。al 发表论文[一种艺术风格的神经算法](https://arxiv.org/abs/1508.06576)。在这篇论文中，研究人员展示了深度神经网络，特别是卷积神经网络，如何开发和提取图像风格的表示，并将这种表示存储在特征图中。想法是然后使用学习的样式表示并且把它应用到另一个图像。更具体地说:

> 该系统使用神经表示来分离和重组任意图像的内容和风格，为艺术图像的创建提供神经算法。[...]我们的工作提供了一条通往算法理解人类如何创造和感知艺术意象的道路 [2](https://arxiv.org/pdf/1508.06576.pdf)

基本上，你训练一个深度神经网络来提取图像风格的表示。然后，您可以将这个样式应用到内容图像(C)中，并创建一个新的图像(C[S] ),它包含 C 的内容，但是样式是 S。al 出版，其他类似的方法和优化也出版了。[实时风格转换和超分辨率的感知损失](https://cs.stanford.edu/people/jcjohns/eccv16/)Johnson 等人。艾儿希多推出了优化工艺的新方法，速度快了三个数量级 [3](https://cs.stanford.edu/people/jcjohns/eccv16/) ，并且拥有高分辨率的图像。(你可以了解更多关于网络在传递风格时所做的技术细节，[这里](https://shafeentejani.github.io/2016-12-27/style-transfer/)，[这里](https://arxiv.org/pdf/1508.06576.pdf)和[这个](https://blog.paperspace.com/art-with-neural-networks/)以前的 Paperspace 帖子。)

<video autoplay="" loop="" controls=""><source src="https://github.com/genekogan/genekogan.github.io/raw/mastimg/style-transfer/picasso-periods.mp4" type="video/mp4"> <source src="https://github.com/genekogan/genekogan.github.io/raw/mastimg/style-transfer/picasso-periods.webm" type="video/webm;codecs=&quot;vp8, vorbis&quot;"> 

巴勃罗·毕加索于 1937 年在玻璃上绘画，分别由他的蓝色、非洲和立体主义时期的作品重新设计。吉恩·科岗。

## 浏览器中的样式转换镜像

在本教程中，我们将训练一个模型来捕捉和学习你想要的任何图像的风格。然后，我们将在浏览器中使用这个模型，用 [ml5.js](https://ml5js.org/) 创建一个交互式镜像，它将使用网络摄像头，并在捕获的图像上应用实时风格传输。以下是智利艺术家[博罗罗](https://en.wikipedia.org/wiki/Carlos_Maturana)于 1993 年使用[通昆 Chungungo Pate 工厂制作的最终效果演示(请允许并启用您的网络摄像头):](https://www.artsy.net/artwork/francis-picabia-udnie-jeune-fille-americaine-danse-udnie-young-american-girl-dance)

[https://paperspace.github.io/training_styletransfer/ml5js_example/](https://paperspace.github.io/training_styletransfer/ml5js_example/)

多亏了 [ml5.js](https://ml5js.org/) ，我们可以在浏览器上完整地运行这个模型。如果你没有读过[的上一篇文章](https://blog.paperspace.com/training-an-lstm-and-using-the-model-in-ml5-js/)， [ml5.js](https://ml5js.org/) 是一个新的 JavaScript 库，旨在让机器学习对艺术家、创意程序员和学生等广大受众变得触手可及。该库在浏览器中提供对机器学习算法和模型的访问，构建在 [TensorFlow.js](https://js.tensorflow.org/) 之上，没有其他外部依赖。

因此，我们将使用 Gradient 的 GPU 加速在 Python 中训练一个模型，将模型导出到 JavaScript，并使用 [ml5.styleTransfer()](https://ml5js.org/docs/StyleTransfer) 方法在浏览器上运行所有内容。

## 安装

您可以在这个库的[中找到这个项目的代码。这个代码基于 github.com/lengstrom/fast-style-transfer](https://github.com/Paperspace/training_styletransfer)的[，它结合了 Gatys 的艺术风格的神经算法、Johnson 的实时风格转换和超分辨率的感知损失以及 Ulyanov 的实例归一化。](https://github.com/lengstrom/fast-style-transfer)

训练该算法需要访问 [COCO 数据集](http://cocodataset.org/#home)。COCO 是一个大规模的对象检测、分割和字幕数据集。我们将使用的数据集版本总共约为 15GB。幸运的是，Paperspace 有[公共数据集](https://paperspace.zendesk.com/hc/en-us/articles/360003092514-Public-Datasets)，你可以从你的工作中访问，所以没有必要下载。公共数据集会自动装载到您的作业和笔记本的只读`/datasets`目录下。

### 安装图纸空间节点 API

我们将使用[纸空间节点 API](https://paperspace.github.io/paperspace-node/) 或 [Python API](https://github.com/Paperspace/paperspace-python) 。如果您没有安装它，您可以使用 npm 轻松安装它:

```py
npm install -g paperspace-node 
```

或者使用 Python:

```py
pip install paperspace 
```

(如果你愿意，也可以从 [GitHub 发布页面](https://github.com/Paperspace/paperspace-node/releases)安装二进制文件)。

创建 Paperspace 帐户后，您将能够从命令行使用您的凭据登录:

```py
paperspace login 
```

出现提示时，添加您的 Paperspace 电子邮件和密码。

如果您还没有 Paperspace 帐户，您可以使用此链接免费获得 5 美元！链接:`https://www.paperspace.com/&R=VZTQGMT`

## 培训说明

### 1)克隆存储库

从克隆或下载项目存储库开始:

```py
git clone https://github.com/Paperspace/training_styletransfer.git
cd training_styletransfer 
```

这将是我们项目的开始。

### 2)选择一个样式图像

将您想要训练风格的图像放在`/images`文件夹中。

### 3)在渐变上运行你的代码

在资源库`root`中，您会发现一个名为`run.sh`的文件。这是一个脚本，其中包含我们将使用 run 来训练我们的模型的指令。

打开`run.sh`并修改`--style`参数以指向您的图像:

```py
python style.py --style images/YOURIMAGE.jpg \
  --checkpoint-dir checkpoints/ \
  --vgg-path /styletransfer/data/imagenet-vgg-verydeep-19.mat \
  --train-path /datasets/coco/ \
  --model-dir /artifacts \
  --test images/violetaparra.jpg \
  --test-dir tests/ \
  --content-weight 1.5e1 \
  --checkpoint-iterations 1000 \
  --batch-size 20 
```

`--style`应该指向您想要使用的图像。`--model-dir`将是保存 ml5.js 模型的文件夹。 `--test`是一个图像，将用于测试每个时期的进程。您可以在此处的[和此处](https://github.com/lengstrom/fast-style-transfer#documentation)的[代码的原始存储库中了解更多关于如何使用所有参数进行培训的信息。](https://github.com/lengstrom/fast-style-transfer/blob/master/docs.md)

现在我们可以开始训练了！类型:

```py
paperspace jobs create --container cvalenzuelab/styletransfer --machineType P5000 --command './run.sh' --project 'Style Transfer training' 
```

这意味着我们希望`create`一个新的`paperspace job`使用一个 Docker 镜像作为基础`container`，该镜像预装了我们需要的所有依赖项。我们还想使用一个`machineType P5000`，我们想运行`command` `./run.sh`来开始训练过程。这个`project`将被称为`Style Transfer training`

当培训过程开始时，您应该看到以下内容:

```py
Uploading styletransfer.zip [========================================] 1555381/bps 100% 0.0s
New jobId: jstj01ojrollcf
Cluster: PS Jobs
Job Pending
Waiting for job to run...
Job Running
Storage Region: East Coast (NY2)
Awaiting logs...

ml5.js Style Transfer Training!
Note: This traning will take a couple of hours.
Training is starting!... 
```

在 P5000 机器上训练该模型需要 2-3 个小时。如果你想让这个过程更快，你可以选择一个更好的 GPU。如果您愿意，也可以关闭终端，训练过程仍将继续，不会中断。如果您登录到[Paperspace.com](https://www.paperspace.com/)，在渐变选项卡下您可以检查模型的状态。您也可以通过键入以下命令来检查它:

```py
paperspace jobs logs --tail --jobId YOUR_JOB_ID 
```

### 4)下载模型

完成后，您应该会在日志中看到以下内容:

```py
Converting model to ml5js
Writing manifest to artifacts/manifest.json
Done! Checkpoint saved. Visit https://ml5js.org/docs/StyleTransfer for more information 
```

这意味着最终模型已准备好，放在您的作业的`/artifacts`文件夹中！
如果你去[Paperspace.com](https://www.paperspace.com/)，在渐变标签下检查你的工作，点击‘工件’你会看到一个名为`/model`的文件夹。这个文件夹包含了架构和所有从我们的模型中学习到的权重。它们被移植成一种 JSON 友好的格式，web 浏览器可以使用这种格式，ml5.js 也可以使用这种格式来加载它。单击右边的图标下载模型或键入:

```py
paperspace jobs artifactsGet --jobId YOUR_JOB_ID 
```

这将下载包含我们训练好的模型的文件夹。一定要下载`/ml5js_example/models`里面的模型。

现在我们准备在 [ml5.js](https://ml5js.org) 中试用我们的模型！

### 5)使用模型

在我们项目的`/root`中，你会发现一个名为`/ml5js_example`的文件夹。这个文件夹包含了一个非常简单的例子，说明如何在 ml5 中加载一个风格转换模型，并使用相机作为输入(它使用 [p5.js](https://p5js.org/) 来使这个过程更容易)。可以看原例子，代码[这里](https://ml5js.org/docs/style-transfer-webcam-example)。现在，您应该更改的行是这样的，在`/ml5js_example/sketch.js`的结尾:

```py
const style = new ml5.styleTransfer('./models/YOUR_NEW_MODEL'); 
```

`YOUR_NEW_MODEL`应该是你刚下载的型号名称。

我们几乎准备好测试模型了。剩下的唯一一件事就是启动一个服务器来查看我们的文件。如果您使用的是 Python 2:

```py
python -m SimpleHTTPServer 
```

如果您使用的是 Python 3:

```py
python -m http.server 
```

请访问 http://localhost:8000 ，如果一切顺利，您应该会看到演示:

[https://paperspace.github.io/training_styletransfer/ml5js_example/](https://paperspace.github.io/training_styletransfer/ml5js_example/)

### 关于训练和图像的一个注记

尝试不同的图像和超参数，发现不同的结果。避免带有大量图案的几何图形或图像，因为这些图像没有足够的网络可以学习的可识别特征。举个例子，这是用智利动态艺术家[马蒂尔德·佩雷斯](https://en.wikipedia.org/wiki/Matilde_P%C3%A9rez)的绢本印刷。

[https://paperspace.github.io/training_styletransfer/ml5js_example/geometric.html](https://paperspace.github.io/training_styletransfer/ml5js_example/geometric.html)

您会注意到结果不如上一个示例，因为输入图像主要由规则的几何图案重复组成，几乎没有独立的特征。

## 更多资源

*   [基因科岗式转移实验](http://genekogan.com/works/style-transfer/)
*   [艺术风格的神经算法](https://arxiv.org/pdf/1508.06576.pdf)
*   [用深度神经网络进行艺术风格转换](https://shafeentejani.github.io/2016-12-27/style-transfer/)
*   [ml5.js](https://ml5js.org)