# TensorFlow 对象检测 API:培训、评估和部署的最佳实践

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment>

本文是学习 TensorFlow 对象检测及其 API 的端到端工作流系列的第二部分。在第一篇文章中，您学习了[如何从头开始创建一个定制的对象检测器](/web/20221203101213/https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api)，但是仍然有很多事情需要您的注意才能真正精通。

我们将探索与我们已经经历过的模型创建过程同样重要的主题。以下是我们将要回答的一些问题:

*   如何评估我的模型，并对其性能进行评估？
*   我可以使用哪些工具来跟踪模型性能并比较多个实验的结果？
*   如何导出我的模型以在推理模式中使用它？
*   有没有一种方法可以进一步提升模型性能？

模型评估

## 很想知道我们的模特在野外会有怎样的表现。为了了解我们的模型在真实数据上的表现，我们需要做一些事情:

选择一组评估指标，

1.  获取单独的数据集来验证和测试您的模型，
2.  使用一组适当的参数启动评估流程。
3.  评估，步骤 1:指标

让我们从一组评估指标开始。你可能还记得在第一篇文章中，我们安装了一个名为 COCO API 的依赖项。

### 我们需要它来访问一组用于对象检测的有用指标:[平均精度和召回率](https://web.archive.org/web/20221203101213/https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52)。如果您不记得这些指标，您一定要阅读一下。计算机视觉工程师经常使用它们。

要使用平均精度和召回率，您应该配置 pipeline.config 文件。`eval_config`块中的`metrics_set`参数应设置为“coco_detection_metrics”。

这是该参数的默认选项，因此您很可能已经有了它。检查您的`eval_config`行是否像这样:

To use mean average precision and recall, you should configure your pipeline.config file. The `metrics_set` parameter in the `eval_config` block should be set to “coco_detection_metrics”.

当我们使用`metrics_set`中设置的“coco_detection_metrics”时，以下是可用的内容:

![](img/dd54454a20d9f9517891bc5e80141c3e.png)

*This is a place within the pipeline.config file where we specify metrics we want to use for evaluation*

您对这组指标的选择不限于精度和召回率。TensorFlow API 中还有一些其他选项。花点时间选择您想要的选项来跟踪您的特定模型。

![Mean average precision](img/f349e593d483ae8d5f83a2ff62018d07.png)

*Mean average precision (mAP) shown as a plot after we enable it for model validation. Note that mAP is calculated for different [IOU](https://web.archive.org/web/20221203101213/https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686) values. Average recall is not shown but also becomes available.*

评估，步骤 2:数据集

如果您仔细遵循了第一篇文章中的说明，那么数据集准备应该听起来很熟悉。
作为提醒，我们准备了模型评估需要的两个文件(validation.record 和 test.record)放在 Tensorflow/workspace/data 中。如果您的数据文件夹中的文件数量与下面的相同，那么您就可以开始下一步了！

以防你错过了。将文件记录在您的数据文件夹中，但仍想进行评估，以下是一些需要考虑的事项:

### `validation.record`需要在培训期间评估您的模型；

需要使用`test.record`来检查已经训练过的最终模型的性能。

```py
Tensorflow/
└─ cocoapi/
└─ ...
└─ workspace/
   └─ data/
      ├─ train.record 
      ├─ validation.record 
      ├─ test.record 
```

机器学习的传统方法需要 3 个独立的集合:用于训练、评估和测试。我强烈建议你遵循它，但是如果你有充分的理由避免这些，那么就只准备这些。记录与你的目的相关的文件。

*   评估，步骤 3:流程启动

如前所述，您可以在两个不同的时间戳进行模型评估:在训练期间或者在模型被训练之后。

培训期间的模型评估称为验证。TensorFlow 对象检测 API 的验证作业被视为一个独立的过程，应与培训作业并行启动。

### 并行启动时，验证作业将等待训练作业在模型训练过程中生成的检查点，并逐个使用这些检查点在单独的数据集上验证模型。

`validation.record`表示模型用于验证的独立数据集。`eval_confi`块中的`metrics_set`参数定义了一组评估指标。

为了启动验证作业，打开一个新的*终端*窗口，导航到 Tensorflow/workspace/，并启动以下命令:

其中:

**<配置文件的路径>** 是用于训练您想要评估的模型的配置文件的路径。应该是来自的配置文件。/models/ <文件夹中放着您选择的车型> /v1/，

**<模型目录路径>** 是评估作业写入日志(评估结果)的目录路径。我的建议是使用以下路径:。/models/ <文件夹中放着您选择的型号> /v1/。鉴于此，您的评估结果将放在培训日志旁边，

```py
python model_main_tf2.py
  --pipeline_config_path=<path to your config file>
  --model_dir=<path to a directory with your model>
  --checkpoint_dir=<path to a directory with checkpoints>
  --num_workers=<int for the number of workers to use>
  --sample_1_of_n_eval_examples=1

```

**<带有检查点的目录路径>** 是您的培训作业写入检查点的目录。也应该是下面的:。/models/ <文件夹中放着您选择的车型> /v1/，

*   **< int 表示要使用的工作线程数量>** 如果您有一个多核 CPU，该参数定义了可用于评估作业的内核数量。请记住，您的培训作业已经占用了您为其分配的内核数量。考虑到这一点，适当地设置用于评估的内核数量。
*   在执行上述命令后，您的评估工作将立即开始。与我们对培训作业所做的类似，如果您想要在 GPU 上进行评估，请在启动评估作业之前通过执行以下命令**来启用它:**
*   其中<gpu number="">定义了您想要使用的 GPU 的订单编号。请注意，订单计数从零开始。对于 CPU 上的验证，使用-1，如下面的命令所示:</gpu>
*   模型性能跟踪

模型性能跟踪简介

```py
export CUDA_VISIBLE_DEVICES= <GPU number>

```

在机器学习中，对于给定的任务，很难事先告诉你哪个模型会给你最好的结果。开发人员通常使用试错法来测试多个假设。

```py
export CUDA_VISIBLE_DEVICES=-1

```

您可以检查不同的模型架构，或者坚持使用一种架构，但尝试不同的参数设置。每一个配置都应该通过单独的培训工作启动来测试，因此跟踪和比较多个实验的工具就派上了用场。

## TensorFlow API 使用 tfevents 格式写入与模型性能相关的日志和优化器状态。您需要跟踪的 TF 事件主要有两个:**与培训相关的**和**与评估相关的**。

### 培训 tfevent 仅限于损失和学习率跟踪。它记录每个时期的步数，所以你可以看到你的训练工作进行得有多快。

当您为模型启动训练作业时，会自动记录此类实验元数据。日志存储在 tensor flow/workspace/models/<folder with="" the="" model="" of="" your="" choice="">/v1/train。当使用 [Tensorboard](https://web.archive.org/web/20221203101213/https://www.tensorflow.org/tensorboard) (我们马上会谈到)可视化时，它看起来是这样的:</folder>

请注意，您可以看到组件级别分解的损失(分别针对**分类**和**本地化**)，也可以看到**总计**值。当您面临一个问题，并且想要检查您的模型以找到问题的根本原因时，它变得特别有用。

如前所述，我们还可以跟踪**学习率**如何随时间变化，以及你的训练工作每秒完成多少**步**。

与培训 tfevent 类似，评估 tfevent 也包含一个具有相同细分的损失部分。除此之外，它还跟踪我们在之前谈到的 [**评估指标**。](https://web.archive.org/web/20221203101213/https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md)

跟踪工具

![Training tf-event](img/f6e7eb820fa3bca3ed334410a793fa40.png)

*Training tf-event (logs) visualized using Tensorboard*

有多种工具可以帮助您跟踪和比较与模型相关的日志。TensorFlow API 中已经内置的是 [Tensorboard](https://web.archive.org/web/20221203101213/https://www.tensorflow.org/tensorboard) 。

Tensorboard 比较好用。为了启动您的 TensorBoard，请打开一个*终端*窗口，导航至 tensor flow/workspace/models/<文件夹，其中包含您选择的型号> /目录。

在那里，使用以下命令启动 Tensorboard:

您可以向–logdir 传递一个文件夹路径，该文件夹包含多个实验的日志(例如:Tensorflow/workspace/models/)。

### 您还可以通过提供特定实验的日志路径来限制可访问的数据(例如:tensor flow/workspace/models/<folder with="" the="" model="" of="" your="" choice="">/)。</folder>

在任何情况下，Tensorboard 都会自动找到所有包含日志的目录，并使用这些数据来构建绘图。你可以在[官方指南](https://web.archive.org/web/20221203101213/https://www.tensorflow.org/tensorboard/get_started)中了解更多 Tensorboard 可以做的事情。

[Neptune.ai](https://web.archive.org/web/20221203101213/https://neptune.ai/) 是一款可供你考虑的替代追踪工具。与 Tensorboard 相比，它提供了更广泛的功能。以下是我发现特别方便的:

Neptune 完全兼容 tfevent (TensorBoard)格式。你所需要做的就是在你的*终端*窗口中启动一个[单命令行](https://web.archive.org/web/20221203101213/https://github.com/neptune-ai/neptune-tensorboard#overview)，

```py
tensorboard --logdir=<path to a directory with your experiment / experiments>
```

您可以只导入那些您认为重要的实验。它允许您过滤掉那些您想要从比较中排除的启动。考虑到这一点，您最终的仪表板将保持整洁，不会因为过多的实验而过载，

你的作品(笔记本、实验结果)可以通过一种非常简单的方式[与他人分享](https://web.archive.org/web/20221203101213/https://docs.neptune.ai/you-should-know/sharing-results-and-models-with-the-team)(只需发送一个链接)，

你可以追踪任何你想追踪的东西。当您还想要跟踪模型参数和/或它的工件时，它变得特别方便。您的硬件利用率也是可见的，所有这些都可以在一个位置获得。

模型导出

*   好了，你的模型现在已经训练好了，你对它的性能很满意，现在想用它来进行推理。让我告诉你怎么做。这将是一个两步走的过程:
*   1.第一步–模型导出。为此，您应该:
*   将导出脚本从 tensor flow/models/research/object _ detection/exporter _ main _ v2 . py
    复制粘贴到
    tensor flow/workspace/exporter _ main _ v2 . py，
*   在 Tensorflow/workspace 中，创建一个名为 exported_models 的新文件夹。这将是您放置所有导出模型的地方，

## 在 tensor flow/workspace/exported _ models 中创建一个子文件夹，用于存储特定的导出模型。将此文件夹命名为您在 tensor flow/workspace/models/<folder with="" the="" model="" of="" your="" choice="">中使用的名称<folder with="" the="" model="" of="" your="" choice="">，</folder></folder>

打开一个新的*终端*窗口，将 Tensorflow/workspace 作为当前工作目录，启动以下命令:

其中:

*   **<配置文件路径>** 是您想要导出的模式的配置文件路径。应该是来自的配置文件。/models/ <文件夹中有您选择的型号> /v1/
*   **<训练模型目录路径>** 是训练过程中放置模型检查点的目录路径。也应该是下面的:。/models/ <文件夹中有您选择的型号> /v1/
*   **<导出模型的目录路径>** 是保存导出模型的路径。应该是:。/exported_models/ <文件夹中有您选择的车型>

2.第二步——在推理模式下运行您的模型。

```py
python exporter_main_v2.py
  --pipeline_config_path=<path to a config file>
  --trained_checkpoint_dir=<path to a directory with your trained model>
  --output_directory=<path to a directory where to export a model>
  --input_type=image_tensor

```

为了方便你，我做了一个 [jupyter 笔记本](https://web.archive.org/web/20221203101213/https://ui.neptune.ai/anton-morgunov/tf-test/n/model-for-inference-36c9b0c4-8d20-4d5a-aa54-5240cc8ce764/6f67c0e3-283c-45de-ae56-405aecd736c0)，里面有你做推理所需的所有代码。你的目标是检查它，并为 **TODO** s 填充所有缺失的值

*   在 jupyter 笔记本中，你会发现两个可以根据你的目标使用的推理函数:`inference_with_plot`和`inference_as_raw_output`。
*   当您只想将模型输出可视化为绘制在输入图像对象上的边界框时，使用`inference_with_plot`。在这种情况下，函数输出将是如下图所示的图形:
*   或者，您可以使用`inference_as_raw_output`而不是绘图，返回一个包含 3 个键的*字典*:

在`detection_classes`键下，有一个包含所有被检测到的类的数组。类作为整数返回，

使用`detection_scores`(数组)查看每个检测类的检测置信度得分。

最后，`detection_boxes`是一个数组，包含每个检测到的对象的边界框的坐标。每个盒子有以下格式-*【y1，x1，y2，x2】*。左上角定义为 *y1* 和 *x1* ，而右下角定义为 *y2* 和 *x2* 。

模型改进的机会

*   在这一部分，我想和你分享一些很酷的方法，可以提升你的模型性能。我在这里的目标是向您提供 TensorFlow API 及其武库中可用内容的高级概述。我也会给你一个实现这些方法的直觉。我们开始吧！
*   图像预处理
*   你应该知道你给你的模型输入了什么。图像预处理在任何计算机视觉应用中都是至关重要的一步。

TensorFlow 在幕后执行图像标准化步骤(如果您喜欢，也可以称为标准化步骤，标准化和标准化之间的差异在这里有很好的描述),我们不能影响它。但是我们可以控制**如何**调整图像的大小，以及**将其调整到哪个尺寸**。
为了更好地理解 TensorFlow API 是如何做到这一点的，让我们来看一下 EfficientDet D-1 模型的 pipeline.config 代码片段:

## EfficientDet D-1 负责调整图像大小的默认方法是`keep_aspect_ratio_resizer`。

这个方法，如上面例子中的`min_dimension`和`max_dimension`参数所定义的，将把一个图像的较小边调整到 640 像素。另一边将被调整大小，以保持原来的长宽比。

存储为 true 将允许填充，这可能需要在调整大小时保持原始纵横比。

### 查看这个调整大小方法的输出很有趣。如果您的原始图像是矩形的，那么在调整大小时，您可能会得到一个被过度填充的图像。如果您通过自己选择的追踪工具进行检查，您的最终图像可能是这样的:

*使用 keep_aspect_ratio_resizer 方法时可能会出现的填充图像示例。|图片来源:[Jakub CIE lik](/web/20221203101213/https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection)T3【如何进行图像分割和对象检测的数据探索(我不得不艰难地学习的东西】*

我们绝对不想给我们的网络提供这样的图片。显然，它有太多无意义的信息被编码为黑色像素。我们怎样才能让它变得更好？我们可以使用不同的调整大小方法。

![image resizer](img/17356c2b931e190fc822691d88592048.png)

*Code snippet within pipeline.config file that defines image resizing step in EfficientDet D-1 model. *

在第一篇文章中，您了解了如何以高级方式进行参数调整。使用这种方法，您会发现 TensorFlow API 中还有其他调整大小的方法。

我们可能特别感兴趣的一个是`fixed_shape_resizer`，它将图像整形为由`height`和`width`参数定义的给定大小的矩形。

看看它在 pipeline.config 文件中的实现:

上图中有两件事值得你注意。

首先，从一种方法切换到另一种方法是多么容易:几行修改，没什么复杂的。

其次，您现在可以完全控制您的输入图像。尝试调整大小的方法和输入图像的大小有助于保留解决对象检测任务所必需的功能。

请记住，你的输入图像越小，网络就越难检测到物体！当您想要检测比原始图像尺寸小的对象时，这就成了一个问题。

图像放大

让我们继续探索与图像相关的方法，还有另一个改进的机会——[图像增强](/web/20221203101213/https://neptune.ai/blog/data-augmentation-in-python)。

![image resizer](img/08a3eaf737294fc909360a50ed2cf00b.png)

*Fixed_shape_resizer method implementation for EfficientDet D-1*

图像增强是一种对输入图像随机应用变换的方法，会在训练数据集中引入额外的方差。反过来，额外的方差导致更好的模型泛化，这对于良好的性能是必不可少的。

TensorFlow API 为我们提供了多种选择！让我们看一下 pipeline.config 文件，以了解增强的默认选项:

正如我们所看到的，有两个默认选项。您必须仔细检查您的问题域，并决定哪些增强选项与您的特定任务相关。

例如，如果你期望所有的输入图像总是在一个特定的方向，`random_horizontal_flip`将会伤害而不是帮助，因为它随机翻转输入图像。扔掉它，因为它与你的案子无关。将类似的逻辑应用于选择其他增强选项。

### 您可能对 TensorFlow API 中可用的其他选项感兴趣。为了方便起见，这里有一个到脚本的[链接，其中列出了所有的方法并做了很好的描述。](https://web.archive.org/web/20221203101213/https://github.com/tensorflow/models/blob/52bb4ab1d8dd42f033762b698a8acffc1b639387/research/object_detection/protos/preprocessor.proto)

值得一提的是，在任何会影响图像方向的变换(旋转、翻转、缩放等)的情况下，TensorFlow 不仅变换图像本身，还变换边界框的坐标。你没有必要为标签转化做任何事情。

锚点生成

图像中对象的边框形状是什么样的？它们大多是方形的还是长方形的？边界框有没有一个特定的长宽比能最好地捕捉到你感兴趣的对象？

![image augmentation options](img/6d511c5107aeb3784309ad929e8b4172.png)

*Default image augmentation options for EfficientDet D-1.*

您应该问自己这些问题，以使您的对象检测能够找到最适合您的对象的盒子。

这对于单阶段对象检测器(如 EfficientDet)变得特别方便，因为预设的锚集用于提出建议。

我们可以将锚点改为自定义数据集中对象的最佳形状吗？肯定的，是的！以下是 pipeline.config 文件中负责锚点设置的代码行:

![ image augmentation options in Tensorflow](img/ac1ac5a7a96ca2aa13b26d0cc5d4c3fc.png)

*List of options for image augmentation available in TensorFlow API*

有一个我们最感兴趣的参数，那就是`aspect_ratios`。它定义了矩形锚边的比率。

### 让我们以`aspect_ratios: 2.0`为例，这样你就能对它的工作原理有所了解。`2.0`值表示锚点的高度=其宽度的 2 倍。这种锚几何图形将最适合那些水平拉伸两倍于其垂直尺寸的对象。

如果我们的物体被水平拉伸 10 倍会怎样？让我们设置一个锚来捕捉这些物体:`aspect_ratios: 10.0`将完成这项工作。

相反，如果您的对象在垂直维度上被拉伸，请将`aspect_ratios`设置在 0 和 1 之间。介于 0 和 1 之间的值将定义锚的宽度比其高度小多少。你可以设置你想要多少锚。只要你觉得有意义就继续加`aspect_ratios`就好。

你甚至可以事先做好功课，为你的机器学习项目经历一个探索阶段，分析你的物体的几何形状。就我个人而言，我喜欢创建两个图来观察高宽比和高宽比的分布。这有助于我了解哪种纵横比最适合我的模型锚:

后处理和防止过拟合

![anchor generator](img/5057552b97b18d2fa58f78e78f62e6dc.png)

*Lines within pipeline.config futile that are responsible for a set of model’s anchors*

与预处理类似，后处理步骤也会影响模型的行为。物体探测器往往会产生数百个建议。大部分不会被录取，会被一些标准淘汰。

TensorFlow 允许您定义一组标准来控制模型建议。让我们看看 pipeline.config 文件中的代码片段:

有一种叫做[非最大抑制](https://web.archive.org/web/20221203101213/https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) (NMS)的方法用于 EfficientDet D-1 内的处理。该方法本身已被证明对绝大多数计算机视觉任务是成功的，所以我不会探索任何替代方法。

这里重要的是与`batch_non_max_suppression`方法一起使用的一组参数。这些参数很重要，可能会对模型的最终性能产生很大影响。让我们看看他们如何做到这一点:

`score_threshold`是一个参数，它定义了分类的最小置信度得分，应达到该得分，这样建议才不会被过滤掉。在默认配置中，它被设置为一个接近 0 的值，这意味着所有建议都被接受。这听起来像是一个合理的值吗？我的个人实践表明，最小过滤最终会给出更好的结果。消除那些最有可能不正确的建议导致更稳定的训练、更好的收敛和更低的过度拟合的机会。考虑将该参数至少设置为 0.2。当您的跟踪工具显示您的网络在评估集上的建议很差，或者/和您的评估指标没有随着时间的推移而改进时，这一点尤其重要；

![width-to-height ratio distribution](img/52c940109703675a56e8759676132cd0.png)

*Example for width-to-height ratio distribution that I plot when looking for the best shape for my anchors.*

### `iou_threshold`是一个参数，让 NMS 对重叠的盒子进行适当的过滤。如果您的模型为对象生成重叠的框，请考虑降低该分数。如果你的图像上有密集分布的物体，考虑增加这个参数；

从名字上来看很简单。你希望每个类有多少个对象？几个，十几个，还是几百个？这个参数将帮助你的网络了解这一点。我在这里的建议是将这个值设置为等于单个类的最大对象数乘以你拥有的 anchors 数(number of `aspect_ratios`);

`max_total_detections`应设置为`max_detections_per_class` *班级总数。将`max_number_of_boxes`设置为与`max_total_detections`相同的数字也很重要。`max_number_of_boxes`位于 pipeline.config 文件的`train_config`部分。

![post processing](img/9b4f67ee0182fd782c0e95fbc85cdc43.png)

*Piece of code (default values are kept) that defines post-processing parameters for*
*EfficientDet D-1*

给定上述设置参数的方法，您将让您的模型知道预期有多少对象以及它们的密度是多少。这将导致更好的最终性能，也将降低过度拟合的机会。

既然我们已经谈到过拟合问题，我也将分享另一个消除它的常用工具——[dropout](https://web.archive.org/web/20221203101213/https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)层，它是这样实现的:

*   Dropout 实现强制您的模型寻找那些最能描述您想要检测的对象的特征。它有助于提高泛化能力。更好的泛化有助于模型更好地抵抗过度拟合。
*   最后但并非最不重要的一点是，您可以通过先进的学习速率控制方法来避免过度拟合并获得更好的模型性能。具体来说，我们感兴趣的是如何推动我们的训练工作，为给定的损失函数找到真正的全局最小值。
*   学习率计划对这一目标至关重要。让我们看看 TensorFlow 在 EfficientDet D-1 的默认配置中为学习率调度提供了什么:
*   [余弦学习率衰减](https://web.archive.org/web/20221203101213/https://medium.com/@scorrea92/cosine-learning-rate-decay-e8b50aa455b)是一个伟大的调度程序，允许你的学习率在整个训练时间内增长和减少。

为什么这种调度方法可以给你更好的模型性能和防止过度拟合？出于几个原因:

以较低的学习率开始可以让您在训练模型的最开始就控制渐变。我们不希望它们变得非常大，所以原始模型的权重不会发生剧烈变化。请记住，我们在自定义数据集上微调我们的模型，没有必要改变神经网络已经学习的低级特征。对于我们的模型，它们很可能保持不变；

![box predictor](img/99c2716c1923cf503489e49915ebbd72.png)

*Dropout with probability = 0.2 set for box_predictor net within EfficientDet D-1*

学习率的初始增加将有助于你的模型有足够的能力不陷入局部最小值，并能够摆脱它；

![dropout layer](img/2c662205da6bdb3b9c31c0dea160d021.png)

*Illustration for a dropout layer (with probability = 0.5) implemented within a simple neural net. | Source: [primo.ai](https://web.archive.org/web/20221203101213/http://primo.ai/index.php?title=Dropout)*

随着时间的推移，平滑的学习率衰减将导致稳定的训练，并且还将让您的模型找到最适合您的数据的可能。

你现在确信学习率计划很重要吗？如果是，下面是正确配置的方法:

![learning rate](img/b0a385dc6f5d9608e9e9677d8b2c69ab.png)

*Learning rate scheduler implementation in a default configuration for EfficientDet D-1*

`learning_rate_base`是您的模型开始训练的初始学习率；

`total_steps`定义你的模型将要训练的总步数。请记住，在培训工作的最后阶段，学习率计划程序将使学习率值接近于零；

*   `warmup_learning_rate`是学习率开始下降前将达到的最大值；
*   `warmup_steps`定义将学习率从`learning_rate_base`提高到`warmup_learning_rate`的步数
*   损失函数操作

您可能遇到过这样的情况:您的模型在定位对象方面表现出色，但在分类方面表现很差。相比之下，分类可能非常好，但对象定位可能更好。

*   当对象检测器被包括到服务流水线中时，这变得尤其重要，其中每个服务都是机器学习模型。在这种情况下，每个模型的输出都应该足够好，以便后续模型将其作为输入进行消化。
*   请这样想:您试图检测图像上的所有文本片段，以便将每个文本片段传递给下一个 [OCR](https://web.archive.org/web/20221203101213/https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/) 模型。如果您的模型检测到所有文本，但有时由于本地化不佳而截断文本，该怎么办？
*   这对于后面的 OCR 来说是个问题，因为它无法读取整个文本。OCR 将能够处理一段剪切的文本，但它的输出对我们来说将毫无意义。我们怎么能这样做呢？
*   TensorFlow 为您提供了一个选项，通过损失函数中的权重来确定对您来说重要的事情的优先级。看看这段代码:

### 您可以更改这些参数的值，为对您最重要的内容赋予更高的权重。或者，您可以降低总损失中特定零件的值。这两种方法最终完成了相同的工作。

如果你决定改变权重值，我个人的建议是从 0.1-0.3 之间的值开始增加权重。更大的值可能会导致严重的不平衡。

结论

您对 TensorFlow API 的熟练程度已达到一个新的水平。你现在可以完全控制你的实验，并且知道如何评估和比较它们，所以只有最好的才会投入生产！

您还熟悉如何将您的模型转移到生产中。您知道如何导出模型，并拥有执行推理所需的所有代码。

希望您现在已经有了进一步改进模型的机会的感觉。试试看。当您看到您的指标增长时，您会喜欢它的。对你的假设设定要有创造性，不要害怕尝试新的想法。也许你的下一个配置会为我们所有人树立一个标杆！谁知道呢？

![weights](img/b627b936c6a1818efc09564539f5d133.png)

*Initial set up for weights within loss function. Equal values for classification and localization.*

下次见！

If you decide to play around with weight values, my personal recommendation would be to start incrementing weights by values around between [0.1-0.3]. Bigger values might lead to a significant imbalance.

## Conclusions

Your proficiency with the TensorFlow API has reached a new level. You’re now in full control of your experiments and know how to evaluate and compare them, so only the best will go to production!

You’re also familiar with how to move your model to production. You know how to export a model and have all code necessary to perform inference. 

Hopefully, you now have a feeling of what your opportunities are for further model improvement. Give it a shot. You’ll love it when you see your metrics grow. Be creative with your hypothesis setting, and don’t be afraid to try new ideas. Maybe your next configuration will set a benchmark for all of us! Who knows?

See you next time!