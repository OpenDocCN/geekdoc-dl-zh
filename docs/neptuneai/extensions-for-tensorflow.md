# TensorFlow 的最佳 ML 框架和扩展

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/extensions-for-tensorflow>

TensorFlow 有一个庞大的库和扩展生态系统。如果您是开发人员，您可以轻松地将它们添加到您的 ML 工作中，而不必构建新的函数。

在本文中，我们将探索一些可以立即开始使用的 TensorFlow 扩展。

首先，让我们从 TensorFlow Hub 查看特定领域的预训练模型。

我们开始吧！

TensorFlow Hub 是一个存储库，包含数百个经过训练的现成模型。您可以找到以下型号:

*   自然语言处理
*   目标检测
*   图像分类
*   风格转移
*   视频动作检测
*   声音分类
*   音高识别

要使用一个模型，首先需要在 [tfhub.dev](https://web.archive.org/web/20221206064514/https://tfhub.dev/) 中识别它。你需要检查它的文档。例如，下面是加载这个 [ImageNet 分类模型](https://web.archive.org/web/20221206064514/https://tfhub.dev/google/imagenet/inception_v1/classification/4)的说明。

```py
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4")
])

```

模型可以按原样使用，也可以进行微调。该模型的文档提供了如何做到这一点的说明。

例如，我们可以通过将' trainable = True '传递给' hub.kerasLayer '来微调上面的模型。

```py
hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4",
               trainable=True, arguments=dict(batch_norm_momentum=0.997))
```

这是一组工具，您可以使用它们来优化模型的执行和部署。

为什么这很重要？

*   它减少了模型在移动设备上的延迟，
*   它降低了云的成本，因为模型变得足够小，可以部署边缘设备。

优化模型可能会导致精确度降低。根据问题的不同，您需要决定一个稍微不太精确的模型是否值得利用模型优化的优势。

优化可以应用于来自 tfhub.dev 的预训练模型，以及您自己训练的模型。也可以从 tfhub.dev 下载优化的模型。

模型优化的技术之一是修剪。在这种技术中，权重张量中不必要的值被消除。这会产生更小的模型，精度非常接近基线模型。

修剪模型的第一步是定义修剪参数。

设置 50%的稀疏度意味着 50%的权重将被置零。“修剪时间表”负责在训练期间控制[修剪](https://web.archive.org/web/20221206064514/https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)。

```py
from tensorflow_model_optimization.sparsity.keras import ConstantSparsity
pruning_params = {
    'pruning_schedule': ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

```

之后，您可以使用上述参数修剪整个模型。

```py
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
model_to_prune = prune_low_magnitude(
    keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='relu')
    ]), **pruning_params)
```

一种替代方案是使用量化感知训练，该训练使用较低精度，例如 8 位而不是 32 位浮点。

```py
import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

```

在这一点上，你将有一个可以感知量化的模型，但是还没有量化。

编译和定型模型后，可以使用 TFLite 转换器创建量化模型。

```py
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
```

你也可以量化模型的某些层。

另一种模型优化策略是[权重聚类](https://web.archive.org/web/20221206064514/https://www.tensorflow.org/model_optimization/guide/clustering/)。在这种技术中，减少了唯一权重值的数量。

TensorFlow 推荐器(TFRS)是一个用于构建推荐系统模型的库。

您可以使用它来准备数据、制定模型、培训、评估和部署。这本[笔记本](https://web.archive.org/web/20221206064514/https://colab.research.google.com/github/tensorflow/recommenders/blob/main/docs/examples/quickstart.ipynb)包含了如何使用 TFRS 的完整示例。

TensorFlow Federated (TFF)是一个基于分散数据的机器学习开源库。在联合学习中，设备可以从共享模型中协作学习。

该模型将在服务器上使用代理数据进行训练。然后，每个设备将下载该模型，并使用该设备上的数据对其进行改进。

这种方法的好处是敏感的用户数据永远不会上传到服务器。一种已经被使用的方法是在[电话键盘](https://web.archive.org/web/20221206064514/https://arxiv.org/abs/1811.03604)中。

TensorFlow Federated 由两层组成:

*   联邦学习(FL) API
*   联邦核心(FC) API

使用联邦学习(FL) API，开发人员可以在现有的 TensorFlow 模型上应用联邦训练和评估。

联邦核心(FC) API 是一个用于编写联邦算法的低级接口系统。

如果你感兴趣，请查看官方 [TensorFlow 联合教程](https://web.archive.org/web/20221206064514/https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)以了解更多信息。

要构建更有效的神经网络架构，您可以插入可区分的图形层。

对神经网络的几何先验和约束进行建模导致可以更鲁棒和有效地训练的架构。

计算机图形学和计算机视觉的结合让我们可以在机器学习问题中使用未标记的数据。Tensorflow Graphics 提供了一套可区分的图形、几何图层和 3D 查看器功能。

这里有一个来自官方文档的代码片段产生的输出的例子。

```py
import numpy as np
import tensorflow as tf
import trimesh

import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.notebooks import threejs_visualization

!wget https://storage.googleapis.com/tensorflow-graphics/notebooks/index/cow.obj

mesh = trimesh.load("cow.obj")
mesh = {"vertices": mesh.vertices, "faces": mesh.faces}

threejs_visualization.triangular_mesh_renderer(mesh, width=400, height=400)

axis = np.array((0., 1., 0.))  
angle = np.array((np.pi / 4.,))  

mesh["vertices"] = tfg_transformation.axis_angle.rotate(mesh["vertices"], axis,
                                                        angle).numpy()

threejs_visualization.triangular_mesh_renderer(mesh, width=400, height=400)
```

该库用于训练具有训练数据隐私的机器学习模型。为此提供的一些[教程](https://web.archive.org/web/20221206064514/https://github.com/tensorflow/privacy/tree/master/tutorials)包括:

*   训练具有不同隐私的语言模型
*   具有差分隐私的 MNIST 卷积神经网络

差分隐私使用[ε和δ](https://web.archive.org/web/20221206064514/https://arxiv.org/abs/1908.10530)表示。

这是一个模型和数据集的图书馆，旨在使深度学习更容易获得，并加速机器学习的研究。

根据官方文件:

" TensorFlow Probability 是一个用于 TensorFlow 中概率推理和统计分析的库"

您可以使用该库对领域知识进行编码，但是它还具有:

*   支持多种概率分布
*   构建深度概率模型的工具
*   变分推理和马尔可夫链蒙特卡罗
*   优化者，如内尔德-米德、BFGS 和 SGLD

这是一个基于伯努利分布的示例模型:

```py
model = tfp.glm.Bernoulli()
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)
```

TensorFlow Extended (TFX)是一个平台，您可以使用它将您的机器学习管道投入生产。

另外，使用 [TensorFlow 的 ModelServer](https://web.archive.org/web/20221206064514/https://www.tensorflow.org/tfx/guide/serving) 可以让您使用 RESTful API 来访问您的模型。

假设您已经安装并配置了该服务器，则可以通过运行以下命令来启动它:

```py
$ tensorflow_model_server -- rest_api_port=8000 --   model_config_file=models.config -- model_config_file_poll_wait_seconds=300
```

该 API 将在本地主机的端口 8000 上可用。设置此服务器需要一些服务器管理知识。

[TensorBoard](/web/20221206064514/https://neptune.ai/blog/tensorboard-tutorial) 是 TensorFlow 的开源可视化工具包。您可以在您的模型培训中使用它作为回调，以便跟踪该过程。它可用于跟踪各种指标，如日志丢失和准确性。TensorBoard 还提供了几个可用于实验的工具。您可以使用它来:

*   可视化图像
*   检查模型权重和偏差
*   可视化模型的架构
*   通过性能分析查看应用程序的性能

仅举几个例子。

* * *

**注意:**作为替代，你也可以跟踪和可视化模型训练运行，并在 [Neptune](https://web.archive.org/web/20221206064514/https://neptune.ai/) 中对你的模型进行版本化。

例如，这里是你如何使用 Neptune 记录你的 Keras 实验。

```py
PARAMS = {'lr': 0.01, 'epochs': 10}
neptune.create_experiment('model-training-run', params=PARAMS)

model.fit(x_train, y_train,
          epochs=PARAMS['epochs'],
          callbacks=[NeptuneMonitor()])

neptune.log_artifact('model.h5')
```

跟踪 TensorFlow 模型训练的替代选项 [Neptune + TensorFlow/Keras 集成](https://web.archive.org/web/20221206064514/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)

这个库可以用于设计、实现和测试强化学习算法。它提供经过广泛测试的模块化组件。组件可以修改和扩展。

这个[笔记本](https://web.archive.org/web/20221206064514/https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb#scrollTo=cKOCZlhUgXVK)展示了如何在 Cartpole 环境下训练一个 [DQN(深度 Q 网络)](https://web.archive.org/web/20221206064514/https://www.tensorflow.org/agents/tutorials/0_intro_rl)代理。初始化代码如下所示:

```py
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

q_net = q_network.QNetwork(
  train_env.observation_spec(),
  train_env.action_spec(),
  fc_layer_params=(100,))

agent = dqn_agent.DqnAgent(
  train_env.time_step_spec(),
  train_env.action_spec(),
  q_network=q_net,
  optimizer=optimizer,
  td_errors_loss_fn=common.element_wise_squared_loss,
  train_step_counter=tf.Variable(0))

agent.initialize()
```

## 最后的想法

在本文中，我们探索了几个可用于扩展 TensorFlow 功能的库。尝试使用我提供的代码片段来熟悉这些工具。

我们讨论了:

*   使用 TensorFlow Hub 的预训练模型，
*   使用 TensorFlow 模型优化工具包优化您的模型，
*   使用 TensorFlow 推荐器构建推荐器，
*   使用 TensorFlow Federated 的分散数据训练模型，
*   使用 TensorFlow Privacy 在私人模式下训练。

这是相当多的，所以选择其中的一个开始，并浏览列表，看看是否有任何工具适合你的机器学习工作流程。