# 8 位创作者和核心贡献者谈论他们来自 PyTorch 生态系统的模型训练库

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem>

### 雅各布·查肯

大部分是 ML 的人。构建 MLOps 工具，编写技术资料，在 Neptune 进行想法实验。

我在 2018 年初使用 py torch 0 . 3 . 1 版本开始训练我的模型。我被 Pythonic 的手感、易用性和灵活性深深吸引。

在 Pytorch 中做事情要比在 Tensorflow 或 Theano 中容易得多。但是我错过的是 PyTorch 的类似 Keras 的高级接口，当时还没有太多。

快进到 2020 年，**我们在 [PyTorch 生态系统](https://web.archive.org/web/20220926085913/https://pytorch.org/ecosystem/)中有 6 个高级培训 API。**

*   但是你应该选择哪一个呢？
*   每一种使用的利弊是什么？

我想:谁能比作者自己更好地解释这些库之间的差异呢？

我拿起我的手机，请他们和我一起写一篇文章。他们都同意，这就是这个职位是如何创建的！

所以，**我请作者们谈谈他们图书馆的以下几个方面:**

*   项目理念
*   API 结构
*   新用户的学习曲线
*   内置功能(开箱即用)
*   扩展能力(研究中集成的简单性)
*   再现性
*   分布式培训
*   生产化
*   流行

…他们确实回答得很透彻🙂

#### 可以跳转到自己感兴趣的库或者直接到最后我的主观对比。

 [本杰明·博桑 核心撰稿人](https://web.archive.org/web/20220926085913/https://www.linkedin.com/in/benjamin-bossan-3114a684/) 

[斯科奇](https://web.archive.org/web/20220926085913/https://github.com/skorch-dev/skorch)发展背后的**理念**可以总结如下:

*   遵循 sklearn API
*   don’t hide PyTorch
*   不要多此一举
*   可以被黑客攻击

这些原则规划了我们的设计空间。关于 **scikit-learn API** ，它最明显的表现在你**如何训练和预测**:

```py
from skorch import NeuralNetClassifier

net = NeuralNetClassifier(...)
net.fit(X_train, y_train)
net.predict(X_test)

```

因为 skorch 正在使用这个**简单且完善的 API** ,所以每个人都应该能够很快开始使用它。

但是 sklearn 的整合比“适应”和“预测”更深入。您可以将您的 skorch 模型无缝集成到 sklearn 的“Pipeline”中，使用 sklearn 的众多指标(无需重新实现 F1、R 等。)，并配合`GridSearchCV`使用。

说到**参数扫描**:你可以使用任何其他的超参数搜索策略，只要有一个 sklearn 兼容的实现。

我们特别自豪的是**你可以搜索几乎任何超参数，而不需要额外的工作**。例如，如果您的模块有一个名为`num_units`的初始化参数，您可以立即对该参数进行网格搜索。

这里有一个**的列表，你可以用网格搜索现成的:**

*   您的`Module`上的任何参数(单元和层数、非线性度、辍学率等)
*   优化器(学习率、动力……)
*   标准
*   `DataLoader`(批量大小，洗牌，…)
*   回调(任何参数，甚至是自定义回调)

这是它在代码中的样子:

```py
from sklearn.model_selection import GridSearchCV

params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 20],
    'module__num_units': [10, 20],
    'optimizer__momentum': [0.6, 0.9, 0.95],
    'iterator_train__shuffle': [True, False],
    'callbacks__mycallback__someparam': [1, 2, 3],
}

net = NeuralNetClassifier(...)
gs = GridSearchCV(net, params, cv=3, scoring='accuracy')
gs.fit(X, y)

print(gs.best_score_, gs.best_params_)

```

据我所知，没有其他框架提供这种灵活性。最重要的是，通过使用 dask 并行后端，您可以 [**将超参数搜索**](https://web.archive.org/web/20220926085913/https://skorch.readthedocs.io/en/stable/user/parallelism.html) 分布到您的集群中，而不会有太多麻烦。

使用成熟的 sklearn API，skorch 用户可以**避免在纯 PyTorch 中编写训练循环、验证循环和超参数搜索时常见的样板代码**。

从 PyTorch 方面来说，我们决定不像 keras 那样将后端隐藏在抽象层之后。相反，**我们公开了 PyTorch** 的众多组件。作为用户，你可以使用 PyTorch 的`Dataset`(想想 torchvision，包括 TTA)`DataLoader`，和学习率调度器。最重要的是，你可以不受限制地使用 PyTorch `Module` s。

因此，我们有意识地努力**尽可能多地重用 sklearn 和 PyTorch 的现有功能**，而不是重新发明轮子。这使得 skorch **易于在你现有的代码库**上使用，或者在你最初的实验阶段后移除它，而没有任何锁定效应。

例如，您可以用任何 sklearn 模型替换神经网络，或者您可以提取 PyTorch 模块并在没有 skorch 的情况下使用它。

在重用现有功能的基础上，我们添加了一些自己的功能。最值得注意的是，skorch **可以开箱即用地处理许多常见的数据类型**。除了`Dataset` s，您还可以使用:

*   numpy 数组，
*   火炬张量，
*   熊猫，
*   保存异构数据的 Python 字典，
*   外部/自定义数据集，如 torchvision 的 [ImageFolder。](https://web.archive.org/web/20220926085913/https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Transfer_Learning.ipynb)

我们已经付出了额外的努力来使这些与 sklearn 一起很好地工作。

此外，我们实现了一个简单而**强大的回调系统**，你可以用它来**根据你的喜好**调整 skorch 的大部分行为。我们提供的一些回调包括:

*   学习率调度程序，
*   评分功能(使用自定义或 sklearn 指标)，
*   提前停车，
*   检查点，
*   参数冻结，
*   以及 TensorBoard 和 Neptune 集成。

如果这还不足以满足您的定制需求，**我们尽力帮助您实施自己的回访或您自己的模型培训师**。我们的文档包含了如何实现[定制回调](https://web.archive.org/web/20220926085913/https://skorch.readthedocs.io/en/stable/user/callbacks.html#callback-base-class)和[定制训练者](https://web.archive.org/web/20220926085913/https://skorch.readthedocs.io/en/stable/user/neuralnet.html#subclassing-neuralnet)的例子，修改每一个可能的行为直到训练步骤。

对于任何熟悉 sklearn 和 PyTorch 的人来说，不重新发明轮子的哲学应该使 skorch 易于学习。由于我们围绕定制和灵活性设计了 skorch，因此应该不难掌握。要了解更多关于 skorch 的信息，请查看这些[示例](https://web.archive.org/web/20220926085913/https://github.com/skorch-dev/skorch/tree/master/examples)和[笔记本](https://web.archive.org/web/20220926085913/https://github.com/skorch-dev/skorch/tree/master/notebooks)。

sko rch**面向生产**并用于生产。我们讨论了一些关于生产化的常见问题，特别是:

*   我们确保**是向后兼容的**，并在必要时给出足够长的折旧期。
*   可以**在 GPU 上训练，在 CPU 上服务，**
*   你可以**腌制一整只包含 skorch 模型的 sklearn `Pipeline`** 以备后用。
*   我们提供了一个助手函数来 **[将您的训练代码转换成命令行脚本](https://web.archive.org/web/20220926085913/https://github.com/skorch-dev/skorch/tree/master/examples/cli)** ，它将您的所有模型参数，包括它们的文档，作为命令行参数公开，只需要三行额外的代码

也就是说，我已经实现了，或者知道有人已经实现了，更多的**研究** -y 的东西，像 **GANs** 和无数类型的**半监督学习**技术。不过，这确实需要对 skorch 有更深入的了解，所以您可能需要更深入地研究文档，或者向我们寻求 github 的指导。

就我个人而言，我还没有遇到任何人使用 skorch 进行强化学习，但我想听听人们对此有什么体验。

自从我们在 2017 年夏天首次发布 skorch 以来，该项目已经成熟了很多，并且围绕它已经发展了一个活跃的社区。在一个典型的星期里，github 上会打开一些问题，或者在 stackoverflow 上提出一个问题。我们会在一天之内回答大多数问题，如果有好的功能需求或 bug 报告，我们会尝试引导报告者自己实现它。

通过这种方式，**在项目的整个生命周期中，我们有 20 多个贡献者，其中 3 个是常客**，这意味着项目的健康不依赖于一个人。

fastai 说，skorch 和其他一些高级框架的最大区别是 skorch 不“包含电池”。这意味着，实现他们自己的模块或者使用许多现有集合中的一个(比如 torchvision)的模块取决于用户。斯科奇提供骨架，但你得带上肉。

### **何时不使用 skorch**

*   超级自定义 PyTorch 代码，可能是强化学习
*   后端不可知代码(在 PyTorch、tensorflow 等之间切换)
*   根本不需要 sklearn API
*   避免非常轻微的性能开销

### **何时使用 skorch**

*   获得 sklearn API 和所有相关的好处，如超参数搜索
*   大多数 PyTorch 工作流都能正常工作
*   避免样板文件，规范代码
*   使用上面讨论的许多实用程序中的一些

 [谢尔盖·科列斯尼科夫 创造者](https://web.archive.org/web/20220926085913/https://www.linkedin.com/in/scitator/) 

### **哲学**

催化剂背后的想法很简单:

*   在一个框架中收集所有技术、开发、深度学习的东西，
*   使重复使用枯燥的日常组件变得容易，
*   在我们的项目中关注研究和假设检验。

为了实现这一点，我们研究了一个典型的深度学习项目，它通常具有以下结构:

```py
for stage in stages:
    for epoch in epochs:
        for dataloader in dataloaders:
            for batch in dataloader:
                handle(batch)
```

想想看，大多数时候，您需要做的就是为新模型指定处理方法，以及如何将数据批量提供给该模型。那么，为什么我们把这么多时间花在实现流水线和调试训练循环上，而不是开发新东西或测试假设呢？

我们意识到有可能**将工程与研究**分开，这样我们就可以**将我们的时间一次性**投入到高质量、可重复使用的**工程**主干上**在所有项目中使用它**。

Catalyst 就是这样诞生的:一个开源的 PyTorch 框架，它允许你编写紧凑但功能齐全的管道，**抽象工程样板文件，**让你专注于项目的主要部分。

> 我们在 Catalyst 的任务。团队将使用我们的软件工程和深度学习专业知识来标准化工作流，并实现深度学习和强化学习研究人员之间的跨领域交流。

我们相信，开发摩擦的减少和思想的自由流动将导致未来数字图书馆的突破，这样的 R&D 生态系统将有助于实现这一目标。

### **学习曲线**

Catalyst 可以被 DL 新手和经验丰富的专家轻松采用，这得益于两个 API:

*   **笔记本 API** ，它的开发重点是**简单的实验和 Jupyter 笔记本的使用**-开始你的可重复 DL 研究之路。
*   **Config API** ，主要关注**可伸缩性和 CLI 接口**——即使在大型集群上也能发挥 DL/RL 的威力。

说到 PyTorch 用户体验，我们真的希望它尽可能简单:

*   您可以像平时一样定义加载器、模型、标准、优化器和调度器:

```py
import torch

loaders = {"train": ..., "valid": ...}

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
```

*   你把这些 PyTorch 对象传递给 Catalyst `Runner`

```py
from catalyst.dl import SupervisedRunner

logdir = "./logdir"
num_epochs = 42

runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,)
```

**在几乎没有样板文件的情况下，将工程与深度学习明确分离**。这是我们觉得深度学习代码应该有的样子。

要开始使用这两种 API，你可以遵循我们的[教程和管道](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst#docs-and-examples)或者如果你不想选择，只需检查最常见的:[分类](https://web.archive.org/web/20220926085913/https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb)和[分割。](https://web.archive.org/web/20220926085913/https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)

### **设计和架构**

关于 **Notebook 和 Config API 最有趣的部分是它们使用了相同的“后端”逻辑**–`Experiment`、`Runner`、`State`和`Callback`抽象，这是 Catalyst 的核心特性。

*   [**实验** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/experiment.py) 包含实验信息的抽象——模型、标准、优化器、调度器以及它们的超参数。它还包含有关使用的数据和转换的信息。总的来说，实验知道**你想要运行什么**。
*   [**Runner** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/runner.py) 知道如何进行实验的类。它包含了**如何**运行实验、阶段(催化剂的另一个显著特征)、时期和批次的所有逻辑。
*   [**状态** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/state.py) 实验和运行程序之间的一些中间存储，保存实验的当前**状态**——模型、标准、优化器、调度器、度量、记录器、加载器等
*   [**回调** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/callback.py) 一个强大的抽象，让你**定制**你的实验运行逻辑。为了给用户最大的灵活性和可扩展性，我们允许在训练循环的任何地方执行回调:

```py
on_stage_start
    on_epoch_start
       on_loader_start
           on_batch_start

       on_batch_end
    on_epoch_end
on_stage_end

on_exception
```

通过实现这些方法，您可以实现任何额外的逻辑。

因此，你可以在几行代码(以及 Catalyst 之后)中**实现任何深度学习管道** **。RL 2.0 版本-强化学习管道)，从可用的原语中组合它(感谢社区，他们的数量每天都在增长)。**

其他一切(模型、标准、优化器、调度器)都是纯 PyTorch 原语。Catalyst 不会在顶层创建任何包装器或抽象，而是让在不同框架和领域之间重用这些构件变得容易。

### **扩展能力/研究中集成的简单性**

由于灵活的框架设计和回调机制，Catalyst 可以很容易地扩展到大量基于 DL 的项目。你可以在 [awesome-catalyst-list](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/awesome-catalyst-list#repositories) 上查看我们的 Catalyst-powered 知识库。

如果您对**强化学习**感兴趣，也有大量基于 RL 的回购和竞争解决方案。来比较催化剂。使用其他 RL 框架，你可以查看[开源 RL 列表](https://web.archive.org/web/20220926085913/https://docs.google.com/spreadsheets/d/1EeFPd-XIQ3mq_9snTlAZSsFY7Hbnmd7P5bbT8LPuMn0/edit?usp=sharing)。

### **其他内置特性(开箱即用)**

知道你可以很容易地扩展它会让你感觉很舒服，但是你有很多现成的特性。其中一些包括:

*   基于灵活的回调系统，Catalyst 已经**轻松集成了**如**常见的深度学习最佳实践**，如梯度累积、梯度裁剪、权重衰减校正、top-K 最佳检查点保存、tensorboard 集成以及许多其他有用的日常深度学习实用程序。
*   由于我们的贡献者和贡献模块， **Catalyst 可以访问所有最新的 SOTA 功能**，如 AdamW、OneCycle、SWA、Ranger、LookAhead 和许多其他研究开发。
*   此外，**我们整合了**像 Nvidia apex、[albuminations](https://web.archive.org/web/20220926085913/https://github.com/albu/albumentations)、 [SMP](https://web.archive.org/web/20220926085913/https://github.com/qubvel/segmentation_models.pytorch) 、 [transformers](https://web.archive.org/web/20220926085913/https://github.com/huggingface/transformers) 、wandb 和 neptune.ai 这样的**流行库**，让您的研究更加人性化。由于这样的集成，Catalyst 完全支持测试时间扩充、混合精度和分布式训练。
*   为了满足行业需求，我们还提供了对 [PyTorch 跟踪](https://web.archive.org/web/20220926085913/https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)的框架式支持，这使得将模型投入生产变得更加容易。此外，我们在每个版本中部署预定义的基于 Catalyst 的 docker 映像，以便于集成。
*   最后，我们支持针对模型服务—[反应](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/reaction)(面向行业)和实验监控—[炼金术](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/alchemy)(面向研究)的额外解决方案。

一切都集成到库中，并涵盖了 CI 测试(我们有一个专用的 gpu 服务器)。感谢 Catalyst 脚本，您可以**安排大量实验，并从命令行在所有可用的 GPU 上并行运行它们(查看 catalyst-parallel-run 了解更多信息)。**

### **再现性**

我们做了大量的工作，使你用催化剂运行的实验具有可再现性。由于基于库的确定性**，基于 Catalyst 的实验是可重复的**不仅在一个服务器上的服务器运行之间**，而且在不同服务器**和不同硬件部件(当然，使用 docker 封装)上的几次运行之间也是如此。感兴趣的话可以在这里看到实验[。](https://web.archive.org/web/20220926085913/https://app.wandb.ai/scitator/classification_reproducubility_check?workspace=user-scitator)

而且，**强化学习实验也是面向再现性的**(就 RL 而言 RL 是可再现的)。例如，通过同步实验运行，由于采样轨迹的确定性，您可以获得非常接近的性能。这是众所周知的困难，据我所知**催化剂有最可再生的 RL 管道。**

为了实现 DL 和 RL 再现性的新水平，我们必须创造几个额外的功能:

*   **完整的源代码转储:**由于实验、运行器和回调抽象，保存这些原语以备后用非常容易。
*   **Catalyst 源代码包:**有了这样的特性，即使使用 Catalyst 的开发版本，您也可以随时重现实验结果。
*   **环境版本化:** Catalyst 转储 pip 和 conda 包版本(稍后可用于定义您的 docker 映像)
*   最后，Catalyst 支持几个**监控工具**，如 Alchemy、Neptune.ai、Wandb，以存储您的所有实验指标和附加信息，从而更好地跟踪研究进度和再现性。

由于这些基于库的解决方案，您可以确保在 Catalyst 中实现的管道是可重复的，并且保存了所有实验日志和检查点以供将来参考。

### **分布式培训**

基于我们的集成，Catalyst 已经有了对分布式培训的本地支持。此外，我们支持 Slurm 培训，并致力于更好地整合 DL 和 RL 管道。

### 生产化

既然我们知道 Catalyst 如何帮助深度学习研究，我们就可以谈论**将训练好的模型部署到生产中**。

正如已经提到的，Catalyst **支持开箱即用的模型跟踪。**它允许您将 PyTorch 模型(使用 Python 代码)转换为 TorchScript 模型(集成了所有内容)。TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法。任何 TorchScript 程序都可以从 Python 进程中保存，并在没有 Python 依赖的进程中加载。

此外，为了帮助 Catalyst 用户将其管道部署到生产系统中，Catalyst。团队有一个 **[Docker Hub](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst#docker) ，带有预构建的基于 Catalyst 的映像**(包括 fp16 支持)。

此外，为了帮助研究人员将他们的想法投入生产和现实世界的应用，我们创造了 Catalyst。生态系统:

*   [**反应** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/reaction) 我们自己的 **PyTorch 服务解决方案**，具有同步/异步 API、批处理模式支持、quest，以及所有其他你可以从一个设计良好的生产系统中期待的典型后端。
*   [**炼金** :](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/alchemy) 我们的**监控工具**用于实验跟踪、模型对比、研究成果共享。

### **人气**

自从 12 个月前第一次发布 pypi 以来，Catalyst 已经在 Github 上获得了 1.5k 颗星，超过 **100k 次下载**。我们很自豪成为这样一个开源生态系统的一部分，非常感谢我们所有的用户和贡献者的不断支持和反馈。

其中一个特别有帮助的在线社区是 [ods.ai:](https://web.archive.org/web/20220926085913/https://opendatascience.slack.com/messages/CGK4KQBHD) 世界上最大的数据科学家和机器学习实践者的 slack 渠道之一(40k+用户)。没有他们的想法和反馈，Catalyst 就不会有今天。

特别感谢我们的早期用户，

这一切都是值得的。

<details><summary class="wp-block-coblocks-accordion-item__title">[Acknowledgments](/web/20220926085913/https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem)</summary>

自从сcatalyst 开始发展以来，许多人以不同的方式影响了它。为了表达我的感激之情，我要向...表示我个人的谢意:</details> 

感谢所有这些支持，Catalyst 已经成为 Kaggle docker image 的一部分，被**添加到 [PyTorch 生态系统](https://web.archive.org/web/20220926085913/https://pytorch.org/ecosystem/)** 中，现在我们正在[开发我们自己的 DL R & D 生态系统](https://web.archive.org/web/20220926085913/https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing)以加速您的研究和生产需求。

阅读更多关于 **Catalyst 的信息。生态系统**，请查看[我们的愿景](https://web.archive.org/web/20220926085913/https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing)和[项目宣言。](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md)

最后，我们总是乐意帮助我们的[催化剂。朋友:](https://web.archive.org/web/20220926085913/https://github.com/catalyst-team/awesome-catalyst-list#trusted-by)公司/初创公司/研究实验室，他们已经在使用 Catalyst，或者正在考虑将它用于他们的下一个项目。

> 感谢阅读，并…打破循环-使用催化剂！

### **何时使用催化剂**

*   拥有灵活和可重用的代码库，没有样板文件。你希望与来自不同深度学习领域的其他研究人员分享你的专业知识。
*   使用 Catalyst.Ecosystem 提高您的研究速度

### **何时不使用催化剂**

*   你才刚刚开始你的深度学习之路——从这个角度来说，低级 PyTorch 是一个很好的入门。
*   你想用一堆不可复制的技巧创建非常具体的、定制的管道🙂

[](https://web.archive.org/web/20220926085913/https://twitter.com/jeremyphoward) [西尔万·古格 核心撰稿人

注意:](https://web.archive.org/web/20220926085913/https://twitter.com/guggersylvain) 

### 下面是关于将于 2020 年 7 月发布的 fastai 的**版本 2。你可以在这里去回购[，在这里](https://web.archive.org/web/20220926085913/https://github.com/fastai/fastai)查文件[。](https://web.archive.org/web/20220926085913/https://docs.fast.ai/)**

What follows is about the **version 2 of fastai that will be released in July 2020**. You can go to repo [here](https://web.archive.org/web/20220926085913/https://github.com/fastai/fastai) and check the documentation [here](https://web.archive.org/web/20220926085913/https://docs.fast.ai/).

Fastai 是一个深度学习库，它提供:

**从业者**:有了可以快速便捷地提供标准深度学习领域最先进成果的高级组件，

*   **研究人员**:用可以混合搭配的低级组件来构建新的东西。
*   它的目标是在不牺牲易用性、灵活性或性能的情况下做到这两点。

得益于精心分层的架构，这成为可能**。它以**解耦抽象**的形式表达了许多深度学习和数据处理技术的通用底层模式。重要的是，这些抽象可以用**清晰简洁地表达出来**，这使得 fastai 变得平易近人**快速高效，同时也是深度可黑客化和可配置的**。**

一个高级 API 提供了**可定制的模型和合理的默认值**，它建立在一个由低级构建块构成的**层级之上。**

本文涵盖了该库功能的一个代表性子集。有关详细信息，请参见我们的 [fastai 论文](https://web.archive.org/web/20220926085913/https://arxiv.org/abs/2002.04688)和文档。

**API**

### 当谈到 fastai API 时，我们需要区分高级和中级/低级 API。我们将在接下来的章节中讨论这两者。

***高级 API***

高级 API 对于初学者和**主要对应用预先存在的深度学习方法感兴趣的从业者非常有用。**

它为主要应用领域提供了简明的 API:

视觉，

*   文字，
*   扁平的
*   时间序列分析，
*   推荐(协同过滤)
*   这些**API 基于所有可用信息选择智能默认值**和行为。

例如，fastai 提供了一个 **`Learner`类**，它集合了架构、优化器和数据，并且**在可能的情况下自动选择一个合适的损失函数。**

再举一个例子，一般来说，训练集应该洗牌，验证集不应该洗牌。fastai 提供了一个单独的 **`Dataloaders`类**，该类自动**构造验证和训练数据加载器，这些细节已经得到处理。**

为了了解这些“清晰简洁的代码”原则是如何发挥作用的，让我们在[牛津 IIT Pets 数据集](https://web.archive.org/web/20220926085913/https://www.robots.ox.ac.uk/~vgg/data/pets/)上微调一个 [imagenet](https://web.archive.org/web/20220926085913/http://www.image-net.org/) 模型，并在单个 GPU 上几分钟的训练内实现接近最先进的精度:

这不是摘录。这是这项任务所需的所有代码行。每一行代码都执行一项重要的任务，让用户专注于他们需要做的事情，而不是次要的细节:

```py
from fastai.vision.all import *

path = untar_data(URLs.PETS)
dls = ImageDataloaders.from_name_re(path=path, bs=64,
    fnames = get_image_files(path/"images"), path = r'/([^/]+)_\d+.jpg$',
    item_tfms=RandomResizedCrop(450, min_scale=0.75), 
    batch_tfms=[*aug_transforms(size=224, max_warp=0.), 
                Normalize.from_stats(*imagenet_stats)])

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)
```

**从库中导入所有必需的棋子**。值得注意的是，这个库是经过精心设计的，以避免这些风格的导入搞乱名称空间。

```py
from fastai.vision.all import * 

```

**将标准数据集**从 fast.ai 数据集集合(如果之前没有下载)下载到一个可配置的位置，提取它(如果之前没有提取)，并返回一个带有提取位置的`pathlib.Path`对象。

```py
path = untar_data(URLs.PETS)

```

设置`Dataloaders`。注意**项目级和批次级转换的分离**:

```py
dls = ImageDataloaders.from_name_re(path=path, bs=64,
    fnames = get_image_files(path/"images"), pat = r'/([^/]+)_\d+.jpg$',
    item_tfms=RandomResizedCrop(450, min_scale=0.75), 
    batch_tfms=[*aug_transforms(size=224, max_warp=0.), 
    Normalize.from_stats(*imagenet_stats)])

```

***项*** 变换应用**到 CPU 上的单个图像**

*   ***批处理*** 变换应用**到 GPU** (如果可用)上的一个小批处理。
*   `aug_transforms()`选择一组数据扩充。与 fastai 中的一贯做法一样，我们选择了一个适用于各种视觉数据集的默认设置，但如果需要，也可以进行完全定制。

创建一个`Learner`，这个**结合了一个优化器、一个模型和一个用于训练的数据**。**每个应用程序(视觉、文本、表格)都有一个定制的函数，创建一个`Learner`** ，它能为用户自动处理任何细节。例如，在这个图像分类问题中，它将:

```py
learn = cnn_learner(dls, resnet34, metrics=error_rate)

```

下载 ImageNet 预训练模型(如果还没有),

*   去掉模型的分类头，
*   用适合于这个特定数据集的报头来替换它，
*   设置适当的优化器、权重衰减、学习率等等
*   微调模型。在这种情况下，它使用 1 周期策略，这是最近用于训练深度学习模型的最佳实践，但在其他库中并不广泛可用。很多事情发生在`.fine_tune()`的引擎盖下:

```py
learn.fine_tune(4)

```

退火学习率和动量，

*   在验证集上打印指标，
*   在 HTML 或控制台表格中显示结果
*   在每批之后记录损失和度量，等等。
*   如果有可用的 GPU，将会使用它。
*   当模型的主体被冻结时，它将首先训练头部一个时期，然后使用区别学习率微调给定的许多时期(这里是 4 个)。
*   fastai 库的**优势之一是 API 跨应用程序的一致性。**

例如，使用 ULMFiT 对 IMDB 数据集上的预训练模型进行微调(文本分类任务)只需 6 行代码:

用户在其他领域得到非常**相似的体验**，比如表格、时间序列或推荐系统。一旦一个`Learner`被训练，你可以用命令`learn.show_results()`来探索结果。这些结果如何呈现取决于应用，在视觉中你得到的是带标签的图片，在文本中你得到的是汇总样本、目标和预测的数据框架。在我们的宠物分类示例中，您会看到类似这样的内容:

```py
from fastai2.text.all import *

path = untar_data(URLs.IMDB)
dls = TextDataloaders.from_folder(path, valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

```

在 IMDb 分类问题中，你会得到这样的结果:

另一个重要的高级 API 组件是**数据块 API，**，它是一个用于数据加载的表达性 API。这是我们所知的第一次尝试，系统地定义为深度学习模型准备数据所必需的所有步骤，并为用户提供一个混合搭配的食谱，用于组合这些片段(我们称之为数据块)。

下面是一个如何使用数据块 API 让 [MNIST](https://web.archive.org/web/20220926085913/https://yann.lecun.com/exdb/mnist/) 数据集为建模做好准备的示例:

***中低档 API***

```py
mnist = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(),
    get_y=parent_label)
dls = mnist.databunch(untar_data(URLs.MNIST_TINY), batch_tfms=Normalize)

```

在上一节中，您看到了如何使用具有大量开箱即用功能的高级 api 快速完成大量工作。然而，有些情况下，你需要调整或扩展已经存在的东西。

这就是中级和低级 API 发挥作用的地方:

**中级 API** 为这些应用中的每一个提供核心的深度学习和数据处理方法，

*   **低级 API** 提供了一个优化的原语库以及功能和面向对象的基础，允许中间层进行开发和定制。
*   可以使用 **`Learner`新型双向回调系统定制训练循环。**它允许梯度、数据、损失、控制流和**任何东西**其他**在训练期间的任何点被读取和改变。**

使用回调来定制数值软件有着悠久的历史，今天几乎所有现代深度学习库都提供了这一功能。然而，fastai 的回调系统是我们所知的第一个支持**完成双向回调**所必需的设计原则的系统:

在训练的每一点都应该有回叫**，这给了用户充分的灵活性。每个回调都应该**能够访问训练循环中该阶段可用的每条信息**，包括超参数、损耗、梯度、输入和目标数据等等；**

*   每次回调都应该能够在使用这些信息之前的任何时候修改所有这些信息，
*   训练循环的所有调整(不同的调度器、混合精度训练、在 [TensorBoard](https://web.archive.org/web/20220926085913/https://www.tensorflow.org/tensorboard) 、 [wandb](https://web.archive.org/web/20220926085913/https://www.wandb.com/) 、 [neptune](https://web.archive.org/web/20220926085913/https://neptune.ai/) 或等效物、[mix](https://web.archive.org/web/20220926085913/https://arxiv.org/abs/1710.09412)、过采样策略、分布式训练、GAN 训练……)都在回调中实现，最终用户**可以将它们与自己的进行混合和匹配，从而更容易试验**和进行消融研究。方便的方法可以为用户添加这些回调，使得混合精度的训练就像说的那样简单

或者**在分布式环境中培训**一样简单

```py
learn = learn.to_fp16()

```

fastai 还提供了一个**新的通用优化器抽象**，允许用几行代码实现最近的优化技术，如 LAMB、RAdam 或 AdamW。

```py
learn = learn.to_distributed()

```

多亏了**将优化器抽象**重构为两个基本部分:

***stats*** ，跟踪并汇总梯度移动平均线等统计数据

*   ***步进器*** ，它结合了统计数据和超参数，使用一些函数来“步进”权重。
*   有了这个基础，我们可以用 2-3 行代码编写 fastai 的大部分优化器，而在其他流行的库中，这需要 50 多行代码。

还有许多其他的中间层和低层 APIs】使得研究人员和开发人员可以在快速灵活的基础上轻松构建新方法。

这个图书馆已经在研究、工业和教学中广泛使用。我们已经用它创建了一个完整的，非常受欢迎的深度学习课程:[程序员实用深度学习](https://web.archive.org/web/20220926085913/https://course.fast.ai/)(最后一次迭代的第一个视频有 256k 的浏览量)。

在撰写本文时，[库](https://web.archive.org/web/20220926085913/https://github.com/fastai/fastai)拥有 **16.9k 恒星，并在超过 2000 个项目**中使用。社区在 [fast.ai 论坛](https://web.archive.org/web/20220926085913/https://forums.fast.ai/)上非常活跃，无论是澄清课程中不清楚的点，帮助调试还是合作解决新的深度学习项目。

**何时使用 fastai**

### 我们的目标是让一些东西对初学者来说足够简单，但对研究人员/从业者来说足够灵活。

*   **何时不使用 fastai**

### 我能想到的唯一一件事是，你不会使用 fastai 在生产中服务于你在不同框架中训练的模型，因为我们不处理那个方面。

*   维克多·福明 核心撰稿人

 [Pytorch Ignite 是一个高级库，帮助在 Pytorch 中训练神经网络。自 2018 年初以来，我们的目标一直是:](https://web.archive.org/web/20220926085913/https://twitter.com/vfdev_5) 

“让普通的事情变得容易，让困难的事情变得可能”。

> **为什么要使用 Ignite？**

### Ignite 的高抽象级别**很少假设用户正在训练的模型类型或多个模型**。我们只要求用户**定义要在训练和可选验证循环**中运行的闭包。它为用户提供了很大的灵活性，允许他们在任务中使用 Ignite，例如共同训练多个模型(即 gan)或在训练循环中跟踪多个损失和指标

**点燃概念和 API**

### 您需要了解 Ignite API 中的一些核心对象:

***引擎*** :精华库

*   ***事件&处理程序*** :与`Engine`交互(如提前停止、检查点、日志记录)
*   ***指标*** :各种任务的现成指标
*   我们将提供一些基础知识来理解主要思想，但可以随意深入挖掘存储库中的[示例](https://web.archive.org/web/20220926085913/https://pytorch.org/ignite)。

***引擎***

它只是遍历提供的数据，执行一个处理函数并返回一个结果。

一个 **`Trainer`是一个`Engine`，以模型的权重更新**作为处理函数。

一个 **`Evaluator`(验证模型的对象)是一个以在线度量计算逻辑**为处理功能的`Engine`。

```py
from ignite.engine import Engine

def update_model(trainer, batch):
    model.train()
    optimizer.zero_grad()
    x, y = prepare_batch(batch)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(update_model)
trainer.run(data, max_epochs=100)
```

这段代码可以悄悄地训练一个模型，并计算总损失。

```py
from ignite.engine import Engine

total_loss = []
def compute_metrics(_, batch):
    x, y = batch
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        total_loss.append(loss.item())

    return loss.item()

evaluator = Engine(compute_metrics)
evaluator.run(data, max_epochs=1)
print(f”Loss: {torch.tensor(total_loss).mean()}”)

```

在下一节中，我们将了解如何使培训和验证更加用户友好。

***事件&经手人***

为了**提高`Engine`的灵活性**，并允许用户在运行的每一步进行交互，**我们引入了事件和处理程序**。这个想法是，用户可以在训练循环内部执行一个自定义代码作为事件处理程序，类似于其他库中的回调。

在每次 *fire_event* 调用时，它的所有事件处理程序都会被执行。例如，用户可能希望在训练开始时设置一些运行相关变量(`Events.STARTED`)，并在每次迭代中更新学习率(`Events.ITERATION_COMPLETED`)。使用 Ignite，代码将如下所示:

```py
fire_event(Events.STARTED)

while epoch < max_epochs:
    fire_event(Events.EPOCH_STARTED)

    for batch in data:
        fire_event(Events.ITERATION_STARTED)

        output = process_function(batch)

        fire_event(Events.ITERATION_COMPLETED)
    fire_event(Events.EPOCH_COMPLETED)
fire_event(Events.COMPLETED)

```

**处理程序(相对于“回调”接口)的酷之处在于，它可以是任何具有正确签名的函数**(我们只要求第一个参数是 engine)，例如 lambda、简单函数、类方法等。我们不需要从一个接口继承，也不需要覆盖它的抽象方法。

```py
train_loader = …
model = …
optimizer = …
criterion = ...
lr_scheduler = …

def process_function(engine, batch):

trainer = Engine(process_function)

@trainer.on(Events.STARTED)
def setup_logging_folder(_):

@trainer.on(Events.ITERATION_COMPLETED)
def update_lr(engine):
    lr_scheduler.step()

trainer.run(train_loader, max_epochs=50)

```

**内置事件过滤**

```py
trainer.add_event_handler(
    Events.STARTED, lambda engine: print("Start training"))

mydata = [1, 2, 3, 4]

def on_training_ended(engine, data):
    print("Training is ended. mydata={}".format(data))

trainer.add_event_handler(
    Events.COMPLETED, on_training_ended, mydata)

```

### 有些情况下，用户希望定期/一次性执行代码，或者使用自定义规则，如:

每 5 个时期运行一次验证，

*   每 1000 次迭代存储一个检查点，
*   在第 20 个时期改变一个变量，
*   在前 10 次迭代中记录梯度。
*   等等。
*   Ignite 提供了这样的**灵活性，将“要执行的代码”与逻辑“何时执行代码”分开。**

例如，为了使**每 5 个时期**运行一次验证，只需简单编码:

类似地，为了**在第 20 个时期**改变一些训练变量一次:

```py
@trainer.on(Events.EPOCH_COMPLETED(every=5))
def run_validation(_):

```

更一般地，用户可以提供自己的事件过滤功能:

```py
@trainer.on(Events.EPOCH_STARTED(once=20))
def change_training_variable(_):

```

**现成的处理程序**

```py
@trainer.on(Events.EPOCH_STARTED(once=20))
def change_training_variable(_):

```

### Ignite 提供了一系列处理程序和指标来简化用户代码:

***检查点*** :保存训练检查点(由训练器、模型、优化器、lr 调度器等组成)，保存最佳模型(按验证分数)

*   ***提前停止* :** 如果没有进展(根据验证分数)则停止训练
*   ***终止南:*** 遇到南就停止训练
*   ***优化器参数调度:*** 串接，添加预热，设置线性或余弦退火，任意优化器参数的线性分段调度(lr，momentum，betas，…)
*   记录到通用平台:TensorBoard、Visdom、MLflow、Polyaxon 或 Neptune(批量损失、度量 GPU 内存/利用率、优化器参数等)。

*   ***指标***

Ignite 还为各种任务提供了一个现成的指标列表**:精度、召回率、准确度、混淆矩阵、IoU 等，大约 20 个回归指标**

例如，下面我们计算验证数据集的验证准确度:

点击[此处](https://web.archive.org/web/20220926085913/https://pytorch.org/ignite/metrics.html#complete-list-of-metrics)和[此处](https://web.archive.org/web/20220926085913/https://pytorch.org/ignite/contrib/metrics.html)查看可用指标的完整列表。

```py
from ignite.metrics import Accuracy

def compute_predictions(_, batch):

    return y_pred, y_true

evaluator = Engine(compute_predictions)
metric = Accuracy()
metric.attach(evaluator, "val_accuracy")
evaluator.run(val_loader)
> evaluator.state.metrics[“val_accuracy”] = 0.98765

```

Ignite 指标有一个很酷的特性，即**用户可以使用基本的算术运算**或 torch 方法构建自己的指标:

**库结构**

```py
precision = Precision(average=False)
recall = Recall(average=False)
F1_per_class = (precision * recall * 2 / (precision + recall))
F1_mean = F1_per_class.mean()  
F1_mean.attach(engine, "F1")

```

### 该库由两个主要模块组成:

***核心*** 模块包含像引擎、指标、一些必要的处理程序这样的基础。它把 **PyTorch 作为唯一的附属国。**

*   ***Contrib*** 模块可能依赖于其他库(如 scikit-learn、tensorboardX、visdom、tqdm 等)，并可能在版本之间有向后兼容性破坏更改。
*   单元测试涵盖了这两个模块。

**扩展能力/研究中集成的简单性**

### 我们相信，我们的事件/处理程序系统相当灵活，使人们能够与培训过程的每个部分进行交互。正因为如此，**我们已经看到 Ignite 被用来训练 GANs** (我们提供了两个基本例子来训练 [DCGAN](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/tree/master/examples/gan) 和 [CycleGAN](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN.ipynb) )或**强化学习模型。**

根据 Github 的“被使用”，Ignite 是被研究人员用于他们的论文的**:**

BatchBALD:深度贝叶斯主动学习的高效多样批量获取， [github](https://web.archive.org/web/20220926085913/https://github.com/BlackHC/BatchBALD)

*   一个寻找可合成分子的模型， [github](https://web.archive.org/web/20220926085913/https://github.com/john-bradshaw/molecule-chef)
*   本地化的生成流， [github](https://web.archive.org/web/20220926085913/https://github.com/jrmcornish/lgf)
*   从生物医学文献中提取 T 细胞的功能和分化特征， [github](https://web.archive.org/web/20220926085913/https://github.com/hammerlab/t-cell-relation-extraction)
*   由于这些(以及其他研究项目)，我们坚信 **Ignite 为您提供了足够的灵活性来进行深度学习研究。**

**与其他库/框架的集成**

### 如果其他库或框架的特性不重叠的话，Ignite **可以很好地与它们配合。我们拥有的一些很酷的集成包括:**

用 Ax ( [点火示例](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb))调整超参数。

*   使用 Optuna 进行超参数调整( [Optuna 示例](https://web.archive.org/web/20220926085913/https://github.com/optuna/optuna/blob/master/examples/pytorch_ignite_simple.py))。
*   登录 TensorBoard，Visdom，MLflow，Polyaxon，Neptune (Ignite 的代码)，Chainer UI (Chainer 的代码)。
*   使用 Nvidia Apex 进行混合精度训练( [Ignite 的例子](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/tree/master/examples/references))。
*   **再现性**

### 我们为 Ignite 培训的可重复性付出了巨大努力:

Ignite 的**引擎自动处理随机状态**，并在可能的情况下强制数据加载器在不同的运行中提供相同的数据样本；

*   Ignite **集成了 MLflow、Polyaxon、Neptune 等实验跟踪系统**。这有助于跟踪 ML 实验的软件、参数和数据依赖性；
*   我们提供了几个关于视觉任务的**可重复训练的示例和[【参考文献】](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/tree/master/examples/references)(灵感来自 torchvision】(例如 CIFAR10 上的分类、ImageNet 和 Pascal VOC12 上的分割)。**
*   **分布式培训**

### Ignite 也支持分布式培训**，但是我们让用户来设置它的并行类型**:模型或数据。

例如，在数据分布式配置中，要求用户正确设置分布式过程组、包装模型、使用分布式采样器等。Ignite 处理度量计算:减少所有进程的值。

我们**提供了几个示例**(例如[分布式 CIFAR10](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10#distributed-training) )来展示如何在分布式配置中使用 Ignite。

**人气**

### 在撰写本文时，Ignite 大约有 **2.5k stars** ，根据 Github 的“用户”功能，有**被 205 个存储库使用。**一些荣誉奖包括:

来自 HuggingFace 的 Thomas Wolf 也在他的一篇博客文章中为图书馆留下了一些令人敬畏的反馈(谢谢，Thomas！):

“使用令人敬畏的 PyTorch ignite 框架和 NVIDIA 的 apex 提供的自动混合精度(FP16/32)的新 API，我们能够在不到 **250 行训练代码**中提取+3k 行比赛代码，并提供分布式和 FP16 选项！”

> Max LapanThis 是一本关于深度强化学习的书，其中第二版的例子是用 Ignite 编写的。

*   [Project MONAI:](https://web.archive.org/web/20220926085913/https://github.com/Project-MONAI/MONAI) 用于医疗保健成像的 AI 工具包。该项目主要关注医疗保健研究，旨在开发医学成像的 DL 模型，使用 Ignite 进行端到端培训。
*   关于其他用例，请看看 [Ignite 的 github 页面](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite#they-use-ignite)和它的“使用者”。

**何时使用 Ignite**

### 使用 Ignite API 的高度可定制模块，删除样板文件并标准化您的代码。

*   当您需要分解的代码，但不想牺牲灵活性来支持复杂的训练策略时
*   使用各种各样的实用工具，如度量、处理程序和记录器，轻松地评估/调试您的模型
*   **何时不使用 Ignite**

### 当有一个超级自定义 PyTorch 代码，其中 Ignite 的 API 是开销。

*   当纯 PyTorch API 或另一个高级库完全满足时
*   感谢您的阅读！ **Pytorch-Ignite** 由 [PyTorch 社区](https://web.archive.org/web/20220926085913/https://github.com/pytorch/ignite/graphs/contributors)用爱献给您！

**哲学**

[](https://web.archive.org/web/20220926085913/https://www.linkedin.com/in/wfalcon/)

### PyTorch Lightning 是 PyTorch 上的一个非常轻量级的包装器，它更像是一个**编码标准，而不是一个框架**。这种格式可以让你摆脱一大堆样板代码，同时**保持易于理解**。

钩子的使用在训练的每个部分都是标准的，这意味着你可以忽略内部功能的任何部分，直到如何向后传球——这非常灵活。

结果是一个框架，**给了研究人员、学生和制作团队终极的灵活性**去尝试疯狂的想法，而不必学习另一个框架，同时自动化掉所有的工程细节。

Lightning 还有另外两个更雄心勃勃的动机:**深度学习社区中研究的可重复性和最佳实践的民主化**。

**显著特征**

### 在不改变你的代码的情况下，在 CPU、GPU 或 TPUs 上训练！

*   唯一支持 TPU 训练的库(训练者(数量 _ TPU _ 核心=8))
*   琐碎的多节点训练
*   琐碎的多 GPU 训练
*   普通的 16 位精度支持
*   内置性能分析器(`Trainer(profile=True)`)
*   与 tensorboard、comet.ml、neptune.ai 等库的大量集成… ( `Trainer(logger=NeptuneLogger(...))`)
*   **团队**

### Lightning 有 90 多名贡献者和一个由 8 名贡献者组成的核心团队，他们确保项目以闪电般的速度向前推进。

**文档**

### [Lightning 文档](https://web.archive.org/web/20220926085913/https://pytorch-lightning.readthedocs.io/en/latest/)非常全面，但简单易用。

**API**

### 在核心部分，Lightning 有一个以两个对象`Trainer`和`LightningModule`T3**为中心的 API。**

培训师抽象出所有的工程细节，照明模块捕获所有的科学/研究代码。这种分离使得研究代码更具可读性，并允许它在任意硬件上运行。

**照明模块**

### 所有的**研究逻辑**进入`LightningModule`。

例如，在癌症检测系统中，这部分将处理主要的事情，如对象检测模型、医学图像的数据加载器等。

它将构建深度学习系统所需的核心**成分进行了分组**:

计算(初始化，向前)。

*   训练循环(training_step)中会发生什么。
*   验证循环中发生了什么(validation_step)。
*   测试循环中发生了什么(test_step)。
*   要使用的优化器(configure _ optimizers)。
*   要使用的数据(训练、测试、评估数据加载器)。
*   让我们看看文档中的例子，了解一下那里发生了什么。

如你所见，LightningModule **构建在纯 PyTorch 代码**之上，并简单地将它们组织在**的九个方法**中:

```py
import pytorch_lightning as pl

class MNISTExample(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()

        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss']
                                for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):

        avg_loss = torch.stack([x['test_loss']
                                for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=0.02)

 @pl.data_loader
    def train_dataloader(self):

        return DataLoader(
            MNIST(os.getcwd(), train=True, download=True,
                  transform=transforms.ToTensor()), batch_size=32)

 @pl.data_loader
    def val_dataloader(self):

        return DataLoader(
            MNIST(os.getcwd(), train=True, download=True,
                  transform=transforms.ToTensor()), batch_size=32)

 @pl.data_loader
    def test_dataloader(self):

        return DataLoader(
            MNIST(os.getcwd(), train=False, download=True,
                  transform=transforms.ToTensor()), batch_size=32)

```

***__init__():*** 定义本模型或多个模型，并初始化权重

*   ***forward():*** 您可以将它视为标准的 PyTorch forward 方法，但它具有额外的灵活性，可以在预测/推理级别定义您希望发生的事情。
*   ***training_step():*** 定义了训练循环中发生的事情。它结合了一个向前传球，损失计算，以及任何其他你想在训练中执行的逻辑。
*   ***validation _ step()*:**定义了验证循环中发生的事情。例如，您可以计算每个批次的损失或准确度，并将它们存储在日志中。
*   ***validation _ end()*:**验证循环结束后，你希望发生的一切。例如，您可能想要计算认证批次的平均损失或准确度
*   ***test_step()* :** 你希望每一批在推断时发生什么。你可以把你的测试时间增强逻辑或者其他东西放在这里。
*   ***【test _ end()*:**与 validation_end 类似，您可以使用它来聚合 test_step 期间计算的批处理结果
*   ***configure _ optimizer()*:**初始化一个或多个优化器
*   ***train/val/test _ data loader()*:**返回训练、验证和测试集的 PyTorch 数据加载器。
*   因为每个 PytorchLightning 系统都需要实现这些方法，所以很容易看到研究中到底发生了什么。

比如要了解一篇论文在做什么，你要做的就是看`LightningModule`的`training_step`！

这种可读性以及核心研究概念和实现之间的紧密映射是 Lightning 的核心。

> **教练**

### 这是深度学习的工程部分发生的地方。

在癌症检测系统中，这可能意味着你使用多少 GPU，当你停止训练时你何时保存检查点，等等。这些细节构成了许多研究的“秘方”，这些研究是深度学习项目的标准最佳实践(即:与癌症检测没有太大的相关性)。

请注意，`LightningModule`没有任何关于 GPU 或 16 位精度或早期停止或日志记录之类的东西。**所有这些都由教练自动处理。**

这就是训练这个模型的全部！培训师为您处理所有事情，包括:

```py
from pytorch_lightning import Trainer

model = MNISTExample()

trainer = Trainer()    
trainer.fit(model)

```

提前停止

*   自动记录到 Tensorboard(或 comet、mlflow、neptune 等)
*   自动检查点
*   更多(我们将在接下来的章节中讨论)
*   所有这些都是免费的！

**学习曲线**

### 因为`LightningModule`只是简单地重组纯 Pytorch 对象，一切都是“公开的”,所以**将 PyTorch 代码重构为 Lightning 格式是微不足道的。**

更多关于从纯 PyTorch 到 Lightning 转换的信息，请阅读[这篇文章](https://web.archive.org/web/20220926085913/https://towardsdatascience.com/how-to-refactor-your-pytorch-code-to-get-these-42-benefits-of-pytorch-lighting-6fdd0dc97538)。

**内置特性(开箱即用)**

### Lightning 提供了大量现成的高级功能。例如，一行程序可以使用以下内容:

随时间截断反向传播

```py
Trainer(gpus=8)

```

```py
Trainer(num_tpu_cores=8)

```

```py
Trainer(gpus=8, num_nodes=8, distributed_backend=’ddp’)

```

```py
Trainer(gradient_clip_val=2.0)

```

```py
Trainer(accumulate_grad_batches=12)

```

```py
Trainer(use_amp=True)

```

*   如果你想看完整的免费魔法特性列表，请点击这里。

```py
Trainer(truncated_bptt_steps=3)

```

**扩展能力/研究中集成的简单性**

### 拥有大量内置功能固然很好，但对于研究人员来说，重要的是不必学习另一个库，直接控制研究的关键部分，如数据处理，而无需其他抽象操作。

这种灵活的形式为培训和验证提供了最大的自由度。这个接口应该被认为是一个系统，而不是一个模型。该系统可能有多个模型(GANs、seq-2-seq 等)或只有一个模型，如这个简单的 MNIST 例子。

因此，研究人员可以自由地尝试他们想做的疯狂的事情，，并且只需要担心结果。

但是也许你需要更多的灵活性。在这种情况下，您可以这样做:

更改后退步骤的完成方式。

*   更改 16 位的初始化方式。
*   添加您自己的分布式培训方式。
*   添加学习率计划程序。
*   使用多个优化器。
*   更改优化器更新的频率。
*   还有很多很多东西。
*   在引擎盖下，Lightning 中的所有东西都被实现为可以被用户覆盖的钩子。这使得培训的每个方面都具有高度的可配置性，而这正是研究或生产团队所需要的灵活性。

但是等等，你会说…这对于你的用例来说太简单了？别担心，闪电是我在 NYU 和脸书人工智能研究所攻读博士学位时设计的，对研究人员来说尽可能灵活。

以下是一些例子:

需要**自己的后传球**？覆盖此挂钩:

*   需要**自己的放大器初始化**？覆盖此挂钩:

```py
def backward(self, use_amp, loss, optimizer):
    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
```

*   想要深入到添加**您自己的 DDP 实现**？覆盖这两个挂钩:

```py
def configure_apex(self, amp, model, optimizers, amp_level):
    model, optimizers = amp.initialize(
        model, optimizers, opt_level=amp_level,
    )

    return model, optimizers
```

像这样的钩子有 10 个，我们会根据研究人员的要求增加更多。

```py
def configure_ddp(self, model, device_ids):

    model = LightningDistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=True
    )
    return model

def init_ddp_connection(self):

    try:

        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        default_port = int(default_port) + 15000

    except Exception as e:
        default_port = 12910

    try:
        default_port = os.environ['MASTER_PORT']
    except Exception:
        os.environ['MASTER_PORT'] = str(default_port)

    try:
        root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
    except Exception:
        root_node = '127.0.0.2'

    root_node = self.trainer.resolve_root_node_address(root_node)
    os.environ['MASTER_ADDR'] = root_node
    dist.init_process_group(
        'nccl',
        rank=self.proc_rank,
        world_size=self.world_size
    )
```

底线是， **Lightning 对于新用户来说使用起来很简单，如果你是一名研究人员或从事前沿人工智能研究的生产团队**,它可以无限扩展。

**可读性和走向再现性**

### 正如我提到的，Lightning 的创建还有第二个更大的动机:可复制性。虽然真正的再现性需要标准代码、标准种子、标准硬件等，但 Lightning 以两种方式为可再现性研究做出了贡献:

为了**标准化 ML 代码**的格式，

*   **将工程与科学分离**以便该方法可以在不同的系统中进行测试。
*   结果是一个用于研究的表达性强、功能强大的 API。

如果每个研究项目和论文都是使用 LightningModule 模板实现的，那么就很容易发现发生了什么(但是可能不容易理解哈哈)

> **分布式培训**

### 闪电**让多 GPU 甚至多 GPU 多节点训练变得琐碎。**

例如，如果您想在多个 GPU 上训练上述示例，只需向训练器添加以下标志:

使用上述标志将在 4 个 GPU 上运行该模型。如果您想在 16 个 GPU 上运行，其中有 4 台机器，每台机器有 4 个 GPU，请将教练标志更改为:

```py
trainer = Trainer(gpus=4, distributed_backend='dp')    
trainer.fit(model)

```

并提交以下 SLURM 作业:

```py
trainer = Trainer(gpus=4, nb_gpu_nodes=4, distributed_backend='ddp')    
trainer.fit(model)

```

考虑到引擎盖下发生了多少事情，这简直太简单了。

```py

source activate $1

 export NCCL_DEBUG=INFO
 export PYTHONFAULTHANDLER=1

srun python3 mnist_example.py

```

有关 Pytorch lightning 分布式训练的更多信息，请阅读这篇关于[“如何使用 Pytorch 在 128 GPUs 上训练 GAN”的文章。](https://web.archive.org/web/20220926085913/https://towardsdatascience.com/how-to-train-a-gan-on-128-gpus-using-pytorch-9a5b27a52c73)

生产化

### Lightning 模型很容易部署，因为它们仍然是简单的 PyTorch 模型。这意味着我们可以利用 PyTorch 社区在支持部署方面的所有工程进展。

**人气**

### Pytorch Lightning 在 Github 上拥有超过 3800 颗星星，最近的下载量达到了 110 万次。更重要的是，该社区正在快速发展，有超过 90 名贡献者，许多来自世界顶级人工智能实验室，每天都在添加新功能。你可以在 [Github](https://web.archive.org/web/20220926085913/https://github.com/PyTorchLightning/pytorch-lightning/issues) 或者 [Slack](https://web.archive.org/web/20220926085913/https://pytorch-lightning.slack.com/join/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A#/) 上和我们交流。

**何时使用 PyTorch 闪电**

### Lightning 是为从事尖端研究的专业研究人员和生产团队而制造的。当你知道你需要做什么的时候，那是很棒的。这种关注意味着它为那些希望快速测试/构建东西而不陷入细节的人增加了高级特性。

*   **何时不使用 PyTorch 闪电**

### 虽然 lightning 是为专业研究人员和数据科学家设计的，但新来者仍然可以从中受益。对于新来者，我们建议他们使用纯 PyTorch 从头构建一个简单的 MNIST 系统。这将向他们展示如何建立训练循环等。一旦他们明白这是如何工作的，以及向前/向后传球是如何工作的，他们就可以进入闪电状态。

[](https://web.archive.org/web/20220926085913/https://www.linkedin.com/in/ethanwharris/)[](https://web.archive.org/web/20220926085913/https://github.com/MattPainter01)

[我们的博客部分将与其他部分略有不同，因为](https://web.archive.org/web/20220926085913/https://github.com/MattPainter01)**[火炬手](https://web.archive.org/web/20220926085913/https://github.com/pytorchbearer/torchbearer)即将结束**(有点)。特别是，**我们加入了 PyTorch-Lightning** 团队。这一举动源于在 NeurIPS 2019 上与威廉·法尔肯的一次会面，并于最近在 PyTorch 博客上宣布。

因此，我们认为我们应该写我们做得好的地方，我们做错的地方，以及我们为什么要转向闪电，而不是试图向你推销火炬手。

**我们做得好的地方**

### lib 变得非常受欢迎，在 GitHub 上获得了 500 多颗星，这远远超出了我们的想象。

*   我们成为 PyTorch 生态系统的一部分。对我们来说，这是一次重要的经历，让我们觉得自己是更广泛的社区中有价值的一部分。
*   我们已经建立了一套全面的内置回调和指标。这是我们的主要成功之一；使用 torchbearer，一行代码就可以实现许多强大的成果。
*   火炬手的一个重要特点是**使极端的灵活性**是*状态*对象。这是一个可变字典，包含核心训练循环使用的所有变量。通过在循环中不同点的回调中编辑这些变量，可以实现最复杂的结果。
*   火炬手拥有**良好的文件**对我们来说一直很重要。我们关注的是可以在你的浏览器中用 Google Colab 执行的示例文档。示例库非常成功，它提供了关于 torchbearer 更强大的用例的快速信息。
*   最后要注意的是，在过去的两年里，我们俩都在用火炬手进行我们的博士研究。我们认为这是一个成功，因为我们几乎**从来不需要为了原型化我们的想法而改变火炬手 API** ，即使是那些荒谬的想法！
*   **我们做错了什么**

### 使这个库如此灵活的*状态*对象也有问题。从任何其他地方访问库的任何部分的能力会像全局变量一样导致滥用。特别是，**一旦不止一个对象作用于它，确定状态对象中的特定变量是如何以及何时被改变的是具有挑战性的**。此外，为了使状态有效，你需要知道每个变量是什么，以及在哪个回调中可以访问它，所以**学习曲线很陡。**

*   火炬手本质上不适合分布式训练，甚至在某种程度上不适合低精度训练。既然状态的每一部分在任何时候都是可用的，那么如何将它分块并在设备间分配呢？PyTorch 可以以某种方式处理这个问题，因为火炬手可以在分发时使用，但目前还不清楚 state 在这些时候会发生什么。
*   **改变核心训练循环并非易事**。Torchbearer 提供了一种完全编写自己的核心循环的方法，但是您必须手动编写回调点，以确保所有内置的 Torchbearer 功能。与库的其他方面相比，再加上较低的文档标准，定制循环过于复杂，大多数用户可能完全不知道。
*   在攻读博士的同时管理一个开源项目变得比预期的更加困难。结果，该库的一些部分经过了彻底的测试并且是稳定的(因为它们对我们的博士工作很重要)，而其他部分则开发不足并且充满错误。
*   在我们最初的成长过程中，我们决定大幅改变核心 API 。这极大地改进了火炬手，但也意味着从一个版本到下一个版本需要大量的努力。这感觉是合理的，因为我们仍然是 1.0.0 之前的稳定版本，但它肯定会促使一些用户选择其他库。
*   **为什么我们要加入 PyTorch Lightning？**

### 我们愿意迁移到 Lightning 的第一个关键原因是它的受欢迎程度。有了 Lightning，我们**成为发展最快的 PyTorch 培训库**的一部分，这已经让它的许多竞争对手黯然失色。

*   我们此举的第二个关键原因，也是 Lightning 成功的一个关键部分，是**它是从头开始构建的，以支持分布式训练和低精度**，这两者在火炬手中实现都具有挑战性。在 Lightning 开发的早期阶段做出的这些实际考虑对现代深度学习实践者来说是非常宝贵的，而**在《火炬手》中进行改造将是一个挑战。**
*   此外，在 Lightning **我们将成为更大的核心开发团队的一部分。**这将使我们能够确保更高的稳定性，支持更广泛的用例，而不是像现在这样只有两个开发人员。
*   最终，我们始终相信，推动事情向前发展的最佳方式是与另一个图书馆合作。这是我们实现这一目标并帮助闪电成为 PyTorch 最好的培训库的机会。

可能有用

### 查看如何使用 Neptune 跟踪模型训练，这要归功于与以下软件的集成:
➡️[catalyst](https://web.archive.org/web/20220926085913/https://docs.neptune.ai/integrations-and-supported-tools/model-training/catalyst)➡️[fastai](https://web.archive.org/web/20220926085913/https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai) ➡️[py torch ignite](https://web.archive.org/web/20220926085913/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-ignite) ➡️[py torch lightning](https://web.archive.org/web/20220926085913/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning)
➡️[sko rch](https://web.archive.org/web/20220926085913/https://docs.neptune.ai/integrations-and-supported-tools/model-training/skorch)

(主观)比较和最终想法

## 雅各布·查肯

### 大部分是 ML 的人。构建 MLOps 工具，编写技术资料，在 Neptune 进行想法实验。

在这一点上，我想给一个…

非常感谢所有作者！

> 哇，这是很多第一手资料，我希望它能让你更容易选择适合你的图书馆。

当我和他们一起写这篇文章并仔细观察他们的库所提供的东西(并创建一些拉请求)时，我得到了自己的观点，我想在这里和你们分享。

[***斯科奇***](https://web.archive.org/web/20220926085913/https://github.com/skorch-dev/skorch)

如果你想要类似 sklearn 的 API，那么 **Skorch** 就是你的 lib。它经过了很好的测试和记录。实际上**给出了比我在撰写这篇文章之前所预期的**更多的灵活性，这是一个很好的惊喜。也就是说，这个图书馆的**重点不是尖端研究，而是生产应用。**我觉得它真的兑现了他们的承诺，并且完全符合设计初衷。我真的很尊重这样的工具/库。

[](https://web.archive.org/web/20220926085913/https://docs.fast.ai/)

 *长期以来，Fastai**一直是人们进入深度学习的绝佳选择。**它可以用 10 行近乎神奇的代码为您提供最先进的结果。但是库还有**的另一面，也许不太为人所知，它让你访问**低级 API**并创建自定义构建块**给研究人员和从业者实现非常复杂的系统的灵活性。也许是超级受欢迎的 fastai 深度学习课程在我的脑海中创造了这个库的错误形象，但我肯定会在未来使用它，特别是最近的 v2 预发布。****

***[Pytorch 点燃](https://web.archive.org/web/20220926085913/https://pytorch.org/ignite/)***

**点燃**是一种有趣的动物。有了它，**有点异国情调**(对我个人来说)，**引擎，事件和处理程序 API** ，你可以**做任何你想做的事情。它有大量开箱即用的功能，我完全理解为什么许多研究人员在日常工作中使用它。我花了一点时间来熟悉这个框架，但是你只需要停止用“回调术语”来思考，你就没事了。也就是说，API 对我来说不像其他一些库那样清晰。不过你应该去看看，因为这对你来说可能是个不错的选择。**

[***催化剂***](https://web.archive.org/web/20220926085913/https://catalyst-team.github.io/catalyst/)

在研究 **Catalyst** 之前，我认为它是一个创建深度学习管道的沉重(有点)框架。现在我的看法完全不同了。**它以一种美丽的方式将工程材料与研究分离开来**。纯 PyTorch 对象进入处理训练的训练器。它非常灵活，有一个单独的模块来处理强化学习。它也**给你很多现成的功能，当谈到可重复性，并在生产中服务模型。**还有我跟你说过的那些多级管道？您可以用最少的开销轻松创建它们。总的来说，我认为这是一个很棒的项目，很多人可以从使用它中受益。

[***py torch***](https://web.archive.org/web/20220926085913/https://github.com/PyTorchLightning/pytorch-lightning)

闪电也想把科学和工程分开，我认为它在这方面做得很好。有大量的内置功能使它更具吸引力。但是这个库有一点不同的是，它通过使深度学习研究实现可读来实现**的可重复性。**遵循 LightningModule 内部的逻辑真的很容易，其中培训步骤(以及其他内容)没有被抽象掉。我认为以这种方式交流研究项目会非常有效。**它很快变得非常受欢迎**，随着**火炬手的作者加入核心开发团队**，我认为**这个项目有一个光明的未来**在它前面，甚至是闪电一样的光明🙂

那么你应该选择哪一个呢？像往常一样，这要视情况而定，但我认为你现在有足够的信息来做出一个好的决定！

**阅读下一篇**

如何使用海王星跟踪 PyTorch 中的实验

## 4 分钟阅读| Aayush Bajaj |发布于 2021 年 1 月 19 日

4 mins read | Aayush Bajaj | Posted January 19, 2021

机器学习开发看起来很像传统的软件开发，因为它们都需要我们编写大量的代码。但其实不是！让我们通过一些要点来更好地理解这一点。

机器学习代码不会抛出错误(当然我说的是语义)，原因是，即使你在神经网络中配置了错误的方程，它仍然会运行，但会与你的预期混淆。用[安德烈·卡帕西](https://web.archive.org/web/20220926085913/https://karpathy.github.io/2019/04/25/recipe/)、*的话说，“神经网络无声无息地失败了”。*

*   机器学习代码/项目严重依赖结果的可重复性。这意味着，如果一个超参数被推动，或者训练数据发生变化，那么它会在许多方面影响模型的性能。这意味着你必须记下超参数和训练数据的每一个变化，以便能够重现你的工作。当网络很小时，这可以在一个文本文件中完成，但是如果是一个有几十或几百个超参数的大项目呢？文本文件现在不那么容易了吧！
*   机器学习项目复杂性的增加意味着复杂分支的增加，必须对其进行跟踪和存储以供将来分析。
*   机器学习也需要大量的计算，这是有代价的。你肯定不希望你的云成本暴涨。
*   有组织地跟踪实验有助于解决所有这些核心问题。海王星是一个完整的工具，可以帮助个人和团队顺利跟踪他们的实验。它提供了许多功能和演示选项，有助于更轻松地跟踪和协作。

Tracking experiments in an organized way helps with all of these core issues. Neptune is a complete tool that helps individuals and teams to track their experiments smoothly. It presents a host of features and presentation options that helps in tracking and collaboration easier.

[Continue reading ->](/web/20220926085913/https://neptune.ai/blog/how-to-keep-track-of-experiments-in-pytorch-using-neptune)

* * **