# 分布式培训:框架和工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/distributed-training-frameworks-and-tools>

深度学习的最新发展已经带来了一些令人着迷的最新成果，特别是在自然语言处理和计算机视觉等领域。成功的几个原因通常来自于大量数据的可用性和[不断增长的深度学习(DL)模型](https://web.archive.org/web/20230307085657/https://towardsdatascience.com/review-of-recent-advances-in-dealing-with-data-size-challenges-in-deep-learning-ac5c1844af73)。这些算法能够提取有意义的模式，并推导出输入和输出之间的相关性。但开发和训练这些复杂的算法可能需要几天，有时甚至几周，这也是事实。

为了解决这个问题，需要一种快速有效的方法来设计和开发新的模型。人们不能在单个 GPU 上训练这些模型，因为这将导致信息瓶颈。为了解决单核 GPU 上的信息瓶颈问题，我们需要使用多核 GPU。这就是**分布式培训**的想法出现的地方。

在本文中，我们将研究一些用于分布式培训的最佳框架和工具。但在此之前，让我们快速了解一下分布式培训本身。

## 分布式培训

DL 训练通常依赖于可伸缩性，可伸缩性简单地意味着 DL 算法学习或处理任意数量数据的能力。本质上，任何 DL 算法的可扩展性取决于三个因素:

## 

*   1 深度学习模型的规模和复杂性
*   2 训练数据量
*   3 包括 GPU 和存储单元等硬件在内的基础设施的可用性，以及这些设备之间的平稳集成

[分布式训练](/web/20230307085657/https://neptune.ai/blog/distributed-training)满足所有三个要素。它负责模型的大小和复杂性，批量处理训练数据，并在称为节点的多个处理器中拆分和分布训练过程。更重要的是，它大大减少了训练时间，使得迭代时间更短，从而使得实验和部署更快。

分布式培训有两种类型:

## 

*   1 数据并行训练

*   2 模型-平行训练

![Distributed training model parallelism vs data parallelism ](img/00fabf769c270c835db5397a2a8a852d.png)

*Distributed training model parallelism vs data parallelism | [Source](https://web.archive.org/web/20230307085657/https://towardsdatascience.com/deep-learning-on-supercomputers-96319056c61f)*

在数据并行训练中，数据根据可用于训练的节点数量划分为子集。并且在所有可用节点中共享相同的模型架构。在训练过程中，所有节点必须相互通信，以确保每个节点的训练相互同步。这是训练模型最有效的方法，也是最常见的做法。

在模型并行训练中，DL 模型根据可用节点的数量被分割成多个部分。每个节点都被馈送相同的数据。在模型并行训练中，DL 模型本身被分成不同的段，然后每个段被馈送到不同的节点。如果 DL 模型具有可以单独训练的独立组件，则这种类型的训练是可能的。请记住，每个节点必须在模型不同部分的共享权重和偏差方面保持同步。

在这两种类型的培训中，数据并行性是非常常用的，当我们发现分布式培训的框架时，您会发现无论模型并行性如何，它们都提供数据并行性。

## 为分布式培训选择正确框架的标准

在我们深入研究框架之前，在选择正确的框架和工具时，有几点需要考虑:

1.  **计算图类型:**整个深度学习社区主要分为两派，一派使用 PyTorch 或动态计算图，另一派使用 TensorFlow 或静态计算图。因此，大多数分布式框架都建立在这两个库之上已经不是什么新闻了。所以，如果你喜欢其中一个，那么你已经做了一半的决定。
2.  **培训成本**:当您处理分布式计算时，负担能力是一个关键问题，例如，一个涉及 BigGAN 培训的项目可能需要大量的 GPU，成本可能会随着数量的增加而成比例增加。因此，价格适中的工具总是正确的选择。
3.  **培训类型**:根据您的培训需求，即数据并行或模型并行，您可以选择其中一种工具。
4.  **效率**:这个基本上是指你需要写多少行才能启用分布式训练，越少越好。
5.  灵活性:你选择的框架可以跨平台使用吗？尤其是当你需要在内部或者云平台上进行培训的时候。

## 分布式培训框架

现在，让我们讨论一些提供分布式培训的图书馆。

在 Neptune 中，你可以[跟踪来自许多进程](https://web.archive.org/web/20230307085657/https://docs.neptune.ai/how-to-guides/neptune-api/distributed-computing)的数据，特别是在不同机器上运行的数据。

### 1\. PyTorch

![Distributed training: PyTorch](img/d0a3e6c2c9df5b69447c30c019e6fbce.png)

*Distributed training: PyTorch | [Source](https://web.archive.org/web/20230307085657/https://github.com/pytorch/pytorch)*

PyTorch 是脸书开发的最流行的深度学习框架之一。它是最灵活、最容易学习的框架之一。PyTorch 允许您非常有效地创建和实现神经网络模块，通过其分布式训练模块，您可以用几行代码轻松实现并行训练。

PyTorch 提供了多种执行分布式培训的方法:

1.  [**nn。DataParallel**](https://web.archive.org/web/20230307085657/https://pytorch.org/docs/stable/nn.html#dataparallel) **:** 这个包可以让你在一台有多个 GPU 的机器上进行并行训练。一个优点是它需要最少的代码。
2.  [**nn。DistributedDataParallel**](https://web.archive.org/web/20230307085657/https://pytorch.org/docs/stable/nn.html#distributeddataparallel) :这个包允许你在多台机器内跨多个 GPU 进行并行训练。配置培训流程还需要一些额外的步骤。
3.  [**torch . distributed . RPC**](https://web.archive.org/web/20230307085657/https://pytorch.org/docs/stable/rpc.html)**:**这个包允许你执行模型并行策略。如果您的模型很大，并且不适合单个 GPU，这将非常有效。

#### 优势

1.  很容易实现。
2.  PyTorch 非常用户友好。
3.  提供现成的数据并行和模型并行方法。
4.  大多数云计算平台都支持 PyTorch。

#### 什么时候用 PyTorch？

在以下情况下，您应该选择 PyTorch:

*   你有大量的数据，因为数据并行很容易实现。

![Distributed training: DeepSpeed](img/78a2c31a8f4e88b0180b61ffd684858f.png)

*Distributed training: DeepSpeed | [Source](https://web.archive.org/web/20230307085657/https://github.com/microsoft/DeepSpeed)*

PyTorch 分布式培训专门研究数据并行。DeepSpeed 构建于 PyTorch 之上，目标是其他方面，即模型并行性。DeepSpeed 由微软开发，旨在为大规模模型提供分布式训练。

当训练具有数万亿个参数的模型时，DeepSpeed 可以有效地应对内存挑战。它减少了内存占用，同时保持了计算和通信效率。有趣的是，DeepSpeed 提供了 3D 并行性，通过它你可以分发数据、模型和管道，这基本上意味着现在你可以训练一个大型的、消耗大量数据的模型，就像 GPT-3 或图灵 NLG 一样。

#### 优势

1.  模型扩展到数万亿个参数。
2.  训练速度提高 10 倍。
3.  民主化人工智能，这意味着用户可以在单个 GPU 上运行更大的模型，而不会耗尽内存。
4.  压缩训练允许用户通过减少计算注意力操作所需的内存来训练注意力模型。
5.  易学易用。

#### 何时使用 DeepSpeed？

在以下情况下，您应该选择 DeepSpeed:

*   你想做数据和模型并行。
*   如果你的代码库是基于 PyTorch 的。

![Distributed training: TensorFlow](img/3dd56b1b7b5215ccb30023d87e54a36f.png)

*Distributed training: TensorFlow | [Source](https://web.archive.org/web/20230307085657/https://github.com/tensorflow/tensorflow)*

TensorFlow 由 Google 开发，它支持分布式训练。它使用数据并行技术进行训练。您可以通过使用 **tf.distribute** API 来利用 TensorFlow 上的分布式培训。该 API 允许您根据自己的需求配置培训。默认情况下，TensorFlow 只使用一个 GPU，但 tf.distribute 允许您使用多个 GPU。

TensorFlow 提供了三种主要类型的分布式培训策略:

1.  **TF . distribute . mirroredstrategy()**:这个简单的策略允许你在一台机器上的多个 GPU 之间分配训练。这种方法也称为同步数据并行。值得注意的是，每个工人节点将有自己的一套梯度。这些梯度然后被平均并用于更新模型参数。

2.  **TF . distribute . multiworkermirroredstrategy()**:这个策略允许你将训练分布到多台机器和单台机器上的多个 GPU 上。所有操作都类似于 tf.distribute.MirroredStrategy()。这也是一种同步数据并行方法。

3.  **TF . distribute . experimental . parameter server strategy()**:这是一种异步数据并行方法，在多台机器上按比例放大模型训练是常见的做法。在这种策略中，参数存储在参数服务器中，工人相互独立。这种策略可以很好地扩展，因为工作节点不需要等待彼此的参数更新。

#### 优势

1.  巨大的社区支持。
2.  这是一个静态的编程范例。
3.  与谷歌云和其他基于云的服务非常好地集成。

#### 什么时候使用分布式张量流？

您应该使用分布式张量流:

*   如果要做数据并行。
*   如果你喜欢与动态相比的静态编程范式。
*   如果你在谷歌云生态系统中，因为 TensorFlow 针对 TPU 进行了很好的优化。
*   最后，如果您有大量数据并且需要高处理能力。

![Distributed training: TensorFlow](img/d0bfe20456f8f26a59836a0309809792.png)

*Distributed training: TensorFlow | [Source](https://web.archive.org/web/20230307085657/https://github.com/tensorflow/tensorflow)*

Mesh Tensorflow 也是 Tensorflow 分布式训练的扩展，但专门设计用于在张量处理单元(TPUs)上训练大型 DL 模型，AI 加速类似于 GPU，但速度更快。虽然 Mesh TensorFlow 可以执行数据并行，但它旨在解决参数无法在一台设备上安装的大型模型的分布式训练。

Mesh TensorFlow 受同步数据并行方法的启发，即每个工人都参与每个操作。除此之外，所有的工人将有相同的程序，它使用像 Allreduce 集体沟通。

#### 优势

1.  它可以训练具有数百万和数十亿参数的大型模型，如:GPT-3，GPT-2，伯特，等等。
2.  工作人员潜在的低延迟。
3.  良好的 TensorFlow 社区支持。
4.  谷歌 TPU 豆荚的可用性。

#### 什么时候使用网格张量流？

应该使用网格张量流:

*   如果你想做模型并行。
*   如果你想开发巨大的模型和实践快速成型。
*   如果您特别是在处理大量数据的自然语言处理领域工作。

![Distributed training: TensorFlow](img/d0bfe20456f8f26a59836a0309809792.png)

*Distributed training: TensorFlow | [Source](https://web.archive.org/web/20230307085657/https://github.com/tensorflow/tensorflow)*

**Apache Spark** 是最知名的开源大数据处理平台之一。它允许用户进行各种与数据相关的工作，如数据工程、数据科学和机器学习。我们已经知道张量流是什么了。但是如果你想在 Apache Spark 上使用 TensorFlow，那么你必须使用 TensorFlowOnSpark。

TensorFlowOnSpark 是一个机器学习框架，允许您在 Apache Spark 集群和 Apache Hadoop 上执行分布式训练。它是由雅虎开发的。该框架允许分布式训练和推理，对共享网格上的现有 TensorFlow 代码进行最小的代码改变。

#### 优势

1.  允许使用现有 TensorFlow 程序轻松迁移到 Spark 集群。
2.  代码中的更改更少。
3.  所有 TensorFlow 功能都可用。
4.  Spark 和 TensorFlow 可以分别高效地推送和拉取数据集。
5.  云开发在 CPU 或 GPU 上简单高效。
6.  可以轻松创建培训管道。

#### 何时使用 TensorFlowOnSpark？

您应该使用 TensorflowOnSpark:

*   如果您的工作流基于 Apache Spark，或者您更喜欢 Apache Spark。
*   如果你的首选框架是 TensorFlow。

![Distributed training: BigDL](img/531f1b0d9edb7b849958cf20ce03bb8b.png)

*Distributed training: BigDL | [Source](https://web.archive.org/web/20230307085657/https://github.com/intel-analytics/BigDL)*

BigDL 也是 Apache Spark 分布式培训的开源框架。它是由英特尔开发的，允许 DL 算法运行 Hadoop 和 Spark 集群。BigDL 的一大优势是，它可以帮助您轻松地构建和处理端到端管道中的生产数据，用于数据分析和深度学习应用程序。

BigDL 提供了两个选项:

1.  您可以像使用 Apache Spark 为数据工程、数据分析等提供的任何其他库一样直接使用 BigDL。
2.  您可以在 Spark 生态系统中横向扩展 python 库，如 PyTorch、TensorFlow 和 Keras。

#### 优势

1.  **端到端管道**:如果您的大数据杂乱而复杂，这通常是在实时数据流的情况下，那么采用 BigDL 是合适的，因为它在端到端管道中集成了数据分析和深度学习。
2.  **效率**:Spark BigDL 采用跨不同组件的集成方法，使得所有组件的开发、部署和运营直接、无缝且高效。
3.  **通信和计算**:由于所有的硬件和软件都是缝合在一起的，所以它们运行流畅，没有任何中断，使得不同工作流之间的通信清晰，计算速度更快。

#### 什么时候使用 BigDL？

您应该使用 BigDL:

*   如果您想开发一个 Apache Spark 工作流，
*   如果您的首选框架是 PyTorch。
*   如果你想持续集成所有组件，如数据挖掘、数据分析、机器学习等等。

![Distributed training: Horovod](img/f5d3f77c42888c6962290ea53a125147.png)

*Distributed training: Horovod | [Source](https://web.archive.org/web/20230307085657/https://github.com/horovod/horovod)*

Horovod 是优步在 2017 年推出的。这是一个开源项目，专门用于分布式培训。它是米开朗基罗的内部组件，米开朗基罗是优步用来实现其 DL 算法的深度学习工具包。Horovod 利用数据并行分布式培训，这使得扩展变得非常容易和高效。它还可以在大约 5 行 python 代码中扩展到数百个 GPU。这个想法是为单个 GPU 编写一个训练脚本，Horovod 可以将其扩展到多个并行的 GPU 上进行训练。

Horovod 是为 Tensorflow、Keras、Pytorch 和 Apache MXNet 等框架构建的。这是易于使用和快速。

#### 优势

1.  如果熟悉 Tensorflow、Keras、Pytorch 和 Apache MXNet，很容易学习和实现。
2.  如果您使用 Apache Spark，那么您可以在一个管道上统一所有的进程。
3.  良好的社区支持。
4.  它很快。

#### 何时使用 Horovod？

您应该使用 Horovod:

*   如果您希望在多个 GPU 之间快速扩展单个 GPU 脚本。
*   如果你使用微软 Azure 作为你的云计算平台。

![Distributed training: Ray](img/cde47843a938193e3e4f343ae066ece5.png)

*Distributed training: Ray | [Source](https://web.archive.org/web/20230307085657/https://github.com/ray-project/ray)*

Ray 是构建在 Pytorch 之上的另一个用于分布式培训的开源框架。它提供了在任何云提供商上启动 GPU 集群的工具。与我们迄今为止讨论的任何其他库不同，Ray 非常灵活，可以在任何地方工作，如 Azure、GCD、AWS Apache Spark 和 Kubernetes。

Ray 在其捆绑包中提供了以下库，用于超参数调整、强化学习、深度学习、缩放等等:

1.  [调优](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/tune.html):可伸缩超参数调优。
2.  [RLlib](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/rllib/index.html) :分布式强化学习。
3.  [Train](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/train/train.html) :分布式深度学习，目前测试版。
4.  [数据集](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/data/dataset.html):分布式数据加载和计算，目前处于测试版。
5.  [发球](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/serve/index.html):可伸缩可编程发球。
6.  [工作流程](https://web.archive.org/web/20230307085657/https://docs.ray.io/en/master/workflows/concepts.html):快速、持久的应用流程。

除了这些库之外，Ray 还集成了第三方库和框架，允许您以最少的代码更改来开发、培训和扩展您的工作负载。下面给出了集成库的列表:

1.  气流
2.  课堂视觉
3.  达斯克
4.  弗兰贝
5.  霍罗沃德
6.  拥抱面部变形金刚
7.  英特尔分析动物园
8.  约翰·斯诺实验室
9.  莱特格姆
10.  路德维希艾
11.  三月
12.  莫丁(莫丁)
13.  插入记号
14.  PyTorch 闪电
15.  RayDP
16.  scikit 很少学习不在场证明
17.  XGBoost
18.  优势
19.  它支持 Jupyter 笔记本

#### 它使您的代码在单台和多台机器上并行运行

1.  它集成了多个框架和库。
2.  它适用于所有主要的云计算平台
3.  什么时候用雷？
4.  你应该用雷:

#### 如果你想进行分布式强化学习

如果您想要执行分布式超参数调整

1.  如果您想在不同的机器上使用分布式数据加载和计算。
2.  如果你想为你的应用服务。
3.  分布式培训的云平台
4.  到目前为止，我们已经讨论了可用于支持分布式培训的框架和库。现在，让我们来讨论和探索云平台，在这里您可以使用硬件来高效地训练您的 DL 模型。但在此之前，让我们列出一些标准，让您能够根据自己的需求选择最佳的云平台。

## **硬件和软件支持:**学习和理解这些平台提供的硬件(如 GPU、TPU、存储单元等)非常重要。除此之外，你还应该看到他们提供的 API，这样(取决于你的项目)你就可以访问托管设施、容器、数据分析工具等等。

可用性区域:可用性区域是云计算中的一个重要因素，它为用户提供了在世界任何地方建立和部署项目的灵活性。用户也可以随时转移他们的项目。

1.  **定价:**平台是根据你的使用情况收费，还是提供基于订阅的模式。
2.  现在，让我们讨论云计算选项。我们将讨论两个极其可行的即用型实验笔记本平台和三个最流行的云计算服务。
3.  Google Colab 是中小型项目中最可靠和最容易使用平台之一。Google Colab 的一个好处是，你可以轻松地连接到 Google Cloud，并且可以使用上面提到的任何 python 库。它提供三种型号:

Google Colab 是免费的，它可以让你访问 GPU 和 TPU。但是你可以使用有限的存储和内存。一旦其中任何一个超过，程序就会停止。

![Magic quadrant for cloud infrastructure as a service](img/d0fd6173b31e5a06cc67914295f80c83.png)

*Magic quadrant for cloud infrastructure as a service | [Source](https://web.archive.org/web/20230307085657/https://www.c-sharpcorner.com/article/top-10-cloud-service-providers/)*

![Distributed training: Google Colab](img/dfcfbbab56f0b1df56fd2c824fdf3a96.png)

*Distributed training: Google Colab | [Source](https://web.archive.org/web/20230307085657/https://colab.research.google.com/)*

**Google Colab Pro** 是 Google Colab 的订阅版本，在这里你有额外的内存和存储空间。你可以运行一个很重的模型，但是它也是有限的。

1.  Google Colab Pro + 是一种基于订阅模式的新服务，也很贵。它提供了更快的 GPU 和 TPU 以及额外的内存，因此您可以在相当大的数据集上运行相当大的模型。
2.  下面给出的是三者的官方对比。
3.  AWS SageMaker 是最受欢迎和最古老的分布式培训云计算平台之一。它与 Apache MXNet、Pytorch 和 TensorFlow 非常好地集成在一起，允许您轻松部署深度学习算法，并减少代码修改。SageMaker API 有 [18+机器学习算法](https://web.archive.org/web/20230307085657/https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)，其中一些算法是从零开始重写的，使整个过程变得可扩展和简单。这些内置算法经过优化，可以充分利用硬件。

SageMaker 还有一个集成的 Jupyter 笔记本，允许数据科学家和机器学习工程师在旅途中构建和开发管道算法，并允许您直接在托管环境中部署它们。您可以从 SageMaker Studio 或 SageMaker 控制台根据您的需求和偏好配置硬件和环境。所有的托管和开发都按照每分钟的**使用量计费。**

![Distributed training: AWS SageMaker](img/06a002f4b3167ba7c82b36dca7790a62.png)

*Distributed training: AWS SageMaker | [Source](https://web.archive.org/web/20230307085657/https://nub8.net/machine-learning-with-amazon-sagemaker/)*

AWS SageMaker 提供数据并行和模型并行分布式培训。事实上，SageMaker 还提供了一种混合训练策略，在这种策略中，您可以同时使用模型和数据并行性。

谷歌云计算是谷歌在 2010 年开发的，旨在加强他们自己的平台，如谷歌搜索引擎和 Youtube。渐渐地，他们开始向公众开放源代码。谷歌云计算提供了所有谷歌平台使用的相同基础设施。

Google 云计算为 TensorFlow、Pytorch、Scikit-Learn 等库提供内置支持。此外，除了在工作流程中配置 GPU 之外，您还可以添加 TPU 来加快培训过程。就像我之前提到的，你可以将你的谷歌实验室连接到谷歌云平台，并访问它提供的所有功能。

![Distributed training: AWS SageMaker](img/bcc6223860423af70ee2663afe2a5a74.png)

*Distributed training: AWS SageMaker | [Source](https://web.archive.org/web/20230307085657/https://aws.amazon.com/blogs/machine-learning/the-aws-deep-learning-ami-now-with-ubuntu/)*

![Distributed training: Google Cloud Computing](img/5e891dde35f0099f14209ff5b1cb346c.png)

*Distributed training: Google Cloud Computing | [Source](https://web.archive.org/web/20230307085657/https://cloud.google.com/)*

它提供的一些功能有:

计算(虚拟硬件，如 GPU 和 TPU)

储物桶，

1.  数据库
2.  建立工作关系网
3.  管理工具
4.  安全性
5.  物联网
6.  API 平台
7.  托管服务
8.  值得注意的是，与 AWS 相比，GCP 的可用性区域较少，但成本也较低。
9.  微软 Azure 是另一个非常受欢迎的云计算平台。OpenAI 的一个流行语言模型 GPT-3 就是在 Azure 中训练出来的。它还提供了[数据并行](https://web.archive.org/web/20230307085657/https://docs.microsoft.com/en-us/azure/machine-learning/concept-distributed-training#data-parallelism)和[模型并行](https://web.archive.org/web/20230307085657/https://docs.microsoft.com/en-us/azure/machine-learning/concept-distributed-training#model-parallelism)方法，并支持 TensorFlow 和 Pytorch。事实上，如果你想优化计算速度，你也可以利用优步的 Horovod。

Azure 机器学习服务面向编码人员和非编码人员。它只是提供了一个拖放方法，可以优化您的工作流程。它还通过自动机器学习减少了手动工作，可以帮助您开发更智能的工作原型。

![Distributed training: Google Cloud Computing](img/8ae55a95010059dcdbb8e0543c79ccd6.png)

*Distributed training: Google Cloud Computing | Source: Author*

![Distributed training: Microsoft Azure](img/62f6df5ae7a4d384c86c449fc4a91dd6.png)

*Distributed training: Microsoft Azure | [Source](https://web.archive.org/web/20230307085657/https://medium.com/analytics-vidhya/azure-machine-learning-service-part-1-80e43e4af71b)*

Azure Python SDK 还允许您在任何 Python 环境中进行交互，如 Jupyter 笔记本、Visual Studio 代码等等。在提供服务方面，它与 AWS 和 GCP 非常相似。这些是 Azure 提供的服务:

人工智能、机器学习和深度学习

计算能力(GPU)

1.  分析学
2.  区块链
3.  容器
4.  数据库
5.  开发者工具
6.  DevOps
7.  物联网
8.  混合现实
9.  移动的
10.  网络等等
11.  让我们一起来比较一下这三个主要工具，让你在选择时有一个更好的视角。
12.  云平台对照表

![Distributed training: Microsoft Azure](img/beb16a7fcd00d0ad73e6564db67325d3.png)

*Distributed training: Microsoft Azure | [Source](https://web.archive.org/web/20230307085657/https://docs.microsoft.com/en-us/azure/azure-portal/azure-portal-dashboards)*

最后的想法

### 在本文中，我们看到了不同的库和工具，它们可以帮助您为自己的深度学习应用程序实现分布式培训。请记住，所有的库都是好的，并且在它们所做的事情上非常有效，最终，这一切都归结于您的偏好和需求。

![Comparison table for cloud platform](img/7f4189da541b7fe12fe77922030e1c6f.png)

*Comparison table for cloud platform | [Source](https://web.archive.org/web/20230307085657/https://medium.com/georgian-impact-blog/comparing-google-cloud-platform-aws-and-azure-d4a52a3adbd2)*

## 您一定已经注意到，所有讨论的框架都主要以某种方式集成了 Pytorch 和 TensorFlow。这个特质很容易帮你隔离选择的框架。一旦你的框架确定下来，你就可以看看它的优势，决定哪种分布式培训工具最适合你。

我希望你喜欢这篇文章。如果你想尝试我们讨论过的所有框架，那么请点击教程链接。

感谢阅读！

参考

Thanks for reading!

### References