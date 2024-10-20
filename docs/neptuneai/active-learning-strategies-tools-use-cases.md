# 主动学习:策略、工具和真实世界用例

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/active-learning-strategies-tools-use-cases>

在本文中，您将了解到:

*   什么是主动学习，
*   为什么我们需要主动学习
*   它是如何工作的
*   有哪些技巧
*   在现实世界中的应用
*   以及哪些框架可以帮助主动学习。

我们开始吧！

## 什么是主动学习？

主动学习是[机器学习](https://web.archive.org/web/20221117203617/https://en.wikipedia.org/wiki/Machine_learning)的一种特殊情况，在这种情况下，学习算法可以交互式地询问用户(或一些其他信息源)以用期望的输出来标记新的数据点。在统计学文献中，它有时被称为[最优实验设计](https://web.archive.org/web/20221117203617/https://en.wikipedia.org/wiki/Optimal_experimental_design)。–[来源](https://web.archive.org/web/20221117203617/https://en.wikipedia.org/wiki/Active_learning_(machine_learning)#:~:text=Active%20learning%20is%20a%20special,also%20called%20optimal%20experimental%20design.)

通过选择看似最重要的数据点，在保持监督/标记数据集的数量最小的同时，创建一个像样的机器学习模型是一项重要的技术。

在贴标困难或耗时的情况下，也考虑这种技术。被动学习，或者由人类先知创建大量标记数据的传统方式，在工时方面需要巨大的努力。

在一个成功的主动学习系统中，算法能够通过一些定义的度量来选择最具信息量的数据点，随后将它们传递给人类标记器，并逐步将它们添加到训练集中。下图所示为示意图。

![Active learning](img/fef5ff2df227cc7e59f660643f0a8ed9.png)

*Diagram of active learning system | Source: Author*

## 为什么我们需要主动学习？

### 动机

主动学习的想法是受已知概念的启发，即并非所有数据点对于训练模型都同样重要。看看下面的数据就知道了。这是两个集合的集群，中间有一个决策边界。

![A cluster of two sets](img/4004a8d21a967ec073e23f2caebff0d0.png)

*A cluster of two sets with a decision boundary in between | Source: Author*

现在假设一个场景，有超过数万个数据点，没有任何标签可以学习。手动标记所有这些点会很麻烦，甚至非常昂贵。为了减轻这种痛苦，如果从一批数据中选择一个随机的数据子集，然后标记为模型训练，最有可能的是，我们最终得到的是一个性能低于标准的模型，如下图所示。问题是，这种随机采样产生的决策边界会导致较低的准确性和其他降低的性能指标。

![Sub-par performance](img/6d5f5cd19622d262d89f14070ad762b5.png)

*A model with sub-par performance | Source: Author*

但是，如果我们设法选择决策边界附近的一组数据点，并帮助模型有选择地学习，会怎么样呢？这将是从给定的未标记数据集中选择样本的最佳方案。这就是主动学习概念的起源和演变。

![Scenario of NLP training](img/bccabbe2622cfe015fc25cc0db250017.png)

*Possible scenario of NLP training | Source: Author*

您在上面看到的插图可以很容易地应用于 NLP 培训中的场景，如获取相关的标注数据集以进行词性标注和命名实体识别等。，可能是一个挑战。

### 数据标签成本

为大规模训练创建标记数据是非常昂贵和耗时的。Google Cloud Platform/1000 单位数据标签的定价详情如下所示。通常情况下，每个实例应该有 2-3 个数据贴标机，以确保更好的标签质量。第 1 层高达 50K 单位，第 2 层高于该容量。

![The pricing details of Data Labelling](img/c7f1a20edcbe0805559dde5d9cbf7b6f.png)

*The pricing details of Data Labelling by Google Cloud Platform/1000 units | Source*

通过查看上面显示的成本，您可以大致了解如果您要标记 100，000 张图像以用于涉及边界框的专门 ML 模型的训练，典型的标记成本是多少。比方说，每幅图像大约有 5 个边界框和两个贴标机。成本可能是 112，000 美元。但是同样，这个成本可能大也可能小，这取决于您的组织和项目预算的大小。

让我们看看医疗保健领域可能发生的另一种情况。想象一下，我们需要标记 10，000 个医学图像，每个图像平均需要 15 个语义分段对象和 3 个标记器来提高准确性。费用可能是 391，500 美元！

我想现在你有一个想法，数据标签的成本很容易飙升。上面提到的数字是非常真实的场景。如果我们在大型图像数据集上训练大型语言模型或基于视觉的模型，这可能会更高。

无论哪种方式，如果您使用内部资源或外部贴标服务，贴标时间将会很长，这是您希望避免的。通常情况下，数据需要由熟悉该领域的人来标记。

主动学习可以大大绕过这些挑战。因此，在某些领域有很大的推动力，如大型 NLP 系统的培训，自动驾驶等。，也就是采用主动学习。以[英伟达](https://web.archive.org/web/20221117203617/https://medium.com/nvidia-ai/scalable-active-learning-for-autonomous-driving-a-practical-implementation-and-a-b-test-4d315ed04b5f)观察的自动驾驶系统中的数据标注场景为例。

一个简单的粗略计算表明，每天行驶 8 小时的 100 辆汽车的车队将需要超过 100 万个贴标机以 30fps 的速度注释所有摄像机的所有帧，以进行对象检测。这完全不切实际。

我想我不需要再强调所涉及的成本——金钱和时间！如果还不服气，就想想可能的碳足迹吧！

### 边缘场景导致的问题

[边缘情况](https://web.archive.org/web/20221117203617/https://medium.com/@livewithai/the-significance-of-edge-cases-and-the-cost-of-imperfection-as-it-pertains-to-ai-adoption-dc1cebeef72c)场景可能是无害的，如下图所示，算法混淆了小狗和纸杯蛋糕。或者，它们可能会像一辆自动驾驶卡车一样危险，导致重大事故，因为它未能检测到夜间穿过道路的深色车辆！这些边缘情况故障看似很少发生，但可能会迅速导致具有高成本影响的重大负面结果。

![Confusion between a puppy and a cupcake](img/75546136c2686a6e488ecbd8f78a797f.png)

*The algorithm is confused between a puppy and a cupcake | [Source](https://web.archive.org/web/20221117203617/https://medium.com/@livewithai/the-significance-of-edge-cases-and-the-cost-of-imperfection-as-it-pertains-to-ai-adoption-dc1cebeef72c)*

处理训练数据中的边缘情况的数据点的数量很少，这最终导致 ML 模型不能在其上训练。

![ Bounding box detections](img/ff5c2203b4674849e62a0c5148d36888.png)

*Example bounding box detections for cars and pedestrians for a day & night image | [Source](https://web.archive.org/web/20221117203617/https://blogs.nvidia.com/blog/2019/06/19/drive-labs-distance-to-object-detection/)*

主动学习系统可以被训练来识别边缘案例，以便它们可以被人类有效地标记，并且可以被馈送到具有足够表示甚至更高权重表示的标记数据池！关于自动驾驶系统中主动学习的更详细的内容将在后面的章节中给出。

## 主动学习如何工作:情景

所以…迷因分开！让我们深入探究一下主动学习是如何工作的。

在经典的主动学习周期中，该算法选择最有价值的数据实例(也可能是边缘情况),并要求它们由人来标记。然后将这些新标记的实例添加到训练集中，并重新训练模型。这些数据实例的选择是通过研究人员通常称之为查询的过程进行的。引用文章[中的一句话基于查询的学习](https://web.archive.org/web/20221117203617/https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_688):

***基于查询的学习是一种主动的学习过程，学习者与教师进行对话，教师应要求提供关于所学概念的有用信息。***

上面引用中提到的老师可能是一个基本的 ML 模型或一个简单的算法，通常事先用一小部分标记数据进行训练。如何形成查询是开发主动学习算法的关键。因此，无论您是在寻找一个模糊案例还是一个边缘案例来学习，它都与您如何设计查询格式密切相关。

在文献中有两种典型的主动学习场景:基于查询合成的和基于采样的。

### 查询合成

它主要基于这样的假设，即靠近分类边界的实例更不明确，标记这些实例将为学习模型提供最多的信息。因此，我们通过创建新的点或者利用基于欧几里德距离的技术选择已经最接近它的点来查询决策边界附近的点。

#### 该算法

1.  第一步是创建一个带有标签数据样本的简单模型。让我们假设一个二元分类的场景，假设 X[1]X[2]是属于两个独立类的两个点。通过使用一种类似于二叉查找树的方法，我们可以通过人类先知查询其中点的标签来找到接近决策边界的实例。这里的中点是指连接点 X[1]X[2]的直线的欧几里德中点。考虑这个例子。

让我们取属于由 X1 (2，5，4)，X2 (6，4，3)定义的不同类的两个点。因为它们有三个特征，找到它们的中点是一个简单的欧几里德运算，它给出中点为

X [中点] = (X [1] + X [2] )/2

X [中点] = (4，4.5，3.5)

因此，如果我们让点(X1 + X2)/2 的标签被查询，那么我们知道我们需要去哪个方向进行进一步的查询，并在非常接近决策边界的实例上着陆。

2.  但是我们如何知道我们是否到达了决策边界呢？为此，我们需要定义相反类别之间概率分数的差异。如果分数差低于这个值，我们就停止查询。

3.  但是在这里，我们冒着只在本地邻居中查询的风险。因此，为了在附近得到更多的点，我们需要尝试更多的技巧，如下所述。但是当然，如果有足够多的点分布在特征空间中，这在某种程度上是可以解决的。

4.  为了在局部邻域之外进行查询，我们可以添加一个相对于相对类的最近对的正交向量，然后找出它的中点。但是如何做到这一点呢？嗯，我们可以再次找出对的中点，然后通过正交添加一条线找到它附近的另一个点。这条线的长度可以是父点之间的欧几里德距离的数量级。该过程简要说明如下。

![Query beyond the local vicinity](img/993873679d01cbc30d57406ad84579cc.png)

*The procedure* of q*uery beyond the local vicinity | Source: Author*

5.  下一步是(从当前模型中)获取该点的标签，并在另一个类中获取最近的未标记点，并再次重复该过程。这样，我们可以生成一堆更接近当前模型决策边界的数据实例。然后从人类先知那里获得这些生成点的标签，并用这些附加数据重新训练先前的模型。再次重复整个过程，直到达到我们的模型性能目标。

![Visualization for generating queries](img/32f1cad686ec13627f2a1a6a6c8d2133.png)

*Visualization for generating queries by finding an opposite pair close to the hyperplane | [Source](https://web.archive.org/web/20221117203617/https://www.sciencedirect.com/science/article/abs/pii/S0925231214008145?via%3Dihub)*

上述方法在理论上非常有效。但是，如果我们正在处理一个复杂的语言模型或计算机视觉用例，其中生成的查询点对人类的 oracle 没有任何意义，该怎么办呢？

如果这是一项计算机视觉任务，这些生成的实例可能无法被人类先知识别。因此，我们从未标记的集合中选择最近的邻居并查询其标记。最近邻搜索可以通过简单的欧几里德距离测量或基于余弦相似性。

我希望你们都熟悉用于手写数字分类的 MNIST 数据集。请看看下面显示的一组图像。顶部显示的是 MNIST 数据集中手写数字的随机样本。底部的一个显示了通过查询和最近邻搜索从同一数据集中选择的一些数字(3，5，7)。但是底下的不都比上面的看起来怪吗？对，没错！就应该是这样的！因为该算法正在寻找边缘情况，或者换句话说，奇怪的情况！

![](img/d4cec28ede12ba226347a5c46f40d1d3.png)

*A random sample of handwritten numbers in the MNIST dataset | [Source](https://web.archive.org/web/20221117203617/https://www.sciencedirect.com/science/article/abs/pii/S0925231214008145?via%3Dihub)*

![](img/148fc65547d28f447fff37cad7fc5b1b.png)

*Numbers (3, 5, 7) selected through querying and nearest neighbour search from the same MNIST dataset | [Source](https://web.archive.org/web/20221117203617/https://www.sciencedirect.com/science/article/abs/pii/S0925231214008145?via%3Dihub)*

你可以在关于[最近邻搜索的文章中详细了解这个过程。](https://web.archive.org/web/20221117203617/https://www.sciencedirect.com/science/article/abs/pii/S0925231214008145?via%3Dihub)

### 使用抽样技术的主动学习

使用采样的主动学习可以归结为以下步骤:

1.  使用人类甲骨文标记子样本数据。
2.  在标记的数据上训练一个相对较轻的模型。
3.  该模型用于预测每个剩余的未标记数据点的类别。
4.  基于模型输出，给每个未标记的数据点打分。
5.  基于这些生成的分数选择子样本，并将其发送出去进行标记(子样本的大小可能取决于标记预算/资源和时间的可用性)
6.  基于累积的标记数据集重新训练该模型。

重复步骤 3-6，直到模型达到所需的性能水平。在稍后的阶段，您也可以增加模型的复杂性。这里有几个基于此的场景。

#### 1.基于流的采样

在基于流的选择性采样中，未标记的数据被连续地馈送到主动学习系统，在该系统中，学习者基于预定义的学习策略来决定是否将这些数据发送给人类先知。这种方法适用于模型在生产中，并且数据源/分布随时间变化的场景。

#### 2.基于池的采样

在这种情况下，数据样本是基于信息值得分从未标记的数据池中选择的，并被发送用于手动标记。与基于流的采样不同，通常会仔细检查整个未标记的数据集，以选择最佳实例。

所以现在我们知道，从更广泛的角度来看，主动学习算法在未标记数据集的子样本上归零，但是这种选择是如何发生的呢？我们在下面讨论这样的策略，允许选择与模型学习最相关的数据样本。

## 主动学习:二次抽样策略

### 基于委员会的策略

通过建立几个模型，从模型预测中选择信息丰富的样本。这里提到的这些模型的集合称为委员会。如果委员会中有 n 个不同的模型，那么我们可以对一个数据样本进行 n 次预测。抽样可以基于投票或产生的方差(在回归变量的情况下)，甚至基于这些模型之间的不一致。

有几种常见的方法可以为每个数据样本生成信息或优先级分数——

#### 熵

根据[维基百科](https://web.archive.org/web/20221117203617/https://en.wikipedia.org/wiki/Entropy)，*熵是一个科学概念，也是一个可测量的物理属性，通常与无序、随机或不确定状态相关联。*

因此，想象一个有三个可预测类的监督学习场景。初始模型预测了类别概率，如 **class_1** (0.45)、 **class_2** (0.40)、 **class_3** (0.15)。前两类对应的概率相当接近，相差只有 0.05。这意味着模型不确定它必须分配给数据实例的标签，因此导致两个或更多类别的接近概率。熵通常计算为所有类别的总和，如下所示。

这里‘x’代表每个类别，P(y|x)代表它们各自的预测概率。为未标记的数据点计算熵值，并发送选择的样本用于标记。更多上下文，请阅读机器学习中关于[主动学习的内容。](https://web.archive.org/web/20221117203617/https://towardsdatascience.com/active-learning-in-machine-learning-525e61be16e5)

#### KL-散度

KL-Divergence 代表两个概率分布之间的差异，或者，换句话说，它是一种统计距离:一个概率分布 P 与另一个参考概率分布如何不同的度量。更多理解请参考这个[kull back-lei bler 散度](https://web.archive.org/web/20221117203617/https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)。

下面从可用的研究论文中触及两个突出的基于委员会的策略。

1.  基于熵的 Bagging 查询

在这种方法中，k 个训练集是通过替换从原始数据集创建的。这些抽出的子集被提供给相同数量的模型(这些模型在模型类型/超参数方面可能不同)进行训练。然后，这些模型用于对未标记的数据样本池进行预测。在这种方法中，用于测量不一致的试探法是熵，因此得名 EQB(基于熵的 bagging 查询)。

2.  自适应最大差异

这是通过分割特征空间并向每个模型提供具有不同特征的数据集来实现的。这样，我们可以在具有不同特征的数据集上训练不同的模型。所使用的度量将与之前的策略相同(熵)。

### 基于高利润的策略

它们特别适用于基于保证金的分类器，如 SVM。在 SVM，支持向量将是最能提供信息的点，因此所选择的数据点将是落在该边缘附近的数据点。到分离超平面的距离可以被认为是测量模型对未标记数据样本的置信度或确定性的良好度量。这一类别中有几个策略可以修改，以应用于基于概率的模型。

#### 边缘取样(毫秒)

支持向量是位于距离逻辑决策边界正好 1 的边缘上的标记数据样本。边际抽样策略基于这样的假设，即落在该边际区域内的数据样本与获取标签最相关。

在上面所示的等式中，F(xi，w)表示任何数据样本和类 w 的超平面之间的距离。 *U* 表示未标记的数据集。该策略选择单个数据样本进行查询。

#### 边缘采样-最近支持向量(MS-cSV)

该策略首先存储来自支持向量的每个数据样本的位置。对于每个支持向量，选择与该支持向量距离最小的数据样本。这样，我们可以在每次迭代中有不止一个未标记的数据样本，消除了简单边缘采样的缺点，即每次迭代只选择一个数据样本来查询人类 oracle。

### 基于概率的策略

这是基于类成员概率的估计。与基于边际的策略不同，这适用于任何可以计算与数据实例相关的概率的模型。

#### 基于概率的最小利润策略

这说明了最高类别和第二高类别的类别预测概率的差异。一旦为整个未标记的数据集计算了它们，就可以基于所生成的分数来抽取样本，并将其发送用于标记。

这里‘x’代表每个类别，P(y|x)代表它们各自的预测概率。

根据上述关系，具有最低裕度的实例将首先被发送用于标记，即，它们是前两个可能的类中确定性最小的实例。

#### 最不自信

这种策略允许主动学习者选择模型在预测或类别分配中最没有信心的未标记数据样本。因此，如果模型预测概率最高的类别为 0.5，则 LC 值变为 0.5。

这种关系可以从下面给出的表格中推导出来

#### 预期模型变化

在该策略中，算法选择导致模型中最大变化的数据实例。这个变化，比方说，可以用 SGD 过程中这个数据实例对应的梯度来衡量(随机梯度下降)。

#### 预期误差减少

该算法试图捕获这些数据实例，这将减少后续迭代中的错误。基于最大程度地减少模型训练误差的能力，逐步选择数据样本。

除了上面讨论的几个，使用的一些不太流行的度量是最大归一化对数概率([**【MNLP】**](https://web.archive.org/web/20221117203617/https://arxiv.org/pdf/1707.05928.pdf))和贝叶斯不一致主动学习( [**【秃头】**](https://web.archive.org/web/20221117203617/https://arxiv.org/pdf/1707.05928.pdf) )。

## 真实世界中的主动学习

主动学习在自然语言处理和计算机视觉领域非常流行。具体来说，在自然语言处理的情况下，用于词性标注的信息提取、命名实体识别(NER)等。，需要大量的训练(标注)数据，而为这类用例标注数据的成本确实很高。大多数高级语言模型都基于深度神经网络，并在大型数据集上进行训练。然而，在典型的训练场景下，如果数据集太小，占上风的深度学习通常会减弱。因此，要使深度学习广泛有用，找到上述问题的有效解决方案至关重要。

### NLP 中的主动学习用例(NER)

下面讨论使用主动学习来改进命名实体识别(NER)模型的用例。这篇[论文](https://web.archive.org/web/20221117203617/https://arxiv.org/abs/1707.05928#:~:text=Deep%20learning%20has%20yielded%20state,large%20amounts%20of%20labeled%20data.)深入探讨了 NER 特有的主动学习。他们将上面讨论的策略/评分标准与为每次迭代的训练选择的随机样本进行了比较。用于基准测试的数据集是 [OntoNotes 5.0](https://web.archive.org/web/20221117203617/https://huggingface.co/datasets/conll2012_ontonotesv5) 。

![Use case of Named Entity Recognition (NER) model using active learning](img/7d6ae096f0c4da0c08e87378bab19b02.png)

*Active learning use case in NLP (NER) | [Source](https://web.archive.org/web/20221117203617/https://arxiv.org/pdf/1707.05928.pdf)*

正如我们在上面看到的，很明显，所有的主动学习策略都远远超过了随机抽样(RAND)的基线表现。

显示基于不同主动学习策略的性能改进与迭代次数的关系的另一种表示如下所示。同样与使用随机抽样技术获得的训练数据进行比较。

![Active learning strategies vs the number of iterations](img/91337fc74cccf611239d138e60c4e20b.png)

*The performance improvement based on different active learning strategies vs the number of iterations | [Source](https://web.archive.org/web/20221117203617/https://arxiv.org/pdf/1707.05928.pdf)*

### 计算机视觉中的主动学习用例(自动驾驶)

自动驾驶可能是目前使用主动学习的最有前途和最有价值的用例，并已证明其巨大的商业价值。世界各地的研究人员都在致力于提高自动驾驶系统近乎完美的性能预期所要求的预测准确性。

为了达到这种高精度或性能预期，以视觉为重点的深度学习模型需要大量的训练数据。但是，选择“正确的”训练数据来捕获所有可能的条件/场景和边缘情况，并且以适当的代表性权重来捕获这些数据是一个巨大的挑战。

下图是一个典型的边缘案例，它可能会在自动驾驶系统中造成混乱并导致潜在的灾难性事件。

![Potential catastrophic event in Autonomous driving systems](img/6fa0138cae57d75ea13fc34e71696892.png)

*A case of potential catastrophic event in Autonomous driving systems | [Source](https://web.archive.org/web/20221117203617/https://www.youtube.com/watch?v=g2R2T631x7k&t=0s)*

那么，我们如何检测这些标签的情况呢？让我们来看看主动学习是如何为一些领先的科技公司解决这些挑战的。

来自[英伟达](https://web.archive.org/web/20221117203617/https://www.nvidia.com/en-in/)的文章[‘自动驾驶的可扩展主动学习’](https://web.archive.org/web/20221117203617/https://medium.com/nvidia-ai/scalable-active-learning-for-autonomous-driving-a-practical-implementation-and-a-b-test-4d315ed04b5f)中的一个观察要点在此呈现。为了优化数据选择，需要考虑以下因素

1.  **规模**:如上所述，一个简单的粗略计算表明，每天行驶 8 小时的 100 辆汽车的车队将需要超过 100 万个贴标机以 30fps 的速度标注所有摄像机的所有帧，以进行物体检测。这是非常不切实际的。
2.  **成本**:如前所述，如果我们不小心标记什么，视觉数据集中的标记成本会非常高！
3.  **性能**:选择合适的帧，例如，我们通常不会遇到的罕见场景。

在上述文章中，他们试图从一组未标记的 200 万帧中选择最佳训练样本，这些帧来自于从道路上的车辆收集的记录。该方法从基于池的采样和基于模型集合之间的**不一致**的查询功能/获取功能开始。让我们假设模型 1 &模型 2 预测具有最高概率的类的数据实例(X1)的类概率为 0.78 & 0.91，另一个数据实例(X2)的类概率为 0.76 & 0.82。这里的类概率分歧是(X1(0.13) & X2(0.06)。显然，X1 的阶级分歧高于 X2，因此 X1 将是主动学习的更优选候选人。获取功能被应用于未标记的帧，以在池中选择最有信息量的帧。

下面的操作是在一个循环中执行的，这几乎类似于经典的主动学习算法，但查询/采样的方式略有不同，这最适合这个特定的用例。

1.  根据当前标记的数据，使用随机参数训练“n”个模型。
2.  查询在训练模型之间显示最大差异的样本。
3.  将选择的数据发送给人类先知进行注释。
4.  将新标记的示例追加到训练数据。
5.  重复步骤 1-4，直到合奏达到所需的性能水平。

选定框架的示例热图如下所示。热图显示了这些选定帧中模糊程度较高的区域。这些就是我们要捕捉的数据点，让模型高效学习！因此，当由人类注释者标记时，这些样本可以作为主动学习的合适候选。

![Active learning sample](img/64873e9ab5287d2fe43a3a355640f3f5.png)

*Sample of selected frames via active learning | [Source](https://web.archive.org/web/20221117203617/https://medium.com/nvidia-ai/scalable-active-learning-for-autonomous-driving-a-practical-implementation-and-a-b-test-4d315ed04b5f)*

除了成本优势之外，使用主动学习还观察到平均精度的显著提高(从异议检测的角度来看)。

![Test data from both manual curation and active learning](img/d0fdb1ea296091bae3b7532170ae9e03.png)

*Mean average precision weighted across several object sizes (wMAP) as well as MAP for large and medium object sizes on test data from both manual curation and active learning | [Source](https://web.archive.org/web/20221117203617/https://medium.com/nvidia-ai/scalable-active-learning-for-autonomous-driving-a-practical-implementation-and-a-b-test-4d315ed04b5f)*

### 医学领域的主动学习用例

[用于优化遗传和代谢网络的多功能主动学习工作流程](https://web.archive.org/web/20221117203617/https://www.nature.com/articles/s41467-022-31245-z.pdf)是医学领域主动学习的经典范例，即使用最少的数据集最大化生物目标函数(取决于多个因素的输出/目标),如下所示。下图显示了基于主动学习的采样策略带来的蛋白质产量的提高。

![ Protein production](img/14941a69280aae2a7158bb676d68234b.png)

*The improvement of protein production | [Source](https://web.archive.org/web/20221117203617/https://www.nature.com/articles/s41467-022-31245-z.pdf)*

## 一些用于主动学习的流行框架

### 1 . modal:python 3 的模块化主动学习框架

[*模态*](https://web.archive.org/web/20221117203617/https://modal-python.readthedocs.io/en/latest/) 是 Python3 的一个主动学习框架，设计时考虑了模块化、灵活性和可扩展性。它建立在 scikit-learn 的基础上，允许您以近乎完全的自由度快速创建主动学习工作流。

modAL 支持前面讨论的许多主动学习策略，比如基于概率/不确定性的算法、基于委员会的算法、减少错误等等。

使用 scikit-learn 分类器(例如 RandomForestClassifier)进行主动学习可以像下面这样简单。

```py
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_training, y_training=y_training
)

query_idx, query_inst = learner.query(X_pool)

learner.teach(X_pool[query_idx], y_new)   code source

```

### 2.libact:Python 中基于池的主动学习

[*libact*](https://web.archive.org/web/20221117203617/https://libact.readthedocs.io/en/latest/) 是一个 python 包，旨在让现实世界的用户更容易进行主动学习。该软件包不仅实现了几种流行的主动学习策略，还通过学习元策略实现了主动学习，使机器能够在运行中自动学习最佳策略。下面是 libact 的一个用法示例:

```py
dataset = Dataset(X, y)
query_strategy = QueryStrategy(dataset) 
labler = Labeler() 
model = Model() 

for _ in range(quota): 
    query_id = query_strategy.make_query() 
    lbl = labeler.label(dataset.data[query_id][0]) 
    dataset.update(query_id, lbl) 
    model.train(dataset) 

```

### 3.羊驼

[*AlpacaTag*](https://web.archive.org/web/20221117203617/https://github.com/INK-USC/AlpacaTag) 是一个基于主动学习的人群标注框架，用于序列标注，如命名实体识别(NER)。如其[文档](https://web.archive.org/web/20221117203617/https://github.com/INK-USC/AlpacaTag)中所述，羊驼的独特优势如下。

*   主动、智能推荐:动态建议注释，并对最有信息量的未标记实例进行采样。
*   自动群体合并:通过合并来自多个标注器的不一致标签来增强实时标注器间的一致性。
*   实时模型部署:用户可以在创建新注释的同时，在下游系统中部署他们的模型。

AlpacaTag 对 NER 用例的注释 UI 如下所示。

![Use cases by AlpacaTag](img/5abae2c98d4d3e03ab004cec6f3cf8f6.png)

*Annotation UI for NER use cases by AlpacaTag | [Source](https://web.archive.org/web/20221117203617/https://github.com/INK-USC/AlpacaTag/wiki/Annotation-Tutorial)*

## 生产中的主动学习

到目前为止，我们试图理解主动学习的概念，它的策略，以及它的一些关键应用。现在让我们深入了解它们是如何在生产系统中实际实现的。

主动学习管道主要属于**自动**和**半自动**类别。下面以生产就绪工作流为例对它们进行了简要介绍。

### 主动学习管道–半自动

在半自动或半主动学习方法中，每个周期自动运行，但需要手动触发。需要人工干预的关键领域是

1.选择用于注释的数据–我们需要以明智的方式选择下一张图像。

2.从每个周期中创建的模型集合中选择最佳模型

需要密切监控半自动流水线的性能指标，例如每个主动学习周期之后的模型性能度量。这种技术本质上容易出错，尤其是当它需要多个学习周期时。

#### 用[自动训练](https://web.archive.org/web/20221117203617/https://huggingface.co/autotrain)和[神童](https://web.archive.org/web/20221117203617/https://prodi.gy/)进行半自动或半主动学习的例子

本节将简要介绍我们如何使用 AutoTrain 和 Prodigy 构建一个主动学习管道。

但在此之前，让我们快速了解一下什么是 AutoTrain & Prodigy！

顾名思义，AutoNLP，现在命名为 AutoTrain，是一个由 Hugging Face 创建的框架，以非常少的代码在可用的数据集上构建我们自己的深度学习模型。AutoNLP 建立在最新的 NLP 模型上，如 Transformers、NLP inference-API 和其他工具。您可以轻松上传您的数据和相应的标签，以启动培训管道。有了 Auto Train，我们可以使用最好的可用模型，它可以根据用例自动进行微调，并服务于最终用户。因此生产准备就绪！

Prodigy 是一款由[爆炸](https://web.archive.org/web/20221117203617/https://explosion.ai/)的商业注释工具。主要是它是一个基于网络的工具，允许你实时注释数据。它支持两种 NLP &计算机视觉任务。但是您可以使用任何其他最适合您的用例及成本约束的开源或商业工具！

运行管道涉及的步骤有

1.  在 AutoTrain 中创建新项目。

![A new Project in AutoTrain](img/2f2f1ebf1ffc272778bde06a8eb8abd5.png)

*Creating a new Project in AutoTrain | Source:*

2.  对于标注部分，Prodigy 为 NLP 任务提供了一个交互界面。您可以将它安装在您的本地机器或云服务器上，一旦 prodigy 安装完毕，您可以按以下格式运行它

*$ prodigy ner . manual labelled _ dataset blank:en dataset . CSV–label label 1，label2，label3*

让我们看看这里的论点，

*   **blank: en** 是分词器。
*   **Dataset.csv** 是您的自定义数据集。
*   **标签 1，标签 2** …..是将用于标记的标签列表。

一旦运行了这个命令，您就可以转到 prodigy 提供的 web UI，它通常运行在一个本地主机端口上，然后开始标记过程。用 Prodigy 为 NER 用例标记数据的过程如下所示。更多详情请参考[篇](https://web.archive.org/web/20221117203617/https://huggingface.co/blog/autonlp-prodigy)。

![ Labeling data for a NER](img/276d71dee2b2ba2b0eab466725087cf7.png)

*The process of labeling data for a NER | [Source](https://web.archive.org/web/20221117203617/https://huggingface.co/blog/autonlp-prodigy)*

下图显示了 Prodigy web 用户界面用于普通文本分类标签。

![Vanilla text classification labeling](img/70d0df1590aa99cb4d64d84bd51ba925.png)

*The Prodigy web UI used for vanilla text classification labeling | [Source](https://web.archive.org/web/20221117203617/https://huggingface.co/blog/autonlp-prodigy)*

3.  创建的标记数据集被转换为 AutoTrain 可读格式，以便上传到 AutoTrain 项目中。请参考这篇[文章](https://web.archive.org/web/20221117203617/https://huggingface.co/blog/autonlp-prodigy)了解更多信息。

4.  运行 NER 的自动列车管道，并可视化所创建的模型集合的准确性。

![The accuracy of the ensemble of models ](img/3379de166568c943855e6ef002737ff5.png)

*Visualization of the accuracy of the ensemble of models created | [Source](https://web.archive.org/web/20221117203617/https://huggingface.co/blog/autonlp-prodigy)*

5.  看看 NER 模型的准确性，如果您对模型的性能不满意，请使用更多的标注数据从第 2 步开始重复该过程。

上述过程是半主动学习，因为没有算法明确选择哪些应该是最好的实体来标记，因为在现实世界的 NER 用例中，这是不可行的。因此，半主动学习在 NER 领域是一个良好的开端！

### 主动学习管道——自动化

因此，在自动流水线中，我们将两个手动干预步骤留给智能算法。可以通过上一节中讨论的任何一种采样技术来选择用于注释的数据，即查询、基于池的采样等。选择最佳模型也可以基于我们选择的任何性能指标或它们的加权值。以下是一些带有 AWS 的 AL 管线的示例。

1.  [AWS 自定义分类模型的主动学习工作流](https://web.archive.org/web/20221117203617/https://aws.amazon.com/)

下面显示的是使用实时分类 API、反馈循环和人工审查工作流的参考架构。这里使用的主要服务是[亚马逊理解](https://web.archive.org/web/20221117203617/https://aws.amazon.com/comprehend/)。

![A reference architecture using Real-time classification](img/58825dc05beacc082bc57e048e01c7c1.png)

*A reference architecture using Real-time classification APIs, feedback loops, and human review workflows | [Source](https://web.archive.org/web/20221117203617/https://aws.amazon.com/comprehend/)*

假设我们有一个基于文本的大型数据集，我们需要通过利用主动学习为其建立一个分类器。在 AWS 提供的众多服务中，Amazon understand 是专门为文本分析而构建的，无论是分类、揭示见解、主题建模还是情感分析。下面简要介绍管道中涉及的一些服务。

*   这个过程从调用一个带有需要分类的文本的 [Amazon API gateway](https://web.archive.org/web/20221117203617/https://aws.amazon.com/api-gateway) 端点开始。API 网关端点然后调用 AWS Lambda 函数，该函数又调用[亚马逊理解](https://web.archive.org/web/20221117203617/https://aws.amazon.com/comprehend/)端点来返回标签&置信度得分。

*   置信度较低的实例会被发送给一个人进行审查，在上图中被命名为隐式反馈。也可能有自定义反馈，它被捕获为显式反馈。

*   然后，人工注释的数据集被馈送到再训练工作流中，并且再训练的模型被测试其性能，并且重复这些步骤，直到达到期望的 KPI 值。

详细的实现管道，可以参考 [AWS 博客。](https://web.archive.org/web/20221117203617/https://aws.amazon.com/blogs/machine-learning/active-learning-workflow-for-amazon-comprehend-custom-classification-part-1/)

2.  [使用亚马逊 SageMaker Ground Truth 的主动学习工作流程](https://web.archive.org/web/20221117203617/https://sagemaker-examples.readthedocs.io/en/latest/ground_truth_labeling_jobs/bring_your_own_model_for_sagemaker_labeling_workflows_with_active_learning/bring_your_own_model_for_sagemaker_labeling_workflows_with_active_learning.html)

这来自 AWS SageMaker 范例库。目前，它采用笔记本的形式，为文本分类标注作业的自动化标注工作流创建所需的资源。请通过以上链接详细了解工作流程。

## 结束了！

机器学习环境中的主动学习就是在训练迭代过程中，在人工标记器的帮助下，动态地、增量地标记数据，以便算法可以从未标记的池中检测出哪些数据对它来说是最有信息价值的。

主动学习的应用使得标注的成本大大降低，并且提高了某些基于自然语言处理和计算机视觉的模型的性能。许多兴趣和研究正在主动学习领域发生。但事实上，只有少数现成的主动学习渠道可供使用。即使利用了它们，也需要进行大量的定制来使它们适应您的特定用例。

因此，根据您的专业知识和用例，您可以从上面讨论的概念和应用程序中获得灵感，自己构建一个，或者利用商业工具。

祝你在积极的学习旅途中好运！

### 参考