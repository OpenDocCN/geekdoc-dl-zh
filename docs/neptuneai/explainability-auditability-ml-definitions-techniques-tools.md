# ML 中的可解释性和可审计性:定义、技术和工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/explainability-auditability-ml-definitions-techniques-tools>

想象一下，你必须向 SaaS 产品的技术负责人展示你新开发的面部识别功能。演示进行得相对顺利，直到首席技术官问你“那么里面到底发生了什么？”而你只能说“没人知道，是黑匣子”。

很快，其他利益相关者就会开始担心。"如果我们不知道某样东西的作用，我们怎么能相信它呢？"。

这是一个合理的担忧。很长一段时间，ML 模型被普遍视为黑箱，因为我们无法解释输入和输出之间的数据发生了什么。但是现在，我们有了解释能力。

在这篇文章中，我们将解释可解释性，探索它的必要性，并讨论简化可解释性的技术和工具。

![Explainability Black Box](img/f43a706a551f9e9d3d200bc39c828afb.png)

*ML Black Box | Source: Author*

## 在 ML 中什么是可解释的，什么是可解释的 AI (XAI)？

机器学习中的可解释性意味着你可以从输入到输出解释你的模型中发生的事情。它使模型透明，解决了黑箱问题。

可解释的人工智能(XAI)是描述这一点的更正式的方式，适用于所有人工智能。XAI 意味着帮助人类专家理解人工智能开发的解决方案的方法。

“可解释性”和“可解释性”经常互换使用。虽然他们目标一致(‘理解模型。

Christoph Molnar 在他的著作《[可解释的机器学习](https://web.archive.org/web/20221218081727/https://christophm.github.io/interpretable-ml-book/)》中，将可解释性定义为人类能够理解决策原因的程度，或者人类能够持续预测 ML 模型结果的程度。

举个例子:你正在构建一个预测时尚行业价格趋势的模型。这个模型可能是可解释的——你可以看到你在做什么。但这还无法解释。一旦你深入挖掘生成结果背后的数据和特征，这就可以解释了。理解哪些特征有助于模型的预测，以及为什么它们会这样，这就是可解释性的全部内容。

一辆汽车需要燃料来驱动，也就是说，是燃料让引擎驱动的——可解释性。理解发动机如何和为什么消耗和使用燃料——可解释性。

本文中提到的大多数工具和技术都可以用于可解释性和可解释性，因为正如我前面提到的，这两个概念都提供了理解模型的视角。

可解释的人工智能是关于更好地理解 ML 模型。他们如何做决定，为什么。模型可解释性的三个最重要的方面是:

1.  透明度
2.  质疑的能力
3.  易于理解

### 可解释性的方法

你可以用两种方式来解释:

1.  **全局—**这是对模型行为的整体解释。它向我们展示了模型的整体视图，以及数据中的要素如何共同影响结果。
2.  **Locally—**这分别告诉我们数据中的每个实例和特征(有点像解释在模型中的某些点上看到的观察结果)，以及各个特征如何影响结果。

### 为什么可解释性很重要？

当机器学习对商业利润产生负面影响时，它会获得坏名声。由于数据科学团队和业务团队之间的脱节，这种情况经常发生。

XAI 将数据科学团队和非技术高管联系起来，改善知识交流，并让所有利益相关者更好地了解产品要求和限制。所有这些都促进了更好的治理。

但是至少还有五个原因说明 ML 可解释性的重要性:

**1。责任:**当一个模型做出错误或流氓决策时，了解导致该决策的因素，或者谁应对该失败负责，对于避免将来出现类似问题是必要的。有了 XAI，数据科学团队可以让组织对他们的人工智能工具有更多的控制权。

**2。信任**:在高风险领域(如医疗保健或金融)，信任至关重要。在 ML 解决方案可以被使用和信任之前，所有的利益相关者必须完全理解这个模型的作用。如果你声称你的模型做出了更好的决策，并注意到了人类看不到的模式，你需要能够用证据来支持它。领域专家自然会对任何声称比他们看得更多的技术持怀疑态度。

**3。合规性**:模型可解释性对于数据科学家、审计员和业务决策者等来说至关重要，以确保符合公司政策、行业标准和政府法规。根据欧洲数据保护法(GDPR)第 14 条，当公司使用自动化决策工具时，它必须提供有关所涉及的逻辑的有意义的信息，以及此类处理对数据主体的重要性和预期后果。世界各地都在实施类似的规定。

**4。性能**:可解释性也可以提高性能。如果你理解了你的模型为什么和如何工作，你就知道该微调和优化什么。

**5。增强的控制**:理解你的模型的决策过程向你展示未知的漏洞和缺陷。有了这些见解，控制就容易了。在低风险情况下快速识别和纠正错误的能力越来越强，特别是当应用于生产中的所有模型时。

### 可解释的模型

ML 中的一些模型具有可解释的特性，即透明性、易理解性和质疑能力。让我们来看看其中的几个。

**1。线性模型:**线性模型如线性回归、带线性核的支持向量机等遵循两个或两个以上变量可以相加使得它们的和也是解的线性原理。例如 y = mx + c。

因此其中一个特性的变化会影响输出。这很好理解和解释。

**2。决策树算法:**使用决策树的模型是通过学习从先验数据中得到的简单决策规则来训练的。因为它们遵循一套特定的规则，理解结果仅仅依赖于学习和理解导致结果的规则。使用 scikit-learn 中的 plot_tree 函数，您可以看到算法如何获得其输出的可视化。使用虹膜数据集:

```py
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
```

我们得到:

**3。广义可加模型(GAM):**GAM 是这样的模型，其中预测变量和因变量(响应)之间的通常关系被线性和非线性平滑函数取代，以模拟和捕捉数据中的非线性。gam 是具有平滑功能的广义线性模型，由于它们的可加性，每个变量都对输出有贡献。因此，我们可以通过简单地理解预测变量来解释 GAM 的输出。

大多数可解释模型的问题在于，它们大多数时候没有捕捉到一些现实世界问题的复杂性，可能是不充分的。此外，因为模型是简单的或线性的，不能保证可解释性。

神经网络或集成模型等是复杂的模型。

因此，对于复杂的模型，我们使用技术和工具使它们变得可以解释。有两种主要方法:

1.  模型不可知方法
2.  特定模型方法

#### 模型不可知

模型不可知的技术/工具可以用在任何机器学习模型上，不管有多复杂。这些不可知的方法通常通过分析特征输入和输出对来工作。石灰就是一个很好的例子。

#### 特定型号

特定于模型的技术/工具特定于单一类型的模型或一组模型。它们取决于特定模型的性质和功能，例如，树解释器。

## ML 中的可解释性技术

让我们从 PDP 开始，对一些有趣的可解释性技术做一个广泛的概述。

### 部分相关图

在其他要素保持不变的情况下，获得一个或两个要素如何影响模型预测结果的全局可视化表示。PDP 告诉您目标和所选特征之间的关系是线性的还是复杂的。PDP 是模型不可知的。

Scikit 学习检查模块提供了一个名为 plot _ partial _ dependence 的部分相关图函数，可创建单向和双向部分相关图:

```py
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0).fit(X, y)
features = [0, 1, (0, 1)]
plot_partial_dependence(clf, X, features)
plt.gcf()

```

![Explainability Partial Dependence Plots](img/b09abbb30c497b31b1b6042c2a3365b6.png)

*The figure shows two one-way and one two-way partial dependence plots for the California housing dataset | Source:* [*scikit-learn.org*](https://web.archive.org/web/20221218081727/https://scikit-learn.org/stable/modules/partial_dependence.html)

### 个体条件期望图(ICE)

这为您提供了模型中某个特征相对于目标特征的效果的局部可视化表示。与 PDP 不同，ICE 以每个样本一行的方式显示对特征的依赖性的单独预测。它也是模型不可知的。您可以使用 [Python](https://web.archive.org/web/20221218081727/https://github.com/AustinRochford/PyCEbox) 和 [R](https://web.archive.org/web/20221218081727/https://github.com/kapelner/ICEbox) 用 PyCEbox 包创建一个冰图。使用 scikit-learn，您可以在您的模型上实现 ICE 绘图，它也使用 plot _ partial _ dependece 函数，并且您必须设置 kind='individual '。

```py
X, y = make_hasplot_partial_dependence(clf, X, features,
    			kind='individual')
```

*注:查看 [scikit-learn 文档](https://web.archive.org/web/20221218081727/https://scikit-learn.org/stable/modules/partial_dependence.html)了解更多细节*。

### 遗漏一列(LOCO)

这是一种非常简单的方法。它留下一列，重新训练模型，然后计算每个 LOCO 模型与原始模型预测得分的差异。如果分数变化很大，被遗漏的变量一定很重要。根据模型宽度(特征的数量)，这种方法可能很耗时。

PDP、ICE 和 LOCO 都有一些缺点:

*   它们不直接捕捉特征交互，
*   它们可能过于近似，这对于自然语言处理中经常使用的分类数据和一次性编码来说可能是个问题。

### 累积局部效应

ALE 图最初是由 D. Apley(等人)在论文“[可视化黑盒监督学习模型](https://web.archive.org/web/20221218081727/https://arxiv.org/abs/1612.08468)中预测变量的效果”中提出的。它与 PDP 的不同之处在于，它在特征上使用一个小窗口，并在预测之间产生差异而不是平均值。由于它不是基于比较平均值，ALE 的偏差更小，性能更好。ALE 的 python 版本可以通过以下方式安装:

```py
 pip install PyALE
```

给定一个具有某些特征的已处理数据集，ALE 图将如下实现:

```py
X[features]
from PyALE import ALE
ale_eff = ale(
    X=X[features], model=model, feature=["carat"], grid_size=50, include_CI=False
)
```

*注:点击[此处](https://web.archive.org/web/20221218081727/https://pypi.org/project/PyALE/)了解更多 ALE 剧情*。

### 局部可解释的模型不可知解释(LIME)

LIME 由华盛顿大学的研究人员开发，旨在通过捕捉特征交互来了解算法内部发生了什么。LIME 围绕特定预测执行各种多特征扰动，并测量结果。它还处理不规则输入。

当维数很高时，保持这种模型的局部保真度变得越来越困难。LIME 解决了一个更可行的任务——找到一个与原始模型局部近似的模型。

LIME 试图通过一系列实验来复制模型的输出。创造者还引入了 SP-LIME，这是一种选择代表性和非冗余预测的方法，为用户提供了模型的全局视图。

*注:你可以在这里了解更多关于[石灰的知识。](https://web.archive.org/web/20221218081727/https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/)*

### 锚

这是由相同的创作者建造的石灰。锚点方法通过使用易于理解的 IF-THEN 规则来解释模型的单个预测，这些规则称为“锚点”，可以很好地支持(锚点)预测。

为了找到锚点，作者使用强化技术结合图搜索算法来探索数据周围的扰动集及其对预测的影响。这是另一种与模型无关的方法。

在最初的[论文](https://web.archive.org/web/20221218081727/https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)中，作者比较了石灰和锚，并可视化了它们如何处理复杂的二元分类器模型( **+** 或**–**)以得出结果。如下所示，LIME 解释通过学习最接近模型的线性决策边界来工作，具有一些局部权重，而锚点使其覆盖范围适应模型行为并使其边界清晰。

![Explainability Anchors](img/9a9ca0b6c40cadb3ed9bd1d9413e9cf1.png)

*LIME vs. Anchors — A Toy Visualization. Figure from Ribeiro, Singh, and Guestrin (2018) | [Source](https://web.archive.org/web/20221218081727/https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)*

主播还接受了各种机器学习任务的测试，如分类、文本生成、结构化预测。

## 沙普利附加解释(SHAP)

SHAP 使用沙普利值的博弈论概念来优化分配功能的重要性。

Shapley 值 SHAP(Shapley Additive exPlanations)是一个特征值对所有可能组合的平均边际贡献。

联合是用于估计特定特征的 Shapley 值的特征组合。这是一种统一的方法来解释机器学习模型的输出，如线性和逻辑回归、 [NLP](https://web.archive.org/web/20221218081727/https://github.com/slundberg/shap#natural-language-example-transformers) 、[提升树模型](https://web.archive.org/web/20221218081727/https://github.com/slundberg/shap#treeexplainer)和上瘾模型。可以通过 [PyPI](https://web.archive.org/web/20221218081727/https://pypi.org/project/shap) 或 [conda-forge](https://web.archive.org/web/20221218081727/https://anaconda.org/conda-forge/shap) 安装；

```py
pip install shap
```

或者:

```py
conda install -c conda-forge shap
```

![Explainability SHAP](img/009ca7383b23055cece1663cc0f56bad.png)

*This shows how each feature is contributing to the model’s output. | [Source](https://web.archive.org/web/20221218081727/https://github.com/slundberg/shap)*

深度 SHAP 是深度学习的 SHAP 的变体，是一种高速近似算法，它使用背景样本而不是单个参考值，并使用 Shapely 方程来线性化 softmax，max，products 等操作。深度 SHAP 由 Tensorflow、Keras 和 Pytorch 支持。

### 深度学习的重要特性(DeepLIFT)

DeepLIFT 是一种深度学习的可解释方法，它使用反向传播将每个神经元的激活与“参考激活”进行比较，然后根据神经元差异记录并分配那个贡献分数。

本质上，DeepLIFT 只是深入挖掘神经网络的特征选择，并找到对输出形成有主要影响的神经元和权重。DeepLIFT 分别考虑了正面和负面的贡献。它还可以揭示其他方法所忽略的依赖性。分数可以在一次反向传递中有效地计算出来。

DeepLIFT 位于 pypi 上，因此可以使用 pip 进行安装:

```py
pip install deeplift

```

### 逐层相关性传播(LRP)

逐层相关性传播类似于 DeepLIFT，它使用一组来自输出的[特意设计的传播规则](https://web.archive.org/web/20221218081727/https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10)进行反向传播，识别神经网络中最相关的神经元，直到您返回输入。所以，你得到了所有的神经元(例如，真正对输出有贡献的像素。LRP 在 CNN 上表现很好，它可以用来[解释 LSTMs](https://web.archive.org/web/20221218081727/https://link.springer.com/chapter/10.1007/978-3-030-28954-6_11) 。

看看这个互动演示，看看 LRP 是如何工作的。

![Explainability LRP](img/64d91743e6504b0ff688577b734daa27.png)

*Visual representation of how LRP does backpropagation from output node through the hidden layer neurons to input, identifying the neurons that had an impact on the model’s output. | [Source](https://web.archive.org/web/20221218081727/https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)*

### 对比解释法(CEM)

对比解释是关于一个事件的事实，如果发现是真的，将构成一个具体事件的真实情况。CEM 方法提供了由模型而不是另一个决策或结果做出的决策和结果的对比解释。

CEM 基于论文[《基于缺失的解释:走向带有相关否定的对比解释》](https://web.archive.org/web/20221218081727/https://arxiv.org/abs/1802.07623)。这里的开源代码实现是[这里的](https://web.archive.org/web/20221218081727/https://github.com/IBM/Contrastive-Explanation-Method)。对于分类模型，CEM 根据相关正面(PP)和相关负面(PN)生成基于实例的解释。

PP 寻找对最初预测的结果影响最小但足够大的特征(例如，图像中的重要像素)，而 PN 识别对最初预测的结果影响最小且必然不影响的特征。PN 提供了一个最小集合，使其区别于最接近的不同类。CEM 可以在 TensorFlow 中实现。要了解更多关于 CEM 的信息，请点击这里。

### 重量

2018 年，Amit (et al)做了一篇论文[“用置信剖面改进简单模型”](https://web.archive.org/web/20221218081727/https://arxiv.org/pdf/1807.07506.pdf)。提出了模型可解释性的深度加权方法。ProfWeight 将预先训练的深度神经网络的高测试精度转移到低测试精度的浅层网络。

就像老师向学生传递知识一样，ProfWeight 使用探针(根据网络的难度在样本中进行加权)来传递知识。

ProfWeight 可以总结为四个主要步骤:

1.  在高性能神经网络的中间表示上附加和训练探针，
2.  在原始数据集上训练一个简单模型，
3.  学习作为简单模型和探针的函数的数据集中的例子的权重，
4.  在最终加权数据集上重新训练简单模型。

### 置换特征重要性

置换特征重要性显示了当单个特征被随机打乱时，模型得分(准确度、F1、R2)的降低。它显示了一个特性对于一个特定的模型有多重要。它是一种模型检验技术，显示特征和目标之间的关系，对于非线性和不透明的估计器很有用。

它可以在 sci-kit 学习库中实现。查看[这里](https://web.archive.org/web/20221218081727/https://scikit-learn.org/stable/modules/permutation_importance.html)看看是怎么做的。

我们知道一些用于 ML 可解释性的方法，那么有什么工具可以让我们的工作变得更容易呢？

### 人工智能可解释性 360 (AIX360)

[AI Explainability 360 toolkit](https://web.archive.org/web/20221218081727/http://aix360.mybluemix.net/)是 IBM 的开源库，支持数据集和机器学习模型的可解释性和可解释性。AIX360 包括一组算法，涵盖了解释的不同维度以及代理可解释性度量。它也有关于不同用例中可解释性的教程，比如信用审批。

### 滑冰者

Skater 是一个开源的、与模型无关的统一 Python 框架，具有模型可解释性和可解释性。数据科学家可以将可解释性构建到现实世界用例的机器学习系统中。

Skater 从全局(基于完整数据集的推理)和局部(推理单个预测)两个方面研究可解释性。它支持深度神经网络、树算法和可伸缩贝叶斯。

*注:点击了解更多关于滑手[的信息。](https://web.archive.org/web/20221218081727/https://oracle.github.io/Skate)*

### 像我五岁一样解释(ELI5)

ELI5 是一个 python 包，用于理解和解释 sklearn 回归器和分类器、XGBoost、CatBoost、LightGBM Keras 等分类器的预测。它通过其统一的 API 为这些算法的这些过程提供可视化和调试。ELI5 理解文本处理，可以高亮显示文本数据。它还可以实现诸如 LIME 和排列重要性之类的技术。

ELI5 在 python 2.7 和 3.4+中工作，它需要 scikit-learn 0.18+。然后，您可以使用以下命令安装它:

```py
pip install eli5
```

或者:

```py
Conda install -c conda-forge eli5
```

*注:点击了解更多信息[。](https://web.archive.org/web/20221218081727/https://eli5.readthedocs.io/en/latest/index.html)*

### 解释性语言

[InterpretML](https://web.archive.org/web/20221218081727/https://interpret.ml/) 是微软开发的开源工具包，旨在为数据科学家、审计员和商业领袖提高模型的可解释性。解释器是灵活和可定制的。在撰写本文时，InterpretML 支持石灰、SHAP、线性模型和决策树。它为模型提供了全局和局部的解释。主要特点:

*   了解不同数据子集的模型性能如何变化，并比较多个模型，
*   探索模型误差，
*   分析数据集统计和分布，
*   探索全球和本地的解释，
*   过滤数据以观察全局和局部特征重要性，
*   运行假设分析，查看编辑数据点要素时模型解释会如何变化。

### 激活地图集

激活图谱可视化了神经网络如何相互作用，以及它们如何随着信息和层的深度而成熟。谷歌与 OpenAI 合作推出了激活地图集。

开发这种方法是为了查看卷积视觉网络的内部工作方式，并获得隐藏网络层中的概念的人类可解释的概述。它从单个神经元的特征可视化开始，但后来发展到神经元的联合可视化。

![Activation Atlases - explainability tools](img/5377cee06a8f1e437699bca3e77f3a51.png)

*An activation atlas of the InceptionV1 vision classification network reveals many fully realized features, such as electronics, buildings, food, animal ears, plants, and watery backgrounds. | Source: [openai.com](https://web.archive.org/web/20221218081727/https://openai.com/blog/introducing-activation-atlases/)*

### 不在场证明解释

Alibi 是一个用于模型检查和解释的开源 Python 库。它提供了解释黑盒算法所需的代码。

Alibi 解释有助于:

*   为可解释的 ML 模型定义 restful APIs，
*   模型监控，
*   黑盒 ML 模型解释算法的高质量参考实现，
*   多用途案例(表格、文本和图像数据分类、回归)，
*   实现最新的模型解释，
*   概念漂移算法偏差检测，
*   模型决策的模型置信度得分。

*注:点击了解更多关于 [Alibi 的信息。](https://web.archive.org/web/20221218081727/https://github.com/SeldonIO/alibi)*

### 假设工具(WIT)

WIT 由 TensorFlow 团队开发，是一个交互式、可视化、无代码的界面，用于在 TensorFlow 中可视化数据集和模型，以便更好地理解模型结果。除了 TensorFlow 模型之外，您还可以使用 XGBoost 和 Scikit-Learn 模型的假设分析工具。

一旦部署了模型，就可以在假设分析工具中的数据集上查看其性能。

此外，您可以按要素对数据集进行切片，并跨这些切片比较性能。然后，您可以确定模型表现最好或最差的数据子集。这对 ML 公平性调查非常有帮助。

该工具可通过 Tensorboard 或 collab notebook 访问。查看 [WIT 网站](https://web.archive.org/web/20221218081727/https://pair-code.github.io/what-if-tool)了解更多信息。

### 微软 Azure

我们都知道 Azure，不需要解释它是什么。Azure 的 SDK 包中有可解释性类。

蔚蓝色。解释包含像 SHAP 树解释器，SHAP 深度解释器，SHAP 线性解释器等功能。

将“pip install azureml-interpret”用于一般用途。

### Rulex 可解释人工智能

Rulex 是一家以一阶条件逻辑规则的形式创建预测模型的公司，每个人都可以立即理解和使用这些模型。

Rulex 的核心机器学习算法逻辑学习机(LLM)的工作方式与传统人工智能完全不同。该产品旨在产生预测最佳决策选择的条件逻辑规则，以便流程专业人员可以立即清楚地了解这些规则。Rulex 规则使每个预测完全不言自明。

与决策树和其他产生规则的算法不同，Rulex 规则是无状态和重叠的。

### 探索和解释的模型不可知语言(DALEX)

Dalex 是一套工具，它检查任何给定的模型，简单的或复杂的，并解释模型的行为。Dalex 围绕每个模型创建了一个抽象层次，使得探索和解释更加容易。它使用 Explain()方法(Python)或 Dalex::explain 函数在模型上创建一个包装器。一旦使用解释函数包装了模型，所有的功能都可以从该函数中获得

Dalex 可与 xgboost、TensorFlow、h2o 配合使用。可以通过 Python 和 r 安装。

r:

```py
install.packages("DALEX")
Dalex::explain

```

Python:

```py
pip install dalex -U
import dalex as dx
exp = dx.Explainer(model, X, y)

```

*注:点击了解更多关于 Dalex [的信息。](https://web.archive.org/web/20221218081727/https://dalex.drwhy.ai/)*

## 结论

为了安全、可靠地包含人工智能，需要人类和人工智能的无缝融合。对于允许从业者容易地评估正在使用的决策规则的质量并减少假阳性的技术，也应该考虑人为干预。

让 XAI 成为一种核心能力，成为你人工智能设计和质量保证方法的一部分。它将在未来支付股息，而且是大量的股息。

理解你的模型不仅仅是一个科学问题。这不是好奇的问题。它是关于知道你的模型在哪里失败，如何修复它们，以及如何向关键的项目涉众解释它们，以便每个人都确切地知道你的模型如何产生价值。

**参考文献:**

1.  [https://www . PwC . co . uk/audit-assurance/assets/explable-ai . pdf](https://web.archive.org/web/20221218081727/https://www.pwc.co.uk/audit-assurance/assets/explainable-ai.pdf)
2.  [https://christophm . github . io/interpretable-ml-book/](https://web.archive.org/web/20221218081727/https://christophm.github.io/interpretable-ml-book/)
3.  [https://link . springer . com/chapter/10.1007% 2f 978-3-030-28954-6 _ 10](https://web.archive.org/web/20221218081727/https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10)
4.  [https://arxiv.org/abs/1802.07623](https://web.archive.org/web/20221218081727/https://arxiv.org/abs/1802.07623)
5.  [https://towards data science . com/understanding-model-predictions-with-lime-a 582 fdff 3a 3b](https://web.archive.org/web/20221218081727/https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b)
6.  [https://sci kit-learn . org/stable/auto _ examples/inspection/plot _ partial _ dependency . html](https://web.archive.org/web/20221218081727/https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html)
7.  [https://www . kdnugges . com/2018/12/four-approach-ai-machine-learning . html](https://web.archive.org/web/20221218081727/https://www.kdnuggets.com/2018/12/four-approaches-ai-machine-learning.html)
8.  [https://link . springer . com/chapter/10.1007% 2f 978-3-030-28954-6 _ 10](https://web.archive.org/web/20221218081727/https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10)
9.  [https://sci kit-learn . org/stable/modules/partial _ dependency . html](https://web.archive.org/web/20221218081727/https://scikit-learn.org/stable/modules/partial_dependence.html)
10.  [http://www . fields . utoronto . ca/talks/Boolean-decision-rules-column-generation](https://web.archive.org/web/20221218081727/http://www.fields.utoronto.ca/talks/Boolean-decision-rules-column-generation)
11.  [https://pair-code.github.io/what-if-tool](https://web.archive.org/web/20221218081727/https://pair-code.github.io/what-if-tool)
12.  [http://aix360.mybluemix.net/](https://web.archive.org/web/20221218081727/http://aix360.mybluemix.net/)
13.  [https://github.com/slundberg/shap](https://web.archive.org/web/20221218081727/https://github.com/slundberg/shap)
14.  [https://dalex.drwhy.ai/](https://web.archive.org/web/20221218081727/https://dalex.drwhy.ai/)