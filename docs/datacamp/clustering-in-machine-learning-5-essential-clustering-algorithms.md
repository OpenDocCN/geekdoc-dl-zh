# 机器学习中的聚类:5 种基本的聚类算法

> 原文：<https://web.archive.org/web/20221129052847/https://www.datacamp.com/blog/clustering-in-machine-learning-5-essential-clustering-algorithms>

## 介绍

聚类是一种无监督的机器学习技术，在模式识别、图像分析、客户分析、市场细分、社交网络分析等领域有很多应用。从航空公司到医疗保健等等，很多行业都在使用集群。

这是一种无监督学习，意味着我们不需要聚类算法的标记数据；这是聚类相对于分类等其他监督学习的最大优势之一。在本集群教程中，您将学习:

*   什么是集群？
*   集群的业务应用/用例
*   5 种基本的聚类算法

*   k 均值
*   均值漂移
*   DBSCAN
*   分层聚类
*   桦树

*   关于集群的常见问题

## 什么是集群？

聚类是以这样的方式排列一组对象的过程，即同一组(称为聚类)中的对象彼此之间比任何其他组中的对象更相似。数据专业人员通常在探索性数据分析阶段使用聚类来发现数据中的新信息和模式。由于聚类是无监督的机器学习，它不需要标记数据集。

聚类本身不是一个特定的算法，而是要解决的一般任务。您可以使用各种算法来实现这一目标，这些算法在理解构成聚类的内容以及如何有效地找到它们方面有很大的不同。

在本教程的后面，我们将比较不同聚类算法的输出，然后详细讨论当今工业中使用的 5 种基本和流行的聚类算法。虽然算法本质上是数学，但本聚类教程旨在建立对算法的直观理解，而不是数学公式。

### 聚类分析的关键成功标准

与分类或回归等监督学习用例不同，聚类不能完全实现端到端自动化。相反，这是一个信息发现的迭代过程，需要经常使用领域专业知识和人类判断来对数据和模型参数进行调整，以实现期望的结果。

最重要的是，因为聚类是无监督学习，不使用标记数据，我们无法计算精度、AUC、RMSE 等性能指标。，比较不同的算法或数据预处理技术。因此，这使得评估聚类模型的性能变得非常具有挑战性和主观性。

聚类模型中的关键成功标准包括:

*   它是可解释的吗？
*   聚类的结果对商业有用吗？
*   你是否在数据中学习到了新的信息或发现了新的模式，而这些是你在聚类之前没有意识到的？

### 构建聚类背后的直觉

在深入研究算法细节之前，让我们用一个水果数据集的玩具例子来建立一个聚类背后的直觉。假设我们有一个巨大图像数据集集合，包含三种水果(I)草莓，(ii)梨，和(iii)苹果。

在数据集中，所有图像都混在一起，您的用例是将相似的水果分组在一起，即创建三个组，每个组包含一种水果。这正是聚类算法要做的事情。

![clustering algorithm](img/d41c03d34415e8521fe174f8ffecec46.png)

图片来源:[https://static . Java point . com/tutorial/machine-learniimg/clustering-in-machine-learning . png](https://web.archive.org/web/20221116200042/https://static.javatpoint.com/tutorial/machine-learniimg/clustering-in-machine-learning.png)

## 集群的商业应用

聚类是一种非常强大的技术，在从媒体到医疗保健、从制造到服务行业以及任何有大量数据的地方都有广泛的应用。让我们来看看一些实际的使用案例:

### 客户细分

根据客户的购买行为或兴趣，使用聚类算法对客户进行分类，以开展有针对性的营销活动。

想象一下，你有 1000 万客户，你想开展定制或有针对性的营销活动。你不太可能开发 1000 万的营销活动，那我们怎么办？我们可以使用聚类将 1000 万客户分成 25 个群，然后设计 25 个营销活动，而不是 1000 万。

![Customer Segmentation](img/a4cbe252180cba81f9e670cadd82fb67.png)

影像来源:https://miro . medium . com/max/845/1 * rfatwk 6 twbrdj 1 或 1rz 8w . png

### 零售集群

零售企业集群有许多机会。例如，您可以收集每家商店的数据，并在商店级别进行聚类，以根据客流量、平均商店销售额、SKU 数量等属性，得出哪些位置彼此相似的见解。

另一个例子是类别级别的聚类。在下图中，我们有八家商店。不同的颜色代表不同的集群。本例中有四个集群。

请注意，商店 1 中的除臭剂类别由红色聚类表示，而商店 2 中的除臭剂类别由蓝色聚类表示。这表明商店 1 和商店 2 的除臭剂品类目标市场完全不同。

![Retail cluster](img/cc822b32595f00ee91740943e84b109d.png)

图片来源:[https://www . dota ctiv . com/hs-fs/hub fs/Category-based % 20 clustering . png？width = 1038&height = 557&name = Category-based % 20 clustering . png](https://web.archive.org/web/20221116200042/https://www.dotactiv.com/hs-fs/hubfs/Category-based%20clustering.png?width=1038&height=557&name=Category-based%20clustering.png)

### 临床护理/疾病管理中的聚类

医疗保健和临床科学又是一个充满机会的领域，在该领域确实非常有影响力。一个这样的例子是 Komaru & Yoshida 等人在 2020 年发表的研究，其中他们收集了 101 名患者的人口统计数据和实验室数据，然后将他们分成 3 组。

每个聚类由不同的条件表示。例如，群组 1 具有低 WBC 和 CRP 的患者。聚类 2 具有高 BMP 和血清的患者，聚类 3 具有低血清的患者。考虑到血液透析后 1 年的死亡率，每个聚类代表不同的生存轨迹。

![Clinical clustering](img/44b75f34d7d1320b5355f3dea1185901.png)

图片来源:[https://els-jbs-prod-cdn . jbs . elsevierhealth . com/CMS/attachment/da 4c B0 c 9-0a 86-4702-8a 78-80 ffffcf 1 f 9 c/fx1 _ lrg . jpg](https://web.archive.org/web/20221116200042/https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/da4cb0c9-0a86-4702-8a78-80ffffcf1f9c/fx1_lrg.jpg)

### 图象分割法

图像分割是将图像分类成不同的组。在使用聚类的图像分割领域已经做了很多研究。如果您想要隔离图像中的对象以单独分析每个对象来检查它是什么，这种类型的聚类非常有用。

在下面的示例中，左侧表示原始图像，右侧是聚类算法的结果。您可以清楚地看到有 4 个集群，它们是图像中基于像素(老虎、草、水和沙子)确定的 4 个不同的对象。
![Image segmentation](img/69c677fdeb9583dc7b4531ee9f96d2a5.png)

| 想要启动你的职业生涯掌握作为机器学习科学家的基本技能吗？看看 DataCamp [机器学习科学家用 Python](https://web.archive.org/web/20221116200042/https://www.datacamp.com/tracks/machine-learning-scientist-with-python) 和[机器学习科学家用 R](https://web.archive.org/web/20221116200042/https://www.datacamp.com/tracks/machine-learning-scientist-with-r) 开设的这些令人惊叹的课程。 |

### 不同聚类算法的比较

在 Python 中流行的机器学习库 [scikit-learn](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/supervised-learning-with-scikit-learn) 中实现了 10 种无监督聚类算法。每种算法在数据集中确定和分配聚类的方式都有根本的差异。

这些算法在数学形式上的潜在差异可以归结为四个方面，我们可以在这四个方面比较和对比这些算法:

*   模型所需的参数
*   可量测性
*   用例，
*   几何，即用于计算距离的公制。

让我们来关注一下这些算法的输出。在下图中，每一列代表不同聚类算法的输出，如 KMeans、Affinity Propagation、MeanShift 等。共有 10 种算法在同一个数据集上进行训练。

一些算法已经产生了相同的输出。请注意凝聚聚类、DBSCAN、光学和光谱聚类都产生了相同的聚类。

但是，如果您注意并比较 KMeans 的输出和 MeanShift 算法的输出，您会注意到这两种算法产生了不同的结果。在 KMeans 的情况下，只有两个组(集群:蓝色和橙色)，而在 MeanShift 的情况下，有三个组，即蓝色、绿色和橙色。

![Comparison of different cluster](img/c87ec22cd1db073da1498132eb5e1731.png)

图片来源:[https://sci kit-learn . org/stable/_ images/sphx _ glr _ plot _ cluster _ comparison _ 001 . png](https://web.archive.org/web/20221116200042/https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)

不幸的是(或者幸运的是)，聚类没有对错之分。确定并做出类似“X 算法在这里表现最好”这样的陈述是如此简单

这是不可能的，正因为如此，集群是一项非常具有挑战性的任务。

最终，哪种算法工作得更好并不取决于任何容易测量的指标，而是取决于解释和输出对于手头用例的有用程度。

## 5 种基本的聚类算法

### k 均值

K-Means 聚类算法无疑是最流行和最广泛使用的任务聚类算法。这主要是因为直觉和易于实现。这是一种基于质心的算法，用户必须定义想要创建的簇的数量。

这通常来自业务用例，或者通过尝试不同的集群数量值，然后评估输出。

K-Means 聚类是一种迭代算法，它创建非重叠的聚类，这意味着数据集中的每个实例只能专门属于一个聚类。获得 K-Means 算法直觉的最简单方法是理解下面的示例图中的步骤。你也可以在我们的 Python 中的 K-Means 聚类教程和 R 中的 T2 K-Means 聚类教程中获得这个过程的详细描述。

1.  用户指定簇的数量。
2.  基于簇的数量随机初始化质心。在下图的迭代 1 中，注意三个质心被随机初始化为蓝色、红色和绿色。
3.  计算数据点和每个质心之间的距离，并将每个数据点分配给最近的质心。
4.  基于所有分配的数据点重新计算质心的平均值，这将改变质心的位置，正如你在迭代 2 - 9 中看到的，直到它最终收敛。
5.  迭代继续进行，直到质心的均值没有变化或者达到参数 max_iter，这是用户在训练期间定义的最大迭代次数。在 scikit-learn 中，max_iter 默认设置为 300。

![K-means](img/ce6b3e9cc692defb0af4e91cdaf89833.png)

图片来源:[https://www . learnbymarketing . com/WP-content/uploads/2015/01/method-k-means-steps-example . png](https://web.archive.org/web/20221116200042/https://www.learnbymarketing.com/wp-content/uploads/2015/01/method-k-means-steps-example.png)

### 均值漂移

与 K-Means 算法不同，MeanShift 算法不需要指定聚类数。该算法本身会自动确定聚类的数量，如果您不确定数据中的模式，这相对于 K-Means 是一个相当大的优势。

MeanShift 也基于质心，并迭代地将每个数据点分配给聚类。MeanShift 聚类最常见的用例是图像分割任务。

均值漂移算法基于核密度估计。类似于 K-Means 算法，MeanShift 算法迭代地将每个数据点分配给随机初始化的最近的聚类质心，并且每个点基于最多点的位置在空间中迭代地移动，即模式(在 MeanShift 的上下文中，模式是该区域中数据点的最高密度)。

这就是为什么均值漂移算法也被称为模式搜索算法。均值漂移算法的步骤如下:

*   选取任意一个随机点，并在该随机点周围创建一个窗口。
*   计算该窗口内所有点的平均值。
*   按照模式的方向移动窗口。
*   重复这些步骤，直到收敛。

![Mean Shift](img/6dee1fea2e4c632c650863a84dcf4373.png)

图片来源:[https://www . researchgate . net/publication/326242239/figure/fig 3/AS:【电子邮件保护】/直观描述平均移动过程(Intuitive-description-of-the-mean-shift-procedure-find-the-dense-regions-in-the . png](https://web.archive.org/web/20221116200042/https://www.researchgate.net/publication/326242239/figure/fig3/AS:645578044231681@1530929208053/Intuitive-description-of-the-mean-shift-procedure-find-the-densest-regions-in-the.png)

### DBSCAN

DBSCAN 或**D**en sity-**B**as**S**partial**C**lustering of**A**applications with**N**oise 是一种无监督聚类算法，其工作前提是聚类是由低密度区域分隔的区域中的密集空间。

与 K-Means 和 MeanShift 相比，该算法的最大优势在于它对异常值具有鲁棒性，这意味着它不会在任何聚类中包含异常值数据点。

DBSCAN 算法只需要用户提供两个参数:

*   围绕每个数据点创建的圆的半径，也称为“ε”
*   minPoints 定义了在该圆内将该数据点分类为核心点所需的最小数据点数。

每个数据点都被一个半径为ε的圆包围，DBSCAN 将它们识别为核心点、边界点或噪声点。如果围绕一个数据点的圆具有由 minPoints 参数指定的最小点数，则该数据点被视为核心点。

如果点的数量低于所需的最小值，则认为它是边界点，如果在任何数据点的ε半径内没有额外的数据点，则认为它是噪声。噪声数据点不会被归类到任何聚类中(基本上，它们是异常值)。

DBSCAN 聚类算法的一些常见用例有:

*   它在分离高密度和低密度的集群方面表现出色；
*   它对非线性数据集非常有效；和
*   它可以用于异常检测，因为它分离出噪声点，并且不将它们分配给任何聚类。

比较 DBSCAN 和 K-Means 算法，最常见的差异是:

*   K-Means 算法对数据集中的所有实例进行聚类，而 DBSCAN 不会将噪声点(异常值)分配给有效的聚类
*   K-Means 对非全局聚类有困难，而 DBSCAN 可以顺利处理
*   K-Means 算法假设数据集中的所有数据点都来自高斯分布，而 DBSCAN 对数据不做任何假设。

你可以在我们的教程中了解更多关于 Python 中的 [DBSCAN。](https://web.archive.org/web/20221116200042/https://www.datacamp.com/tutorial/dbscan-macroscopic-investigation-python)

![DBSCAN](img/fc5e5da238e7d9e0cf310f4053bb4b8d.png)

图片子:[https://miro . medium . com/proxy/1 * TC 8uf-h0 nquflc 8-0 uinq . gif](https://web.archive.org/web/20221116200042/https://miro.medium.com/proxy/1*tc8UF-h0nQqUfLC8-0uInQ.gif)

### 分层聚类

分层聚类是一种构建聚类层次结构的聚类方法。这种方法有两种。

*   这是一种自下而上的方法，在开始时，每个观察都被视为自己的聚类，当我们从下往上移动时，每个观察都被合并成对，而对又被合并成聚类。
*   **divided**:这是一种“自上而下”的方法:所有的观察从一个集群开始，当我们从上到下移动时，分裂被递归地执行。

在分析来自社交网络的数据时，层次聚类是迄今为止最常见和最流行的聚类方法。图中的节点(分支)根据它们之间的相似程度进行相互比较。通过将彼此相关的较小节点组链接在一起，可以创建较大的分组。

层次聚类最大的优点是易于理解和实现。通常，这种聚类方法的输出在如下图像中进行分析。它被称为树状图。

您可以在我们的 Python 中的[聚类分析](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/cluster-analysis-in-python)课程中了解更多关于层次聚类和 K-Means 聚类的信息。

![Hierarchical clustering](img/a81f04a2703e1398295ad1761b19f479.png)

图片来源:[https://www . research gate . net/profile/Rahmat-Widia-Sembiring/publication/48194320/figure/fig 1/AS:【电子邮件保护】/Example-of-a-dendrogram-from-hierarchical-clustering . png](https://web.archive.org/web/20221116200042/https://www.researchgate.net/profile/Rahmat-Widia-Sembiring/publication/48194320/figure/fig1/AS:307395533262848@1450300214331/Example-of-a-dendrogram-from-hierarchical-clustering.png)

### 桦树

BIRCH 代表基于平衡迭代层次的聚类。它用于 K-Means 不能实际扩展的非常大的数据集。BIRCH 算法将大数据分成小簇，并试图保留尽可能多的信息。然后对较小的组进行聚类以获得最终输出，而不是直接对大型数据集进行聚类。

BIRCH 通常用于通过生成其他聚类算法可以利用的信息摘要来补充其他聚类算法。用户必须定义训练 BIRCH 算法的聚类数，类似于我们在 K-Means 中定义它的方式。

使用 BIRCH 的一个好处是它可以渐进地动态聚类多维数据点。这样做是为了在给定的内存和时间限制下创建最高质量的集群。在大多数情况下，BIRCH 只需要在数据库中进行一次搜索，这使得 BIRCH 具有可伸缩性。

BIRCH 聚类算法最常见的用例是，它是 KMeans 的内存高效替代方案，可用于对由于内存或计算限制而无法通过 KMeans 处理的大型数据集进行聚类。

## 结论

聚类是一种非常有用的机器学习技术，但它不像一些监督学习用例那样简单，如[分类](https://web.archive.org/web/20221116200042/https://datacamp.com/courses/supervised-learning-in-r-classification)和[回归](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/introduction-to-regression-in-r)。这主要是因为性能评估和评估模型的质量很难，而且有一些关键参数，如用户必须正确定义的聚类数，以获得有意义的结果。

然而，在广泛的行业中有大量的集群用例，即使对于数据科学家、机器学习工程师和数据分析师来说，这也是一项重要的技能。

如果您想了解更多关于聚类和无监督机器学习的知识，并学习使用 Python 和 R 语言实现，下面的课程可以帮助您取得进步:

*   [https://www.datacamp.com/courses/cluster-analysis-in-python](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/cluster-analysis-in-python)
*   [https://www.datacamp.com/courses/cluster-analysis-in-r](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/cluster-analysis-in-r)
*   [https://www . data camp . com/courses/unsupervised-learning-in-python](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/unsupervised-learning-in-python)
*   [https://www.datacamp.com/courses/unsupervised-learning-in-r](https://web.archive.org/web/20221116200042/https://www.datacamp.com/courses/unsupervised-learning-in-r)

聚类是一种无监督的机器学习技术。它不需要用于训练的标记数据。

不，聚类算法不需要标记数据。如果您已经标记了数据，则需要监督分类算法。

是的，就像有监督的机器学习一样，如果你的数据中有分类特征，你必须用 one-hot-encoding 这样的技术对它们进行编码。一些算法，如 K-Modes，被设计成直接接受分类数据而不进行任何编码。

没错，聚类就是机器学习。具体来说，就是无监督的机器学习。

聚类可用于描述性和预测性分析。它更常用于探索性数据分析，即描述性分析。

没有一种确定的方法来衡量聚类算法的性能，就像在有监督的机器学习中一样(AUC、准确度、R2 等)。).模型的质量取决于对输出和用例的解释。但是，有一些工作区指标，如同质性得分、轮廓得分等。

是的，聚类算法根据数据集中的组来分配标签。最终，它是数据集中的一个新的分类列。因此，聚类通常用于监督学习任务中的特征工程。