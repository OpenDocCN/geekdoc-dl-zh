# 数据去神秘化:不同类型的人工智能偏见

> 原文：<https://web.archive.org/web/20221129041202/https://www.datacamp.com/blog/data-demystified-the-different-types-of-ai-bias>

欢迎来到为期一个月的数据揭秘系列的最后一篇文章。作为[数据扫盲月](https://web.archive.org/web/20220909225543/https://www.datacamp.com/data-literacy-month/)的一部分，本系列阐明了数据世界的关键概念，并试图回答你可能不敢问的问题。如果你想从头开始，请阅读我们系列的第一篇文章:[什么是数据集？](https://web.archive.org/web/20220909225543/https://www.datacamp.com/blog/data-demystified-what-exactly-is-data)

![Data Demystified: The Different Types of AI Bias](img/64ffb9107ddee11d74e57d6438dd40d7.png)

在这篇文章中，我们将继续从[之前的数据揭秘](https://web.archive.org/web/20220909225543/https://www.datacamp.com/blog/data-demystified-avoiding-the-ai-hype-trap)条目的主题，并讨论人工智能的潜在有害影响，它如何使对某些人群的偏见永久化，以及每个人都应该意识到的不同类型的人工智能偏见。

## 人工智能偏见的问题

今天大多数人工智能系统都利用了机器学习。根据[的定义](https://web.archive.org/web/20220909225543/https://www.datacamp.com/blog/data-demystified-the-difference-between-data-science-machine-learning-deep-learning-and-artificial-intelligence)，机器学习应用先进的统计技术从过去的数据中学习模式，并对未来事件做出预测。

机器学习的广泛采用导致它做出有偏见的预测的案例急剧增加。有偏见的人工智能算法一直是人工智能社区的一个严重问题，是用于模型训练的数据的产物。偏见可以以多种形式表现出来——可能是社会的或结构性的，也可能存在于对特定性别、肤色、宗教或国籍的偏见中。

因此，人工智能算法在试图模仿人类判断时，会从训练数据中学习偏差。让我们回顾一些过去的例子，在这些例子中，有偏见的人工智能预测对社会和整个人类产生了负面影响:

### 性别偏见:亚马逊的招聘引擎

亚马逊开发了一个招聘引擎来自动筛选求职者的简历，以便进一步面试。然而，该算法反映了它从过去的数据中学到的偏见，最终只选择了男性候选人的资料。

### 种族偏见:PredPol 算法

PredPol，或预测性警务，建立了一个犯罪活动高发地区的热图，并将少数民族特定的地点确定为热点地区。该算法是在有偏见的输入数据上训练的，这些数据包括从这些地区报告的几起犯罪事件。

### 种族偏见:COMPAS 算法

惩教罪犯管理概况替代制裁(COMPAS)软件用于评估罪犯重复犯罪的可能性。然而，作为 2016 年调查的一部分，[算法出现了偏差](https://web.archive.org/web/20220909225543/https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)。该软件认为黑人罪犯比白人罪犯更有可能再次犯罪。

![machine learning biases](img/0c40afe64f233a50bcb1ef714c61eae7.png)

从上面的例子可以看出，机器学习算法除了从其他数据规律中学习偏差，还从训练数据中学习偏差。除非在源头进行处理，否则偏差会以多种形式出现在 AI/ML 管道中。随着人工智能在组织和社会中变得更加广泛，每个人都应该意识到人工智能系统不同类型的偏见。以下是人工智能中最常见的三种偏见。

## 人工智能偏见的三种常见类型

### 偏见

当训练数据反映现有的偏见、刻板印象和社会假设时，这些偏见会嵌入到学习模型中；这种偏见被称为偏见。例如，当您搜索“医生”时，搜索结果包含许多男性医生的图像。相比之下，对“护士”的类似搜索会得到女护士的图像。这充分说明了[基于性别的社会陈规定型观念](https://web.archive.org/web/20220909225543/https://towardsdatascience.com/gender-bias-word-embeddings-76d9806a0e17)。

### 样本选择偏差

当训练数据不能代表所研究的人群时，就会出现样本选择偏差。一个例子是人工智能系统被训练来检测皮肤癌。如果原始数据集不代表更广泛的人群，则该系统对于数据集中[代表性不足的群体的成员来说表现不佳。](https://web.archive.org/web/20220909225543/https://www.theguardian.com/society/2021/nov/09/ai-skin-cancer-diagnoses-risk-being-less-accurate-for-dark-skin-study#:~:text=AI%20skin%20cancer%20diagnoses%20risk%20being%20less%20accurate%20for%20dark%20skin%20%E2%80%93%20study,-This%20article%20is&text=AI%20systems%20being%20developed%20to,with%20dark%20skin%2C%20research%20suggests.)

### 测量偏差

测量偏差来自数据收集或测量过程中的误差。例如，如果来自用于为图像识别系统提供数据的照相机的图像质量差，这可能导致对特定人群有偏见的结果。另一个例子可以来自于人的判断。例如，医疗诊断算法可以被训练为基于替代指标(如医生出诊而不是实际症状)来预测疾病的可能性。

## 为负责任的人工智能开发数据素养

在整个这个月，我们强调了数据素养对个人和组织的重要性。数据素养允许非技术利益相关者与数据和人工智能专家交谈，并理解人工智能系统的局限性。更重要的是，它促进了主题专家和人工智能专家之间的双向对话，允许对人工智能系统的潜在危害进行深思熟虑的讨论。

为了让自己具备进行这些对话的必要知识，请参加我们的[理解机器学习](https://web.archive.org/web/20220909225543/https://www.datacamp.com/courses/machine-learning-for-everyone)课程，开始您的数据素养之旅。有关更多数据扫盲和数据去神秘化的内容，请查看以下资源:

*   [查看我们为数据扫盲月计划的内容](https://web.archive.org/web/20220909225543/https://www.datacamp.com/data-literacy-month/)
*   [开始学习我们的数据素养基础技能课程](https://web.archive.org/web/20220909225543/https://www.datacamp.com/tracks/data-literacy-fundamentals)
*   [订阅数据框架播客](https://web.archive.org/web/20220909225543/https://www.datacamp.com/podcast)
*   [查看我们即将举办的活动](https://web.archive.org/web/20220909225543/https://www.datacamp.com/webinars)
*   [从头开始，阅读我们解密数据的第一个条目](https://web.archive.org/web/20220909225543/https://www.datacamp.com/blog/data-demystified-what-exactly-is-data)