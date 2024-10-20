# 如何合乎道德地使用机器学习来推动决策

> 原文：<https://web.archive.org/web/20221129040116/https://www.datacamp.com/blog/how-to-ethically-use-machine-learning-to-drive-decisions>

## 关注坚实的数据基础和工具

拥有高质量的数据本身就是一个巨大的挑战。我们建议希望利用机器学习、人工智能和数据科学的公司考虑 Monica Rogati 的 *[人工智能需求层次](https://web.archive.org/web/20220522132745/https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007)* ，它将机器学习作为拼图的最后一块。

[![](img/cb1f29883717bd4d8bce3652cd8a26de.png)](https://web.archive.org/web/20220522132745/https://www.datacamp.com/resources/ebooks/definitive-guide-to-machine-learning-for-business-leaders)

来源:[黑客月](https://web.archive.org/web/20220522132745/https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007)

这种层次结构表明，在机器学习发生之前，您需要坚实的数据基础和用于提取、加载和转换数据(ETL)的工具，以及用于清理和聚合来自不同来源的数据的工具。

这需要强大的[数据工程](https://web.archive.org/web/20220522132745/https://www.datacamp.com/community/blog/the-path-to-becoming-a-data-engineer)实践——你需要利用数据库，了解如何正确处理数据，安排工作流程，并利用云计算。

所以在你雇佣你的第一个机器学习工程师之前，你应该首先设置你的数据工程、数据科学和数据分析功能。

## 当心你的数据和算法中的偏见

机器学习只能和你给它的数据一样好。如果你的数据有偏差，你的模型也会有偏差。例如，[亚马逊开发了一个 ML 招聘工具](https://web.archive.org/web/20220522132745/https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G)来预测申请人的成功，该工具基于具有十年培训数据的简历，这些数据有利于男性，因为整个科技行业历史上男性占主导地位，这导致 ML 工具也对女性有偏见。

这就是为什么数据伦理近年来成为如此重要的话题。随着越来越多的数据生成，如何使用这些数据的影响也急剧扩大。这需要原则性的考虑和监测。正如谷歌的首席决策科学家 Cassie Kozyrkov 所类比的那样，一个老师的好坏取决于他们用来教学生的书。如果书有偏见，他们的教训也会有偏见。

密切关注你的模型并改进它

请记住，当您的模型投入生产、进行预测或执行分类时，机器学习的工作并没有结束。已经部署并正在工作的模型仍然需要被监控和维护。

如果您有一个基于交易数据预测信用卡欺诈的模型，那么每次您的模型做出预测并根据预测采取行动时，您都会获得有用的信息。除此之外，您试图监控和预测的活动(在本例中是信用卡欺诈)可能是动态的，会随着时间的推移而变化。在这个过程中，生成的数据不断变化，这被称为*数据漂移*——这证明了定期更新模型是多么重要。

[![](img/e9ed8a354e5c9cb897e10052e93ef45f.png)](https://web.archive.org/web/20220522132745/https://www.datacamp.com/resources/ebooks/definitive-guide-to-machine-learning-for-business-leaders)

来源:[数据块](https://web.archive.org/web/20220522132745/https://databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html)

[![](img/deff1ed8bd9241f33cf69e355a5e51be.png)](https://web.archive.org/web/20220522132745/https://www.datacamp.com/resources/ebooks/definitive-guide-to-machine-learning-for-business-leaders)