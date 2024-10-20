# 学习 Caret 的 3 个理由

> 原文：<https://web.archive.org/web/20230101103415/https://www.datacamp.com/blog/3-reasons-to-learn-caret>

机器学习是对从数据中学习并对数据进行预测的算法的研究和应用。从搜索结果到自动驾驶汽车，它已经在我们生活的各个领域表现出来，是数据科学领域最令人兴奋和发展最快的研究领域之一。由 Max Kuhn 维护的`caret`包是 R 社区中用于预测建模和监督学习的首选包。这个广泛使用的包为所有 R 最强大的机器学习设施提供了一致的接口。需要更有说服力的吗？在本帖中，我们探讨了你应该学习`caret`包的 3 个原因。之后，你可以参加 DataCamp 的[机器学习工具箱](https://web.archive.org/web/20220810040859/https://www.datacamp.com/courses/machine-learning-toolbox/)课程，该课程由`caret`软件包的合著者扎卡里·迪恩-迈尔&马克斯·库恩教授！

![Learn caret](img/f82738f351391c365943efcaeec36a2d.png)

### **1。它可以帮你找到一份数据科学的工作**

有没有通读过数据科学的招聘信息，看到过“预测建模”、“分类”、“回归”或“机器学习”这样的词？如果你正在寻找一个数据科学的职位，你将有机会拥有所有这些主题的经验和知识。幸运的是，`caret`套餐已经覆盖了你。`caret`包以 R 被誉为机器学习的“瑞士军刀”；能够以直观、一致的格式执行许多任务。查看 Kaggle 最近发布的数据科学家职位，这些职位都在寻找具有 R 和机器学习知识的候选人:

*   [Amazon.com 大学的数据科学家](https://web.archive.org/web/20220810040859/https://www.kaggle.com/jobs/17343/amazon-data-scientist-seattle-wa)
*   [CVS Health 的数据科学家](https://web.archive.org/web/20220810040859/https://www.kaggle.com/jobs/17332/cvs-health-data-scientist-woonsocket-ri)

### **2。这是最受欢迎的 R 包之一**

这个`caret`包每月直接下载超过 38000 次，是 R 社区中最受欢迎的包之一。随之而来的是巨大的好处，包括大量的文档和有用的教程。您可以安装`Rdocumentation`包来直接在您的 R 控制台中访问有用的文档和社区示例。只需复制并粘贴以下代码:

```py
# Install and load RDocumentation for comprehensive help with R packages and functions
install.packages("RDocumentation")
library("RDocumentation")
```

当然，学习广泛使用的软件包的另一个好处是你的同事也可能在他们的工作中使用`caret`——这意味着你可以更容易地在项目上合作。另外，`caret`也是大量附加机器学习和建模包的依赖包。理解`caret`是如何工作的将使学习更有用的 R 包变得更容易和更流畅。

### **3。很好学，但是很厉害**

如果你是一个初学 R 的用户，`caret`包为执行复杂的任务提供了一个简单的界面。例如，您可以用一种简单、方便的格式训练多种不同类型的模型。您还可以监视各种参数组合并评估性能，以了解它们对您尝试构建的模型的影响。此外，`caret`软件包通过比较特定问题的精确度和性能，帮助您决定最合适的模型。

完成下面的代码挑战，看看用`caret`构建模型和预测值有多简单。我们已经将`mtcars`数据集分成了训练集`train`和测试集`test`。这两个对象在控制台中都可用。您的目标是根据重量预测`test`数据集中每辆车的每加仑英里数。自己看看`caret`包如何只用两行代码就能处理这项任务！

eyJsYW5ndWFnZSI6InIiLCJwcmVfZXhlcmNpc2VfY29kZSI6IiAgICAgICAgIyBMb2FkIGNhcmV0IHBhY2thZ2VcbiAgICAgICAgICBsaWJyYXJ5KGNhcmV0KVxuICAgICAgICAjIHNldCBzZWVkIGZvciByZXByb2R1Y2libGUgcmVzdWx0c1xuICAgICAgICAgIHNldC5zZWVkKDExKVxuICAgICAgICAjIERldGVybWluZSByb3cgdG8gc3BsaXQgb246IHNwbGl0XG4gICAgICAgICAgc3BsaXQgPC0gcm91bmQobnJvdyhtdGNhcnMpICogLjgwKVxuXG4gICAgICAgICMgQ3JlYXRlIHRyYWluXG4gICAgICAgICAgdHJhaW4gPC0gbXRjYXJzWzE6c3BsaXQsIF1cblxuICAgICAgICAjIENyZWF0ZSB0ZXN0XG4gICAgICAgICAgdGVzdCA8LSBtdGNhcnNbKHNwbGl0ICsgMSk6bnJvdyhtdGNhcnMpLCBdIiwic2FtcGxlIjoiIyBGaW5pc2ggdGhlIG1vZGVsIGJ5IHJlcGxhY2luZyB0aGUgYmxhbmsgd2l0aCB0aGUgYHRyYWluYCBvYmplY3Rcbm10Y2Fyc19tb2RlbCA8LSB0cmFpbihtcGcgfiB3dCwgZGF0YSA9IF9fXywgbWV0aG9kID0gXCJsbVwiKVxuXG4jIFByZWRpY3QgdGhlIG1wZyBvZiBlYWNoIGNhciBieSByZXBsYWNpbmcgdGhlIGJsYW5rIHdpdGggdGhlIGB0ZXN0YCBvYmplY3RcbnJlc3VsdHMgPC0gcHJlZGljdChtdGNhcnNfbW9kZWwsIG5ld2RhdGEgPSBfX18pXG4gICAgICAgXG4jIFByaW50IHRoZSBgcmVzdWx0c2Agb2JqZWN0XG5yZXN1bHRzIiwic29sdXRpb24iOiIjIEZpbmlzaCB0aGUgbW9kZWwgYnkgcmVwbGFjaW5nIHRoZSBibGFuayB3aXRoIHRoZSBgdHJhaW5gIG9iamVjdFxubXRjYXJzX21vZGVsIDwtIHRyYWluKG1wZyB+IHd0LCBkYXRhID0gdHJhaW4sIG1ldGhvZCA9IFwibG1cIilcblxuIyBQcmVkaWN0IHRoZSBtcGcgb2YgZWFjaCBjYXIgYnkgcmVwbGFjaW5nIHRoZSBibGFuayB3aXRoIHRoZSBgdGVzdGAgb2JqZWN0XG5yZXN1bHRzIDwtIHByZWRpY3QobXRjYXJzX21vZGVsLCBuZXdkYXRhID0gdGVzdClcbiAgICAgICBcbiMgUHJpbnQgdGhlIGByZXN1bHRzYCBvYmplY3RcbnJlc3VsdHMiLCJzY3QiOiJ0ZXN0X2V4cHJlc3Npb25fb3V0cHV0KFwibXRjYXJzX21vZGVsXCIsIGluY29ycmVjdF9tc2cgPSBcIlRoZXJlJ3Mgc29tZXRoaW5nIHdyb25nIHdpdGggYG10Y2Fyc19tb2RlbGAuIEhhdmUgeW91IHNwZWNpZmllZCB0aGUgcmlnaHQgZm9ybXVsYSB1c2luZyB0aGUgYHRyYWluYCBkYXRhc2V0P1wiKVxuXG50ZXN0X2V4cHJlc3Npb25fb3V0cHV0KFwicmVzdWx0c1wiLCBpbmNvcnJlY3RfbXNnID0gXCJUaGVyZSdzIHNvbWV0aGluZyB3cm9uZyB3aXRoIGByZXN1bHRzYC4gSGF2ZSB5b3Ugc3BlY2lmaWVkIHRoZSByaWdodCBmb3JtdWxhIHVzaW5nIHRoZSBgcHJlZGljdCgpYCBmdW5jdGlvbiBhbmQgdGhlIGB0ZXN0YCBkYXRhc2V0P1wiKVxuXG5zdWNjZXNzX21zZyhcIkNvcnJlY3Q6IFNlZSBob3cgZWFzeSB0aGUgY2FyZXQgcGFja2FnZSBjYW4gYmU/XCIpIn0=

![Learn caret 2](img/f82738f351391c365943efcaeec36a2d.png)

### **想自己学？**

你很幸运！DataCamp 刚刚发布了一个全新的[机器学习工具箱](https://web.archive.org/web/20220810040859/https://www.datacamp.com/courses/machine-learning-toolbox/)课程。该课程由一揽子计划的合著者 Max Kuhn 和 Zachary Deane-Mayer 教授。您将通过 24 个视频和 88 个互动练习，直接向编写软件包的人学习。该课程还包括一个客户流失案例研究，让您可以测试您的`caret`技能，并获得实际的机器学习经验。你还在等什么？[立即参加课程！](https://web.archive.org/web/20220810040859/https://www.datacamp.com/courses/machine-learning-toolbox/)