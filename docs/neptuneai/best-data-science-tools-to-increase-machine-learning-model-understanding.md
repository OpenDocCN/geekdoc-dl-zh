# 增强机器学习模型理解的最佳数据科学工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/best-data-science-tools-to-increase-machine-learning-model-understanding>

有一个广泛的工具目录，可以用来帮助您增加对机器学习模型的理解。它们分为不同的类别:

在本文中，我将尽可能简要地向您介绍一些工具，向您展示 ML 工具生态系统是多么丰富。

## 1.交互式 web 应用工具

这个开源的 ML 工具允许你为你的模型构建定制的 web 应用。你可以用一种非常互动和易懂的方式展示你的模型，这样任何人都可以很容易地使用它。

只需几分钟，您就可以构建和部署漂亮而强大的数据应用程序。

要使用 Streamlit，您只需使用以下命令将它与 pip 一起安装:

```py
pip install streamlit
```

Streamlit 的用户界面会询问你是否想重新运行应用程序并查看更改。这允许您在快速迭代循环中工作:您编写一些代码，保存它，检查输出，再编写一些，等等，直到您对结果满意为止。

查看 [Streamlit 文档](https://web.archive.org/web/20221206043148/https://docs.streamlit.io/en/stable/getting_started.html)以了解它是如何工作的。

用于构建 web 应用程序的最流行的轻量级 Python 框架之一。您可以使用它为您的模型开发一个 web API。它基于 Werkzeug WSGI 工具包和 Jinja2 模板引擎。

Flask 有一个简单的架构，你可以非常容易地学习它。强烈建议将其用于构建小型应用程序。您可以快速部署模型并设置 REST API。

要开始使用 Flask，请设置虚拟环境，并使用以下命令安装它:

```py
pip install flask
```

有关详细信息，请查看[烧瓶文档。](https://web.archive.org/web/20221206043148/https://flask.palletsprojects.com/en/1.1.x/)

Shiny 是一个用于构建交互式 web 应用程序的 R 包。如果你已经知道 R 语言，那么用它来构建一个应用程序和分享你的作品将会非常容易。它有一个真正的交互式和直观的设计。

你可以在网页上托管独立的应用程序，或者将它们嵌入到 [R Markdown](https://web.archive.org/web/20221206043148/https://rmarkdown.rstudio.com/) 文档中，或者构建[仪表盘](https://web.archive.org/web/20221206043148/https://rstudio.github.io/shinydashboard/)。你也可以用 [CSS 主题](https://web.archive.org/web/20221206043148/https://rstudio.github.io/shinythemes/)、 [htmlwidgets](https://web.archive.org/web/20221206043148/https://www.htmlwidgets.org/) 和 JavaScript [动作](https://web.archive.org/web/20221206043148/https://github.com/daattali/shinyjs/blob/master/README.md)来扩展你闪亮的应用。

要开始使用 Shiny，请使用以下命令:

```py
install.packages("shiny")
```

关于 Shiny 的详细信息，查看他们的[官方教程](https://web.archive.org/web/20221206043148/https://shiny.rstudio.com/tutorial/written-tutorial/lesson1/) [。](https://web.archive.org/web/20221206043148/https://shiny.rstudio.com/tutorial/written-tutorial/lesson1/)

## 2.数据分析工具

DABL 代表数据分析基线库。您可以使用它来自动化在模型开发的早期阶段发生的重复过程，比如数据清理、预处理、分析、处理缺失值或者将数据转换成不同的格式。

这是一个新的 Python 库，所以它的功能有限，但它非常有前途。

要开始使用 DABL，请使用以下命令:

```py
pip install dabl
```

想了解更多关于 DABL 的信息，请看这篇文章。

用于创建数据科学应用程序和构建机器学习模型的开源数据分析工具。

您可以通过其模块化数据管道概念来集成机器学习和数据挖掘的各种组件。KNIME 已经被用于 CRM、文本挖掘和商业智能等领域。

它提供了一个交互式 GUI 来使用拖放构建器创建工作流。它支持多线程的内存数据处理，KNIME server 支持基于团队的协作。

要开始使用 KNIME，请访问[文档](https://web.archive.org/web/20221206043148/https://docs.knime.com/2020-07/analytics_platform_installation_guide)。

帮助您准备和分析数据的数据科学平台。非常用户友好，你可以拖放代码。

它有几个数据探索功能，您可以使用这些功能从您的数据中获得有价值的见解。它为数据分析提供了超过 14，000 个运算符。

要开始使用 RapidMiner，请点击[此链接。](https://web.archive.org/web/20221206043148/https://rapidminer.com/get-started/)

SAS 代表统计分析系统，它是用来分析统计数据的。它帮助您使用 SAS SQL 和自动代码生成进行数据分析。你可以很容易地将它与 Excel 等 MS 工具集成。

SAS 允许您创建交互式仪表板和报告，以更好地了解复杂的数据。

要开始使用 SAS，请查看本教程。

## 3.模型可解释性工具

Eli5 是一个 Python 包，可以让你解释机器学习分类器的预测。它支持以下包和框架:

*   XGBoost–解释 XGBClassifier、XGBRegressor 的预测，并帮助验证特性的重要性。
*   CatBoost–解释 CatBoostClassifier、CatBoostRegressor 的预测，并帮助验证功能的重要性。
*   Scikit-learn–解释 sci kit-learn 线性分类器和回归器的权重和预测，并帮助验证决策树的特征重要性。

要了解更多关于 Eli5 的信息，请查看[文档](https://web.archive.org/web/20221206043148/https://eli5.readthedocs.io/en/latest/overview.html)。

它代表沙普利加法解释。它基于沙普利价值观。SHAP 值通过分解预测来显示每个要素的影响，这可能会产生强大的模型洞察力。

SHAP 值的一些应用有:

*   一个模型说，银行不应该贷款给某人，法律要求银行解释每次拒绝贷款的依据。
*   医疗保健提供者希望确定哪些因素导致了患者的疾病风险，以便通过有针对性的健康干预措施直接解决这些风险因素。

要了解更多关于 SHAP 的信息，请查看这篇教程。

它代表描述性机器学习解释。这是一个 R 包，主要是为了模型的可解释性而构建的。

机器学习的可解释性变得越来越重要。这个软件包可以帮助您提取洞察力，并清楚地了解您的算法是如何工作的，以及为什么一个预测优于另一个预测。

Dalex 便于比较多种型号的性能。使用 Dalex 有几个[优势:](https://web.archive.org/web/20221206043148/https://medium.com/@ModelOriented/dalex-v-1-0-and-the-explanatory-model-analysis-419585a4ba91)

*   它包括用于局部解释的独特和直观的方法，
*   它可以定制预测的输出，
*   它为结果比较提供了方便的方法。

要了解更多关于 Dalex 的信息，请查看官方 GitHub 页面。

* * *

## 4.模型调试工具

一个机器学习的可视化调试工具，由优步的团队开发。它的开发是为了使模型迭代过程更加明智和可行。

数据科学家可以使用它来查看总体摘要，并检测数据中预测不准确的部分。Manifold 还通过显示性能较好和较差的数据子集之间的特征分布差异，解释了模型性能较差的潜在原因。

Manifold 服务于大多数 ML 模型，包括大多数分类和回归模型。

要了解更多关于优步流形的信息，请查看官方页面。

## 5.模型性能调试工具

MLPerf 正在成为 ML 工作负载中有趣实验的主要部分，比较不同类型的专用基础设施或软件框架。它用于建立有用的标准来衡量 ML 硬件、软件和服务的培训绩效。

MLPerf 的目标是服务于商业和学术团体，确保可靠的结果，加速 ML 的进展，并使竞争系统的公平比较成为可能，同时鼓励创新以改进最先进的 ML。

要了解更多关于 MLPerf 的信息，请查看这篇介绍性文章。

## 6.实验跟踪工具

面向数据科学家的轻量级且功能强大的[实验跟踪](/web/20221206043148/https://neptune.ai/experiment-tracking)工具。它可以轻松地与您的工作流程集成，并提供广泛的跟踪功能。

您可以使用它来跟踪、检索和分析实验，或者与您的团队和经理共享它们。Neptune 非常灵活，可以与许多框架一起工作，并且由于其稳定的用户界面，它支持很好的可伸缩性(达到数百万次运行)。

它还允许您存储、检索和分析大量数据。

要了解更多关于海王星的信息，请查看[网站](/web/20221206043148/https://neptune.ai/product)。

它代表重量和偏见。这是一个 Python 包，可以让你实时监控模型训练。它很容易与 Pytorch、Keras 和 Tensorflow 等流行框架集成。

此外，它允许您将运行组织到项目中，在项目中您可以轻松地比较它们，并确定最佳执行模型。

在这篇介绍性文章中了解更多关于 WandB 的信息。

Comet 帮助数据科学家管理和组织机器学习实验。它可以让您轻松地比较实验，记录收集的数据，并与其他团队成员合作。它可以很容易地适应任何机器，并与不同的 ML 库很好地合作。

要了解更多关于 Comet 的信息，请查看官方文档。

用于跟踪机器学习实验和部署模型的开源平台。每个元素都由一个 MLflow 组件表示:跟踪、项目和模型。

这意味着，如果你正在使用 MLflow，你可以轻松地跟踪一个实验，组织它，为其他 ML 工程师描述它，并将其打包成一个机器学习模型。

MLflow 旨在实现从一个人到大型组织的可伸缩性，但它最适合个人用户。

要了解更多关于 MLflow 的信息，请查看官方文档。

## 7.生产监控工具

Kubeflow 使得部署机器学习工作流变得更加容易。它被称为 Kubernetes 的机器学习工具包，旨在利用 Kubernetes 的潜力来促进 ML 模型的扩展。

Kubeflow 背后的团队正在不断开发其功能，并尽最大努力让数据科学家的生活更轻松。Kubeflow 有一些跟踪功能，但它们不是该项目的主要焦点。作为补充工具，它可以很容易地与列表中的其他工具一起使用。

要了解更多关于 Kubeflow 的信息，请查看官方文档。

## 结论

这就结束了我们的不同机器学习工具的列表。正如您所看到的，生态系统是广泛的，这个列表并没有涵盖所有的工具。

有什么需求就有什么需求，不需要手动做事。使用这些工具来加快您的工作流程，让您作为数据科学家的生活更加轻松。

祝你好运！