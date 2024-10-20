# 什么是 ModelOps，它与 MLOps 有什么不同？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/modelops>

在过去的几年里，我们已经看到现实生活中人工智能和机器学习解决方案的增加。在大公司中，这些解决方案必须在数百个用例中实施，并且很难手动完成。

在企业层面，人工智能解决方案和机器学习模型的部署需要可操作化。数据科学家和 ML 工程师拥有创建和部署模型的工具，但这只是一个开始。模型需要在生产中部署，以处理真实世界的用例。因此，我们需要一个框架或方法来减少人工劳动并简化 ML 模型的部署。这个框架就是 ModelOps。

它是由 IBM 研究人员在 2018 年 12 月[提出的，被称为“可重用、平台无关、可组合的人工智能工作流的编程模型”。该论文的作者 Waldemar Hummer 和 Vinod Muthusamy 后来将这一想法扩展为“基于云的框架和平台，用于人工智能(AI)应用程序的端到端开发和生命周期管理”。](https://web.archive.org/web/20221218082900/https://en.wikipedia.org/wiki/ModelOps)

> “人工智能(AI)模型操作化(ModelOps)是一组主要关注所有 AI 和决策模型的治理和全生命周期管理的功能。这包括基于机器学习(ML)、知识图、规则、优化、自然语言技术和代理的模型。与 MLOps(只关注 ML 模型的可操作性)和 AI ops(IT 运营的 AI)相反，ModelOps 关注的是所有 AI 和决策模型的可操作性。”–[Gartner](https://web.archive.org/web/20221218082900/https://www.gartner.com/en/information-technology/glossary/artificial-intelligence-model-operationalization-modelops-)

![Model lifecycle governence ModelOps](img/676a1669477b3362934de2c736925311.png)

*Model lifecycle governance | Source: [Forrester](https://web.archive.org/web/20221218082900/https://2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com/wp-content/uploads/2020/09/Forrester_ModelOps.png)*

在 2020 年 3 月的[，ModelOp，Inc .出版了第一份全面的 ModelOps 方法指南。该指南涵盖了模型操作的功能，以及实施模型操作实践的技术和组织要求。](https://web.archive.org/web/20221218082900/https://cio-wiki.org/wiki/ModelOps)

基本上，ModelOps 是部署、监控和管理机器学习模型的工具、技术和最佳实践的集合。这是在企业层面扩展和管理人工智能的关键能力。

![ModelOps Enterprise Capability](img/c531492a3742804bc92eba6512ae8b93.png)

*ModelOps Enterprise Capability | [Source](https://web.archive.org/web/20221218082900/https://cio-wiki.org/images/f/f0/ModelOps_Enterprise_Capability.png)*

它基于 DevOps 的概念，但经过调整以确保机器学习模型的高质量。一般来说，模型操作包括:

您可以将 ModelOps 视为对 [MLOps](/web/20221218082900/https://neptune.ai/blog/mlops-what-it-is-why-it-matters-and-how-to-implement-it-from-a-data-scientist-perspective) 的扩展，其主要重点是通过持续的重新培训和同步部署来保持已部署的模型为未来做好准备。

> “真正的 ModelOps 框架允许您在这些不同的环境中实现标准化和可扩展性，以便开发、培训和部署流程能够以平台无关的方式一致运行。”–[首席信息官-维基](https://web.archive.org/web/20221218082900/https://cio-wiki.org/wiki/ModelOps#cite_note-1)

## 为什么是 ModelOps？企业人工智能的关键

2018 年，Gartner 向大型企业询问 AI 采用情况。经理们曾预计，到第二年，他们 23%的系统将集成人工智能。Gartner 在 2019 年进行了跟进，发现只有 5%的部署成功投入生产。大多数企业无法快速扩展并将人工智能集成到他们的系统中。

这种未部署模型的建立最终会影响公司的发展。模型需要复杂的重新训练。对于每个新的业务领域，都有一组新的数据。此外，模型需要放置在 24/7 运行环境中，因为大多数数据科学家使用开源建模工具，如 Jupyter notebook 或 R studio。数据科学家不知道或无法访问可以观察到延迟的环境。

这就是 ModelOps 的用武之地。它可以帮助解决这些挑战，并使组织能够轻松地扩展和管理人工智能计划。

> “与传统软件相比，模型对业务负有更大的责任。他们必须接受监管审查和合规。一个适当的运营模式可以极大地改变特定业务部门的总体业绩。因此，业务部门和合规部门之间的整合至关重要。”–[福布斯](https://web.archive.org/web/20221218082900/https://www.forbes.com/sites/cognitiveworld/2020/03/31/modelops-is-the-key-to-enterprise-ai/?sh=125f73856f5a)

在过去的几年里，企业已经看到了人工智能开发的激增。但是，正如一项调查显示的那样，*“84%的首席执行官认为他们必须利用人工智能(AI)来实现他们的增长目标，然而 76%的人报告说他们正在为如何扩大规模而挣扎。”*

![Nuances to ModelOps, source: Gartner](img/f6d249cfbb21f523c05c6565831f19d3.png)

*The Nuances of ModelOps | Source: [Gartner](https://web.archive.org/web/20221218082900/https://www.gartner.com/en)*

ModelOps 在动态环境中表现出色。只要定义的条件发生变化，就可以很容易地调整模型。企业针对不同的业务问题采用不同类型的模型。ModelOps 使他们能够相应地切换或扩展系统。

ModelOps 就像是数据科学家、数据工程师、应用程序所有者和基础设施所有者之间的桥梁。它促进动态协作并提高生产率。企业使用 ModelOps 解决以下挑战:

*   **法规遵从性—**为了遵从法规要求，我们需要系统地再现每个模型的培训、评估和评分。模型监控有助于加强合规性关口和控制，确保满足所有业务和法规要求。
*   **筒仓环境–**当一个模型从部署到监控的过程中，涉及到多个团队。跨团队的无效协作会使扩展人工智能变得困难。团队需要走到一起，ModelOps 有助于创建一个环境，在这个环境中，模型可以很容易地从数据科学团队转移到 IT 生产团队。
*   **不同的模型有不同的解决方案——**企业对于不同的业务问题会有上百种模型。每个模型都考虑了特定的业务流程变化、独特的客户细分等。ModelOps 提供单一视图来查看工作流、审计、性能调整和治理，以控制成本和创造价值。
*   **复杂的技术–**有各种各样的解决方案、工具和技术可用于解决数据和分析问题。很难通过持续创新来管理所有这些。即使是最专业的团队也可能跟不上。ModelOps 使集成和采用新技术变得更加容易。

## modelos 用例

许多经理都难以证明分析的价值，因为分析解决方案往往无法投入生产。ModelOps 可以解决这个问题。

以下是 ModelOps 被广泛用于克服模型部署挑战的一些领域:

*   **金融—**例如，银行一直在使用统计模型进行信贷审批，如今大多数运营决策都是由实时分析推动的。这种基于模型的方法帮助银行减少了工时，但大规模管理这些复杂的模型也很困难。这些模型应该是公平和稳健的，这有助于做出公正的决策。ModelOps 使得监控模型偏差或异常并相应地更新它们变得更加容易。
*   **医疗保健–**人工智能可以提高效率和改善患者护理，同时减少代价高昂的管理错误。但是，机器学习模型必须用当前数据、新的 KPI 等来更新。此外，还对异常情况进行监控。更新后的模型应该在不同的系统上随时可用，例如，移动应用程序或实验室的系统，以保持结果同步。
*   **零售—**当新冠肺炎来袭时，一切都必须转移到网上，但很难有效地部署和监控人工智能解决方案。ModelOps 提供了监控模型的能力，并创建关键指标的多级视图来查看生产中的模型性能。为了了解增长领域并减少数据科学家和 IT 专家的工作，零售公司选择了 ML 操作员的自动化和标准化。像 Domino's Pizza 这样的公司能够提高大规模管理多个模型的效率。

## modelos 平台

有几个在线中心可以指导企业完成模型生命周期。一些最常用的平台是:

**2016 年，ModelOp 成立，旨在解决模型部署和维护之间的巨大差距。为了操作机器学习模型，该团队提出了 ModelOp Center。该中心通过确保治理和法规要求的实施，帮助加快了模型的可操作性。**

**![ModelOp ModelOps](img/f39377a64ad5cc87f26b68215a02fc83.png)

*ModelOp on AI Initiatives | Source: [ModelOp](https://web.archive.org/web/20221218082900/https://www.modelop.com/company/)* 

*"* *ModelOp Center 自动治理、管理和监控跨平台和团队部署的 AI、ML 模型，从而实现可靠、合规和可扩展的 AI 计划。”*

模型中心可以帮助您完成模型生命周期中的步骤:

1.  **注册—**在ModelOp center 中，您可以使用 Jupyter 插件或 CLI 注册模型，并通过预定义的元素(模型源、附件、模式和模型平台)传递信息。该模型的源代码可分为 4 个功能:初始化、评分、指标和培训。

您还可以随时通过编辑模式或向其中添加新资产来更新注册的模型。

![ModelOp ModelOps](img/61f1c1a0258f5b7b6b4bcb247d0e253a.png)

*Registering a model | Source: [ModelOp](https://web.archive.org/web/20221218082900/https://modelopdocs.atlassian.net/wiki/spaces/dv24/pages/1216283001/Register+a+Model)*

2.  **编排–**ModelOp Center 有一个 MLC(模型生命周期)管理器，可以自动执行模型操作，如部署、监控和治理。该模型可以在生产中高效、轻松地部署和监控。这也让企业可以灵活地只自动化生命周期的一部分。

MLC 管理器是一个执行、监控和管理 MLC 进程的框架。MLC 流程通过外部事件触发，例如:

1.  基于时间的
2.  数据到达
3.  通知
4.  人工干预
5.  标记为–准备生产

使用 ModelOp Center 可以处理一些常见的 MLC 流程:生产化、刷新、再培训和性能监控。

3.  **监控—**ModelOp Center 为全面的模型监控提供了各种指标。您可以使用 F1 得分、ROC、AUC 等常用指标来评估模型。通过使用 SLA 或数据度量，您还可以监控操作、质量、风险和过程性能。

4.  **治理—**一个中央存储库，用于治理模型生命周期的每一步。治理确保了模型的标准表示。利益相关者可以查看生产模型清单以查看细节，修改和改进模型，或者操作流程。

ModelOp Center 还集成了许多开发平台、IT 系统和企业应用程序。这些集成可以用来扩展人工智能投资，释放人工智能的价值。

![ModelOp integrations](img/a1f9434ac9321660008b092ec7de9d46.png)

*ModelOps: orchestration and model life cycle | Source: [ModelOp](https://web.archive.org/web/20221218082900/https://www.modelop.com/blog/including-modelops-in-your-ai-strategy/)*

有了 ModelOp Center 这样的平台，企业可以加快模型部署速度，降低业务风险。

建立模型不是问题，而是组织。我们正在构建 Datatron 来解决这一问题，方法是加快部署、及早发现问题，并提高大规模管理多个模型的效率。

![Datatron ModelOps](img/1e33286bb4d6092d7d43ec0d2fd575f7.png)

*Automating and accelerating ML management | Source: [Datatron](https://web.archive.org/web/20221218082900/https://www.datatron.com/platform/)*

Datatron 是一个一站式解决方案，您可以在其中自动化、优化和加速 ML 模型。它支持广泛的框架和语言，如 TensorFlow、H2O、Scikit-Learn、SAS、Python、R、Scala 等。此外，它支持内部和基于云的基础架构。他们将模型操作活动分为以下几类:

1.  **模型目录—**探索由您的数据科学团队构建和上传的模型，所有这些都来自一个集中的存储库。
2.  **模型部署–**通过几次点击创建可扩展的模型，并使用任何语言或框架进行部署。

3.  **模型监控—**创建条件警报，比较模型预测，检查模型准确性，并检查模型是否会衰退。

4.  **模型治理–**您可以轻松验证您的模型并执行内部审计。该平台还可以用来调试和解释模型。

5.  **模型管理—**动态在运行时选择最佳模型，以帮助提供更好的预测并减少错误。强制执行延迟以防止任何服务中断，并对模型序列执行 A/B 测试。
6.  **模型工作流–**设置工作流有助于将业务逻辑与模型结果相集成。例如，您可以设置预处理工作流，在该工作流中定义数据源和特征工程流程，然后将它们输入到模型中。

SAS 模型管理软件用于简化分析模型的部署和管理。但是，企业在实施和持续增值方面需要帮助，因此 SAS 推出了 ModelOps。这是一个包含模型管理器软件和服务的组合包。一些关键特征是:

1.  管理分析模型
2.  交付详细的项目计划
3.  评估当前模型
4.  SAS 模型管理器的实现和激活支持。

Superwise 每天保证多个预测的健康。它可以预测性能随时间和版本的变化，并使您能够在没有反馈时获得可见性。

![Superwise ModelOps](img/8880eb3987bd5dbf103acfcfec12d1df.png)

*EmpoweringAI Stakeholder | Source: [Superwise](https://web.archive.org/web/20221218082900/https://www.superwise.ai/)*

**它支持高级事件管理，有助于模型监控并提供模型性能的粒度可见性。**

**![Superwise ModelOps](img/a1484df181f809413c0d1e0d6e44899f.png)

*One solution to monitor, analyze and optimize | Source: [Superwise](https://web.archive.org/web/20221218082900/https://www.superwise.ai/solution)*

它是 API 驱动的，因此解决方案可以轻松集成到现有环境中。它可以通过查看数据来发现功能、生成 KPI 和设置阈值。

Modzy 成立于 2019 年，其明确的目的是建立一个人类和机器共同工作的世界，胜过单独工作。

“Modzy 为人工智能提供了十年前不存在的规模，它拥有一个 ModelOps 平台和现成模型的市场。”

![Modzy ModelOps](img/46c062115685d84302f22500fd084dd6.png)

*Easy integration | Source: [Modzy](https://web.archive.org/web/20221218082900/https://www.modzy.com/platform/)*

*   Modzy 集成了广泛的工具、框架、数据管道和 CI/CD 系统。
*   它提供了预定义模型的列表，您可以在新的输入数据中重用这些模型。

![Modzy ModelOps](img/071a65154d806f64be996aee82437df1.png)

*Searching for a sentiment analysis model | Source: [Modzy](https://web.archive.org/web/20221218082900/https://docs.modzy.com/docs)*

*   它拥有 Python、JavaScript 和 Java SDKs，使开发人员能够轻松地将 ML 功能集成到他们的应用程序中。

## modelos 的优势

在前面的章节中，我们谈到了 ModelOps 给企业带来了什么，以及为什么他们需要与 ModelOps 集成来扩展 AI 系统。让我们探索更多这些优势:

*   **加速部署—**许多大型企业发现按时交付人工智能解决方案具有挑战性。要么是缺乏熟练的数据科学家，要么是调整或发现模型的问题需要太多时间。ModelOps 平台提供了所有模型和各种管道的单一视图，有助于加速模型部署。这样，企业可以更专注于创新和价值创造。
*   **减轻模型漂移–**可能有数百个模型，管理它们可能会很困难。ModelOps 有助于解释模型是如何得到的，并测量它们的公平性，同时有助于纠正是否存在任何偏离定义基线的情况。
*   **人工智能结果驱动——有价值的见解—**ModelOps 通过生成关键见解和模式，帮助将模型结果映射到业务 KPI。管理者可以利用自动化、预测和优化等战略手段。这有助于创建符合您业务需求的解决方案。
*   **更简单的入门–**model ops 是一个统一的环境，可带来巨大收益，同时减少在模型构建、部署和管理上投入的时间。这些平台帮助您装载模型和流程，并让您通过几次点击来监控和管理它们。
*   **经济—**ModelOps 平台与云集成，有助于经济地优化云服务和人工智能模型。企业可以选择灵活的服务消费进行建模。

## modelos 的特点

我们在上一节中已经介绍了一些 ModelOps 工具，它们中的大多数都有共同的特征，因为它们都服务于一个目标——将模型操作化。

ModelOps 是开启价值的钥匙，它是你的人工智能技术栈中的结缔组织。让我们来看看其中的一些特性，看看为什么 ModelOps 如此重要:

*   **生成模型管道—**只需最少的人工干预和几次点击，ModelOps 平台就能自动生成管道。一旦第一次设置完成，整个建模生命周期——准备数据、选择模型、特征工程、超参数优化——将实现自动化。
*   **监控模型–**监控数百种不同的模型是一项挑战。ModelOps 有助于监控各种型号。它寻找任何可能的偏见，然后学习如何解决它。
*   **部署模型–**在 ModelOps 平台上构建的模型可以轻松集成到任何应用程序中。您几乎可以在任何地方部署和发送模型。
*   **一站式解决方案–构建、运行和管理模型–**数据科学家使用开源平台(如 Jupyter、R 等)构建模型。然后，IT 专业人员使用不同的平台在生产中部署这些模型。这一过程通常需要时间，并缩短上市时间。ModelOps 平台旨在通过这一复杂而漫长的过程拯救企业。它们是一站式解决方案，团队可以在其中轻松地构建、运行和管理模型。

## ModelOps 和 MLOps 一样吗？

ModelOps 和 MLOps 之间只有一线之隔，如果你观察它们的架构，你会发现 ModelOps 是 MLOps 的延伸。模型操作指的是在生产中操作和管理人工智能模型的过程。ModelOps 可以被视为 SDLC，它帮助组织生命周期流程，即创建模型、测试、部署和监控它。

ModelOps 做 MLOps 做的一切，甚至更多。它有助于企业从人工智能投资中获得更多回报。MLOps 使数据科学家和 IT 专业人员能够有效地协作和沟通，同时自动化机器学习模型。ModelOps 是 MLOps 的发展，关注于 ML 模型的持续再培训、同步开发和部署、决策优化和转换模型。

![ModelOps vs MLOps](img/af409bc931a724989c1aeba4da2e1c8f.png)

*AI at Scale with MLOps & ModelOps | [Source](https://web.archive.org/web/20221218082900/https://www.modzy.com/blog/modelops-vs-mlops/)*

“ModelOps 已经成为解决人工智能部署最后一英里交付挑战的关键环节。ModelOps 是 MLOps 的超集，指的是在生产系统中操作和管理人工智能模型所涉及的过程。”–[*Modzy*](https://web.archive.org/web/20221218082900/https://www.modzy.com/blog/modelops-vs-mlops/)

## 模型操作 vs 模型操作

很容易混淆 ModelOps 和 MLOps。为了理解这些差异，我们需要知道它们到底是如何帮助建模的。创建可扩展的人工智能解决方案需要这两者。

让我们来看看这两者之间的一些共同差异:

| 性能 | MLOps | 莫德洛斯 |
| --- | --- | --- |
|  | 

关注机器学习模型的可操作化

 | 

所有 AI 和决策模型可操作化

 |
|  | 

模型开发、部署和性能监控的连续循环

 | 

关注模型的治理和全生命周期管理

 |
|  | 

亚马逊 SageMaker，海王星，DataRobot，MLFlow，还有[更有](https://web.archive.org/web/20221218082900/https://neptune.ai/blog/best-mlops-tools)

 | 

Cnvrg，Cloudera，ModelOps，Modzy，SAS

 |
|  | 

旨在通过为各种团队和利益相关者创建有效的协作环境来创建支持人工智能的应用

 | 

使用仪表盘提供人工智能使用的透明度，向业务领导汇报

 |

要了解更多差异，请检查 ModelOp [中心](https://web.archive.org/web/20221218082900/https://www.modelop.com/modelops-and-mlops/)。

## 结论

通过这篇文章，我们了解了什么是 ModelOps，以及企业如何使用它来操作 AI 解决方案。有各种各样的平台和工具可以帮助创建模型工作流，只需点击几下鼠标就可以进行监控和治理。我们还确定了 MLOps 和 ModelOps 是如何不同的，但是属于相同的概念。

总之，使用 ModelOps，创建人工智能解决方案并将其部署到生产中的企业将不得不不断更新这些模型，以便它们不会过时，并能够跟上市场。有了 ModelOps，企业可以自动完成这些更新。

ModelOps 解决方案通过为商业领袖提供量身定制的信息和见解，解决了当今人工智能采用中最紧迫的问题之一。这种在整个企业中对人工智能使用的透明性，以一种商业领袖可以理解的方式为模型提供了可解释性。一句话:ModelOps 促进信任，从而增加人工智能的采用。”–[*数据科学中心*](https://web.archive.org/web/20221218082900/https://www.datasciencecentral.com/profiles/blogs/modelops-vs-mlops)****