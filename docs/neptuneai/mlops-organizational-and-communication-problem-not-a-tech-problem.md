# 当 MLOps 是一个组织和沟通问题而不是技术问题时

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-organizational-and-communication-problem-not-a-tech-problem>

在本文中，您将获得对 [MLOps 及其阶段](https://web.archive.org/web/20221206003726/https://ml-ops.org/content/mlops-principles#:~:text=The%20complete%20MLOps%20process%20includes,%2C%20and%20%E2%80%9CML%20Operations%E2%80%9D.)的简要概述。您还将了解 MLOps 何时是一个组织和沟通问题，何时是一个技术问题，以及如何解决这些挑战。

## 什么是 MLOps？

MLOps 深受 DevOps 概念的启发，在 devo PS 中，开发团队(Dev)和运营团队(Ops)通过系统化的标准流程进行协作。

MLOps，机器学习和运营的结合是将任何数据科学项目的开发和生产对应方结合起来的举措。换句话说，它试图在端到端的 ML 管道中引入结构和透明度，以便数据科学家能够以有组织的方式工作，并与数据工程师和生产端的技术/非技术利益相关者顺利互动。

## 为什么实施 MLOps 会有阻力？

数据科学团队在企业界相对较新，任何数据科学项目的流程仍然无法正确绘制。涉及大量的研究和数据迭代，使用固定的系统无法跟踪这些数据。与常规的工程团队不同，ML 团队的过程和方法不是绝对的，并且经常有一个悬而未决的问题:如果这都不行，还有什么可以？

这就是为什么[数据科学团队](/web/20221206003726/https://neptune.ai/blog/roles-in-ml-team-and-how-they-collaborate)在他们的方法上设置条款、条件和时间表时，常常感到受到限制。

## 为什么需要 MLOps？

想象一下，有两片中间有很多馅的干面包。它会变得很乱，而且很有可能大部分东西会掉下来。解决这个问题的方法是添加一些像奶酪一样的粘性物质，将两片和馅料粘在一起。MLOps 仅仅是:将机器学习项目的整个过程结合在一起的工具**。**

查看 [MLOps 工具前景](https://web.archive.org/web/20221206003726/https://mlops.neptune.ai/)找到最适合您的使用案例的 MLOps 工具。

就像两片面包一样，建模和部署团队经常不同步，缺乏对彼此需求和工作方法的理解。如果没有 MLOps 将这些部分整合在一起，来自两端的努力和辛勤工作在到达最终消费者之前必然会减少。

据估计，在[左右，60%的机器学习项目](https://web.archive.org/web/20221206003726/https://gritdaily.com/why-60-percent-of-machine-learning-projects-are-never-implemented/)从未实施。作为数据科学社区的一员，我目睹了几个由于不可扩展和低效的框架而无法到达最终客户的项目。数据科学对于预测极其重要，但是如果预测仍然局限于数据科学家的系统，对最终客户没有好处。

尽管 MLOps 仍然是一个正在试验的概念，但它可以大大缩短 ML 项目的时间表，并确保更高比例的项目得到部署。通常，全球的 ML 团队都有持续的抱怨，一个项目平均需要花费 [6 个月](https://web.archive.org/web/20221206003726/https://thenewstack.io/add-it-up-how-long-does-a-machine-learning-deployment-take/)来开发和部署。

使用 MLOps，可以显著降低流程的时间复杂度；由于数据科学团队刚刚步入 IT 世界，这是坚持分析和建立正确秩序的最佳时机。

## MLOps 和 DevOps 有什么相似之处？(优势和挑战)

MLOps 的概念是仿照 DevOps 设计的，因此两者之间有一些核心相似之处:

*   **CI/CD 框架**–**持续集成和持续交付**是软件和机器学习解决方案的核心框架，因为对现有代码的改进是不断提出和添加的。
*   **敏捷方法**–敏捷方法旨在通过将端到端过程划分为多个冲刺，使持续集成和改进的过程不那么繁琐。使用 sprint 的好处是每个 sprint 的结果都有很高的透明度，并且分布在项目中涉及的所有团队和涉众之间。这确保了在解决方案开发的后期不会出现意外。总的来说，敏捷方法解决了以下问题:
    *   **漫长的开发到部署周期**——当开发团队构建一个完整的解决方案并将其作为一个整体移交给部署团队时，会出现主要的延迟。在这种转移中有一个很大的缺口，因为许多依赖关系经常没有解决。在敏捷方法的帮助下，交接是冲刺式的，依赖性从开发本身的最初几个阶段就被清除了。
    *   **团队之间缺乏同步**–每个数据科学项目不仅仅包括一个开发和部署团队。许多利益相关者，如项目经理、客户代表和其他决策者也参与其中。如果没有 MLOps 的实施，ML 解决方案的可见性将会大大降低，利益相关者只能在开发的后期阶段尝到它的甜头。这有可能搞乱整个项目，因为众所周知，面向业务的团队会提供能够扭转整个解决方案的专家建议。通过敏捷方法，他们的输入可以从一开始就被采纳，并在所有 sprints 中被考虑。

## MLOps 和 DevOps 有什么不同？

尽管 mlop 和 DevOps 的核心框架相似，但一些明显的差异使得 mlop 更加独特和难以处理。关键原因是数据。数据的变化会给 MLOps 带来额外的麻烦。例如，数据的变化可能导致超参数组合的变化，这可能使建模和测试模块的行为完全不同。让我们看一下 MLOps 的不同阶段，以便更好地理解它们的区别。

## MLOps 的阶段

在 ML 项目中，有五个总括阶段:

1.  **用例识别**:这是开发和业务团队与客户合作来理解和制定手头问题的正确用例的时候。这可能包括计划和软件文档的准备，以及从客户端获得它们的审查和批准。
2.  **数据工程和处理**:这是第一个技术接触点，数据工程师和数据科学家在这里查看各种来源，从这些来源可以获得数据以用于解决方案开发。确定来源之后，构建管道，以正确的格式简化数据，并检查流程中的数据完整性。这需要数据科学家和数据工程师之间的密切合作，也需要不同团队的点头，例如可能使用同一数据主的咨询团队。
3.  **ML 解决方案构建:**这是 MLOps 生命周期中最紧张的部分，在 CI/CD 框架的帮助下，在连续的周期中开发和部署解决方案。一个 ML 解决方案涉及多个实验，有多个版本的模型和数据。因此，每个版本都必须被很好地跟踪，以便在以后的阶段可以很容易地参考最佳的实验和模型版本。开发团队经常不使用标准的工具来跟踪这些实验，最终导致过程混乱，增加了工作和时间需求，并且很可能降低了最佳的模型性能。成功创建的每个模块都应该迭代地部署，同时为更改核心元素(如超参数和模型文件)留出足够的空间。
4.  **生产部署:**一旦在本地环境中构建、测试和部署了 ML 解决方案，就应该将其连接到生产服务器并部署在其上，生产服务器可以是公共云、本地或混合云。接下来是一个简短的测试，检查解决方案是否在生产服务器上工作，这通常是因为相同的解决方案之前已经在本地服务器上测试过。然而，有时由于配置不匹配，测试会失败，开发人员必须迅速查明原因并找到解决方法。
5.  **监控:**解决方案在生产服务器上成功部署后，必须在几批新数据上进行测试。根据数据类型的不同，监控的关键时间从几天到几周不等。一些特定的和预先决定的目标度量用于识别模型是否继续服务于目的。触发器用于在性能低于预期时发出提示警报，并通过返回到解决方案构建的第三阶段来要求优化。

## 当 MLOps 成为组织和沟通的挑战时

*   **漫长的审批链**–对于需要反映在生产服务器上的每一项更改，都必须获得相关机构的批准。这需要很长时间，因为验证过程很长，最终会延迟开发和部署计划。然而，这个问题不仅仅是生产服务器的问题，在提供不同的公司资源或集成外部解决方案方面也存在。
*   **预算内供应**–有时，由于预算限制或资源在多个团队之间共享，开发团队无法使用公司的资源。有时，ML 解决方案也特别需要新的资源。例如，具有高性能计算或巨大存储容量的资源。这样的资源，即使对扩展 ML 解决方案至关重要，也不符合大多数组织的预算标准。在这种情况下，ML 团队必须找到一个通常是次优的解决方案，如果可能的话，让解决方案以同样的活力工作。
*   **不同于自主开发的框架**的 ML 堆栈——大多数公司一直在开发的软件部署框架对于部署 ML 解决方案来说可能是次优的，甚至是不相关的。例如，基于 python 的 ML 解决方案可能必须通过基于 Java 的框架进行部署，以符合公司现有的系统。这可能会给开发和部署团队带来双重工作，因为他们必须复制大部分代码库，这对资源和时间都是一种负担。
*   **安全先行**–每一个机器学习模型或模块几乎总是一个大型知识库的一部分。例如，随机森林模型的一个版本被编码在 scikit-learn 库中，用户可以简单地使用函数调用来启动模型训练。最常见的情况是，用户不知道代码在做什么，也不知道除了数据之外，它还在利用哪些来源。因此，为了确保安全性，在将代码上传到生产服务器之前，需要由相关的安全团队检查封装功能的代码。此外，云上的安全 API、公司的安全基础设施和许可因素是时间密集型的，需要加以注意。
*   **在所有团队之间同步相关数据**–从客户处获得的公共数据主数据由多个团队使用，如销售团队、咨询团队，当然还有开发团队。为了避免数据中的差异，必须维护一组公共的列定义和一个公共的数据源。尽管这听起来是可行的，但是在现实世界的场景中，由于涉及到多个参与者，这通常会成为一个挑战，导致团队之间为了解决内部创建的数据问题而来回奔波。此类问题的一个例子是，在构建了整个解决方案之后，开发人员意识到客户入职团队对其中一列进行了不同的填充。

## 解决 MLOps 中组织和沟通挑战的解决方案

*   **ML 栈的虚拟环境**——对于已经实施了*敏捷方法*并采用了大部分相关框架的新公司和初创公司来说，回溯的本土框架不是问题。然而，相对较老的公司，在以前建立的框架上统一运作，可能看不到 ML 团队在资源优化方面的最佳结果。这是因为团队将忙于计算如何通过可用的框架最好地部署他们的解决方案。一旦弄清楚了，他们必须为他们想要部署的每个解决方案重复次优的过程。
    解决这个问题的长期办法是投入资源和时间创建一个独立的 ML 堆栈，它可以集成到公司框架中，但也可以减少开发方面的工作。对此的快速解决方案是利用虚拟环境为最终客户部署 ML 解决方案。Docker 和 Kubernetes 等服务在这种情况下非常有用。
*   **成本效益分析**—为了减少审批的长队和预算限制，数据科学开发团队通常需要深入业务方面，并对限制供应与可以在这些供应上运行的工作数据科学解决方案的投资回报进行彻底的成本效益分析。团队可能需要与业务运营团队合作，以获得准确的反馈和数据。组织中的关键决策者具有短期或长期利润导向的观点，而承诺增长的成本效益分析可能是打开一些瓶颈的驱动因素。
*   **公共键引用**–为了保持对来自客户的主数据进行操作的多个团队之间的透明度，可以使用引用键，因为在开发团队必须多次复制和下载数据的情况下，公共数据源是不实际的。列定义的公共键可能不是避免数据差异的理想解决方案，但它是对现有过程的改进，在现有过程中，数据列很容易被误解，尤其是当列名没有说明性时。
*   **使用经过验证的代码源**:为了缩短在生产服务器上上传或批准机器学习库时进行安全检查所需的时间，开发人员可以将他们的代码引用限制在经过验证的代码库，如 TensorFlow 和 scikit-learn。如果使用了 Contrib 库，开发人员必须对其进行彻底检查，以验证输入和输出点。这是因为，在有一些安全问题的情况下，重新开发和安全检查的循环重新开始，可能会疯狂地减慢这个过程。

## 当 MLOps 是一项技术挑战时

MLOps 的主要技术挑战出现在解决方案并行开发和部署的第三阶段。以下是一些特定于 MLOps 的技术挑战，这些挑战与 DevOps 完全不同:

*   **超参数版本化**–在找到最佳解决方案之前，每个 ML 模型都必须经过多组超参数组合的测试。然而，这不是主要的挑战。输入数据的变化会降低所选组合的性能，必须重新调整超参数。尽管代码和超参数由开发人员控制，但数据是影响受控元素的独立因素。必须确保数据和超参数的每一个版本都得到跟踪，以便能够以最少的麻烦找到和再现最佳结果。
*   **多次实验**–随着数据的不断变化，大量代码也必须在数据处理、特征工程和模型优化方面进行修改。这种迭代是无法计划的，并且会增加构建 ML 解决方案所需的时间，影响所有后续阶段和计划的时间表。
*   **测试和验证**–该解决方案必须在多个未知数据集上进行测试和验证，以了解该解决方案是否能在生产环境中处理传入的数据。这里的挑战是，经过训练的模型可能会偏向训练数据集。为了避免这一点，测试和验证必须在模型创建本身的迭代中完成。假设建模迭代的次数不固定，用看不见的数据集进行测试所花费的时间会随着迭代次数的增加而增加。
*   **部署后监控**—这里的主要挑战是新数据可能与历史数据模式不一致。例如，在疫情期间，股票市场曲线违背了所有的历史模式，并以机器学习解决方案无法预测的方式下跌。这种外部因素经常起作用，并且必须相应地收集、更新和处理数据，以保持解决方案的最新性。

解决大多数技术挑战的一站式解决方案是系统地记录整个过程，这样就有很高的可见性，并易于访问和导航。记录、存储、组织和比较模型版本和输出是从大量迭代中获得最佳可见性和结果的关键。 [Neptune.ai](/web/20221206003726/https://neptune.ai/) 平台在一个地方管理所有的建模元数据，除了日志、存储、显示、组织和比较之外，它甚至还提供了查询所有 MLOps 元数据的功能。

了解更多关于 Neptune 的特性以及它如何帮助你组织你的 ML 工作流程。

## 结论

作为一个概念，MLOps 是非常新颖的，即使它今天几乎没有触及数据科学和机器学习社区的领域，它也通过强调它确实是当前的需要而留下了印记。尤其是当 ML 实现的数量呈指数级增长时。通过解决一些组织和沟通问题，可以消除 MLOps 的许多多余的麻烦，使其对开发人员和操作更加友好。