<!--yml

category: 未分类

date: 2024-07-01 18:16:57

-->

# PyTorch 开源流程：ezyang 的博客

> 来源：[`blog.ezyang.com/2021/01/pytorch-open-source-process/`](http://blog.ezyang.com/2021/01/pytorch-open-source-process/)

PyTorch 是一个相当大且活跃的开源项目，有时候有人来问我们如何管理 PyTorch，是否有一些经验可以应用到他们自己的项目中。本文试图描述一些截至 2021 年使 PyTorch 作为一个开源项目有效运作的过程。我不会声称我们所做的一切都是处理事务的最佳方式，但至少可以说，这里描述的一切在实践中都是有效的。

**背景。** 并非所有开源项目都一样，PyTorch 有一些独特之处，这些独特之处可能会减少我下文所描述内容在其他背景下的适用性。以下是 PyTorch 作为一个项目的一些定义特征：

+   **大多数全职 PyTorch 开发人员在 Facebook 工作。** 需要明确的是，还有许多全职 PyTorch 开发人员在其他公司工作：NVIDIA、Intel、Quansight、Microsoft、AMD、IBM、Preferred Networks、Google 和 Amazon 都有员工致力于 PyTorch 的开发。但大多数全职开发人员在 Facebook，这使得 PyTorch 区别于业余爱好者的开源项目或由某种基金会运行的项目。

+   **PyTorch 是一个联邦项目。** 如 Nadia Eghbal 所说，PyTorch 是一个具有高贡献者增长和用户增长的项目。在我的[PyTorch 现状（2020）演讲](https://www.youtube.com/watch?v=xwvtzGm8TsI)中，我详细介绍了更多细节，但可以简单地说，我们有超过九家公司为 PyTorch 做贡献，还有一大批其他贡献者（占我们所有提交的 40%）。这使得管理 PyTorch 有时特别具有挑战性，下文描述的许多流程都是为了应对这种活动规模的增长带来的困难。

+   **PyTorch 的表面积很广。** CPU、CUDA、ROCm、ONNX、XLA、serving、分布式、量化等等。对于单个贡献者来说，精通项目的每一个领域是不可能的，因此其中一些挑战就是确保合适的人看到他们需要看到的东西。

好的，那么 PyTorch 是如何处理其规模的？以下是我们所做的一些事情。

**问题分类。** PyTorch 每天收到的 Bug 报告太多，任何一个人都无法跟踪所有这些问题。受到这篇[apenwarr 文章](https://apenwarr.ca/log/20171213)的启发，我们在 Facebook 贡献者之间设置了轮值制度，作为所有这些问题的首次分类的第一线。问题分类的黄金法则是，在分类过程中不修复 Bug；分类的目标是（1）通过适当的 GitHub 标签将 Bug 路由到正确的人，以及（2）寻找高优先级 Bug 并提高对这些 Bug 的认识。每周，我们都会举行会议审查高优先级 Bug（以及其他标记为需要分类审查的 Bug）并讨论这些问题。轮值本身每天轮换一次，以防止人们让一周的问题积压在积压队列中，我们使用一个相对复杂的[搜索查询](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+-label%3Afx+-label%3Ajit+-label%3A%22oncall%3A+jit%22+-label%3Acaffe2+-label%3A%22oncall%3A+quantization%22+-label%3A%22oncall%3A+java%22+-label%3A%22oncall%3A+distributed%22+-label%3A%22oncall%3A+visualization%22+-label%3A%22oncall%3A+mobile%22+-label%3A%22triage+review%22+-label%3Atriaged+updated%3A%3E%3D2019-04-02+sort%3Aupdated-desc+)，以确保只有相关问题显示给轮值处理。

问题分类的最重要后果是你可以整体取消对 PyTorch 仓库的关注。相反，通过观察各种标签（使用我们的[cc bot](https://github.com/pytorch/pytorch/issues/24422)），您可以相信，即使分类人员不知道您对问题感兴趣，也会抄送与主题相关的问题！每周的会议确保所有维护者共同了解当前影响 PyTorch 的主要问题，并帮助社交化我们作为项目认为的“高优先级”问题。最后，[高优先级](https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22high+priority%22) 标签是在项目中寻找有影响力的问题的好方法，即使您对项目不甚了解。

**拉取请求分类。** 同样，我们收到了大量来自一次性贡献者的随意拉取请求。这些人并不处于找到评审者审查他们贡献的好位置，因此我们也有一个分类人员查看这些拉取请求，并确保有人被指派进行审查。如果 PR 特别简单，分类人员可能会直接合并它们。实际上，有一些良好的自动化工具可以做到这一点（例如，[homu](http://huonw.github.io/blog/2015/03/rust-infrastructure-can-be-your-infrastructure/)），但我们懒得设置任何一个，并且手动指定评审者似乎不会增加太多负担在现有轮值之上。

**树拥抱班次。** PyTorch 拥有一个庞大的 CI 系统，涵盖了许多不同的系统配置，大多数贡献者依赖于此来测试他们的更改是否安全。有时候人们会打破主分支。与问题处理班次分开，我们有一个树拥抱班次，他们的工作是在主分支打破时回滚任务。这个班次主要关注 [CI HUD](https://ezyang.github.io/pytorch-ci-hud/build1/pytorch-master)，并在一个配置中导致主分支破坏时回滚提交。

**导入到 Facebook 基础设施。** 实际上，我们直接在 PyTorch 的 HEAD 分支上运行 Facebook 基础设施。使这一切成为可能的工具是 [fbshipit](https://github.com/facebook/fbshipit)，它在 Facebook 的内部单库和我们的公共 GitHub 仓库之间同步提交。这种设置对我们来说有点双刃剑：要求 Facebook 和 GitHub 同步意味着只有 Facebook 员工才能实际提交拉取请求（我们尽量在外部维护者那里简化这一过程，但归根结底还是要有 Facebook 的员工点击绿色按钮），但这也意味着我们不必担心定期进行 "大规模导入" 到 Facebook 基础设施（我们过去曾经做过，而且相当困难）。我们非常有兴趣解决这种情况，并提出了一些关于如何改变我们进行内部发布的建议，以便让外部贡献者直接提交拉取请求。

**RFCs。** 大多数功能讨论发生在 GitHub 问题中，但有时候，某个功能太大、太复杂，无法在 GitHub 问题中充分讨论。在这种情况下，它们可以在 [rfcs 仓库](https://github.com/pytorch/rfcs/) 中讨论（灵感来自 [Rust RFCs 过程](https://github.com/rust-lang/rfcs)）。这个仓库的正式流程尚未完全固定，但通常人们会在那里讨论问题，如果他们觉得在 GitHub 问题中讨论太困难。我们还没有关于引导未经请求的 RFCs 的流程。

**结论。** PyTorch 的开源流程并不像火箭科学那样复杂：有一个班次，班次做一些事情。魔鬼在细节中：所有 PyTorch 的班次职责都经过仔细界定，以确保您的班次职责不会花费无限的时间；它们是您可以在一两个小时内完成并结束一天的事情。您可以说我们过分依赖班次，而自动化可能更好，但我们发现班次需要更少的基础设施投入，并且与 Facebook 现有的流程和工作流程很好地整合在一起。它们可能不适合每个地方，但至少对我们来说，它们似乎做得不错。
