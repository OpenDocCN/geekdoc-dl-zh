# NLU 和 NLP  有什么区别

> 原文：<https://www.algolia.com/blog/product/what-is-natural-language-understanding-and-how-is-it-different-from-nlp/>

自然语言理解，也称为 NLU，是一个术语，指的是计算机如何理解人说的和写的语言。是的，这几乎是重复的，但它值得一提，因为虽然 NLU 的建筑是复杂的，结果可能是不可思议的，但 NLU 的潜在目标是非常清楚的。

为例，据估计每天有 3200 亿封电子邮件发送[](https://www.statista.com/statistics/456500/daily-number-of-e-mails-worldwide/)。这是大量自然语言的创造和消费，如果计算机能够更好地理解它，它可以帮助那些与电子邮件互动的人。NLU 可以确定电子邮件是否是垃圾邮件，电子邮件是否具有高优先级，或者是否有其他相关的电子邮件要与收件人共享。所有这些努力都有助于人们充分利用电子邮件。

## [](#the-difference-between-nlu-and-nlp)NLU 和 NLP 的区别

当然，自然语言理解和[自然语言处理或 NLP](https://www.algolia.com/blog/product/what-is-natural-language-processing-and-how-is-it-leveraged-by-search-tools-software/) 之间的区别也是一个一直存在的问题。答案还是在名字里。自然语言处理是关于 *处理* 自然语言，或者取文本，转换成更容易让计算机使用的片段。一些常见的 NLP 任务有 [去除停用词、分词或者拆分复合词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/in-depth/language-specific-configurations/) 。NLP 还可以识别词类或文本中的重要实体。

回到自然语言理解的用途，我们可以想到其他的例子，比如:

*   总结新闻文章和博客文章
*   检测网页的语言以提供翻译
*   确定销售拜访记录中的关键主题
*   对推文中表达的情绪进行分类
*   服务客户服务请求的机器人
*   为搜索请求提供正确的产品
*   智能语音助手

这些例子只是自然语言理解的一小部分。你能想到的任何可以从理解自然语言交流中受益的领域都可能是 NLU 的领域。

## [](#why-natural-language-understanding-is-important)为什么自然语言理解很重要

自然语言的理解是复杂的，而且看起来像魔术一样，因为自然语言是复杂的。语言在很小的空间里包含了大量的信息。一个明显的例子是句子“ [奖杯放不进棕色的手提箱，因为它太大了。](https://cmte.ieee.org/futuredirections/2014/08/20/whats-too-big-the-trophy-or-the-suitcase/) “你可能马上就明白什么是太大了，但这对一台计算机来说真的很难。

我们不能简单地编写一个程序来检查短语“was too big”并理解该短语指的是第一项。首先，因为这个短语可能改为“太大了”或“太重了”或“太大了”第二，因为有公式表明这种“规则”是不成立的，比如“棕色手提箱因为太大而不适合奖杯。”甚至有一些措辞可能会让人们感到困惑，比如“我没有把奖杯放在棕色的行李箱里，因为它太大了。”是奖杯太大放不下行李箱，还是行李箱太大带不动？

## [](#natural-language-understanding-is-built-atop-machine-learning)自然语言理解是建立在机器学习之上的

正是因为这个原因，NLU 非常依赖机器学习。机器学习(ML)可以获取大量文本，并随着时间的推移学习模式。这可以用所谓的 *分布假说* 来解释，该假说认为“通过一个词所结交的朋友”，你可以了解这个词的很多信息以“帽子”这个词为例。一个 ML 模型可能会看到这样的短语，“那个人头上戴着一顶帽子”或者“我戴上帽子是为了遮挡阳光。”如果模型看到了足够多这样的短语，它就开始发现一些模式。然后，抛出这句话，“我戴了一顶棒球帽来遮挡阳光”，它可以感觉到“帽子”和“棒球帽”之间可能有相似之处。加上短语“这个男人头上戴着一顶棒球帽”，相似性会更强。

可以想象，这些 ML 模型需要大量的数据。OpenAI 在 [上训练了他们的 GPT-2 模型 15 亿个参数](https://openai.com/blog/better-language-models/) ，紧接着又在 [上训练了 GPT-3 175 亿个参数](https://arxiv.org/abs/2005.14165) 。这些数据通常是从网上公开可用的数据中抓取的，但随后会在特定的数据集上进行微调。这种微调允许模型更好地理解给定的数据集。例如，微调可以帮助模型更好地理解医疗数据。

过去十年，计算和机器学习的进步增强了 NLU 的力量和能力。我们可以预期，在未来几年内，NLU 将变得更加强大，并更多地集成到软件中。

有关自然语言理解 [应用的更多信息](https://www.algolia.com/blog/product/what-is-natural-language-understanding/) ，以及了解如何在您的网站或应用中利用 Algolia 的搜索和发现 API， [请联系我们的专家团队](https://www.algolia.com/contactus/) 。