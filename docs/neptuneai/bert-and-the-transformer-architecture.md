# 关于正在重塑人工智能格局的 BERT 和 Transformer 架构，你需要知道的 10 件事

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/bert-and-the-transformer-architecture>

目前，很少有人工智能领域比 NLP 更令人兴奋。近年来，可以执行类似人类语言任务的语言模型(LM)已经发展到比任何人预期的更好。

事实上，他们表现得如此之好，以至于[人们怀疑](/web/20230308023938/https://neptune.ai/blog/ai-limits-can-deep-learning-models-like-bert-ever-understand-language)他们是否达到了[一般智力](https://web.archive.org/web/20230308023938/https://chatbotslife.com/is-gpt-3-the-first-artificial-general-intelligence-a7390dca155f)的水平，或者我们用来测试他们的评估标准跟不上。当像这样的技术出现时，无论是电力、铁路、互联网还是 iPhone，有一点是明确的——你不能忽视它。它将最终影响现代世界的每一个部分。

了解这样的技术很重要，因为这样你就可以利用它们。所以，我们来学习吧！

我们将涵盖十个方面，向您展示这项技术的来源、开发方式、工作原理以及在不久的将来会有什么样的前景。这十件事是:

1.  **什么是 BERT 和变压器，为什么我需要了解它？**像 BERT 这样的模型已经对学术界和商业界产生了巨大的影响，因此我们将概述这些模型的一些使用方法，并澄清围绕它们的一些术语。
2.  在这些模型之前，我们做了什么？要了解这些模型，重要的是要了解这一领域的问题，并了解在 BERT 等模型出现之前我们是如何解决这些问题的。通过这种方式，我们可以了解以前模型的局限性，并更好地理解 Transformer 架构关键设计方面背后的动机，这是大多数 SOTA 模型(如 BERT)的基础。
3.  **NLPs“ImageNet 时刻；预训练模型:**原来我们都是自己训练模型，或者你要针对某个特定任务，全面训练一个模型。实现性能快速发展的关键里程碑之一是创建预先训练的模型，这些模型可以“现成”使用，并根据您的具体任务进行调整，只需很少的努力和数据，这一过程称为迁移学习。理解这一点是理解为什么这些模型在一系列 NLP 任务中一直表现良好的关键。
4.  **了解变形金刚:**你可能听说过伯特和 GPT-3，但是关于[罗伯塔](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/roberta.html)、[艾伯特](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/albert.html)、 [XLNet](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/xlnet.html) ，或者[龙前](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/longformer.html)、[改革者](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/reformer.html)，或者 [T5 变形金刚](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/model_doc/t5.html)呢？新模型的数量看起来势不可挡，但是如果您理解 Transformer 架构，您将有机会了解所有这些模型的内部工作方式。这和你理解 RDBMS 技术的时候是一样的，让你很好的掌握 MySQL、PostgreSQL、SQL Server 或者 Oracle 之类的软件。支撑所有数据库的关系模型与支撑我们的模型的转换器架构是一样的。明白了这一点，RoBERTa 或 XLNet 就成了使用 MySQL 或 PostgreSQL 的区别。学习每个模型的细微差别仍然需要时间，但是你有一个坚实的基础，你不是从零开始。
5.  **双向性的重要性**:当你读这篇文章时，你并没有严格地从一边读到另一边。你不是从一边到另一边一个字母一个字母地读这个句子。相反，你正在向前跳跃，从你现在所处的位置之前的单词和字母中学习上下文。事实证明，这是变压器架构的一个关键特性。Transformer 架构支持模型以双向方式处理文本，从开始到结束，从结束到开始。这是以前模型局限性的核心，以前的模型只能从头到尾处理文本。
6.  伯特和变形金刚有什么不同？ BERT 使用 Transformer 架构，但在几个关键方面有所不同。对于所有这些模型，理解它们与 Transformer 的不同是很重要的，因为这将定义它们可以做好哪些任务，以及它们将努力完成哪些任务。
7.  **记号化器——这些模型如何处理文本**:模型不会像你我一样阅读，所以我们需要对文本进行编码，以便它可以被深度学习算法处理。如何对文本进行编码对模型的性能有很大的影响，这里的每个决定都要进行权衡。所以，当你看另一个模型时，你可以先看看所使用的记号赋予器，并且已经了解了关于那个模型的一些东西。
8.  **掩饰——聪明的工作与努力的工作**:你可以努力工作，也可以聪明地工作。和深度学习 NLP 模型没什么区别。这里的艰苦工作只是使用一个普通的 Transformer 方法，并向模型中投入大量数据，使其性能更好。像 GPT-3 这样的模型有令人难以置信的大量参数，使它能够以这种方式工作。或者，您可以尝试调整训练方法，以“强制”您的模型从更少的内容中学到更多。这就是像伯特这样的模型试图用掩蔽来做的事情。通过理解这种方法，你可以再次用它来观察其他模型是如何被训练的。他们是否采用创新技术来提高这些模型能够从给定的数据中提取多少“知识”？还是他们采取了一种更野蛮的方法，扩大规模，直到你打破它？
9.  **微调和迁移学习**:BERT 的主要优势之一是它可以针对特定领域进行微调，并针对许多不同的任务进行培训。像伯特和 GPT-3 这样的模型是如何学习执行不同的任务的？
10.  鳄梨椅——伯特和其他变形金刚型号的下一步是什么？为了回顾 BERT 和 Transformer 架构，我们将展望这些模型的未来

## 1.什么是伯特和变压器，为什么我需要了解它？

![bert_models_layout](img/6954d9ab949eec4f7688eac8ab733d06.png)

*The Transformer (Muppet) family | Source:* [*PLM Papers*](https://web.archive.org/web/20230308023938/https://github.com/thunlp/PLMpapers)

为了理解 BERT 和 Transformer 的范围和速度，让我们看看这项技术的时间框架和历史:

*   **2017**:Transformer 架构于 2017 年 12 月在一篇谷歌机器翻译论文中首次发布“ [*注意力是你所需要的全部*](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1706.03762.pdf) ”。那篇论文试图找到能够自动翻译多语言文本的模型。在此之前，许多机器翻译技术都涉及一些自动化，但它受到重要规则和基于语言的结构的支持，以确保翻译对于像 Google Translate 这样的服务来说足够好。
*   **2018** : BERT(来自变压器的双向编码器表示)于 2018 年 10 月在“ [*深度双向变压器语言理解预训练*](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1810.04805.pdf) 中首次发布。

![Google translate Transformer](img/3785ded066ffe5a192f72da3544af69f.png)

*Improvements in Google translate with the Transformer | Source: [Google AI Blog](https://web.archive.org/web/20230308023938/https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)*

起初，Transformer 主要影响机器翻译领域。新方法带来的改进[很快被注意到](https://web.archive.org/web/20230308023938/https://techcrunch.com/2017/08/31/googles-transformer-solves-a-tricky-problem-in-machine-translation/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAADJP_SK_QC2Ss-ElKYgaRQRvlQAQXzl4uQah3ux6XBsZJ1v9vAvSYvx2pkdvdG22tEeGn7KQjgbXy9bmaFW-y5RjzLO78NPul_MaiBLr845GoXNvZhFkdWzmoVJ4yy2urF0dGjIoRvlpP-iJXVUo1sS0JT4GWTWb09sUjuMDvXO7)。如果它停留在翻译领域，那么可能现在就不会读这篇文章了。

翻译只是一系列自然语言处理任务中的一项，包括词性标注、命名实体识别(NER)、情感分类、问答、文本生成、摘要、相似性匹配等等。以前，这些任务中的每一个都需要一个经过特殊训练的模型，所以没有人需要学习所有的任务，而且你通常只对你自己的领域或特定的任务感兴趣。

然而，当人们开始关注 Transformer 架构，并怀疑它是否能做一些不仅仅是翻译文本的事情时，这种情况发生了变化。他们观察了该架构能够“集中注意力”于特定单词并比其他模型处理更多文本的方式，意识到这可以应用于广泛的其他任务。

我们将在第 5 节中讨论转换器的“注意力”能力，我们将展示它如何使这些模型能够双向处理文本，并查看与当时正在处理的句子的特定上下文相关的内容。

一旦人们意识到 Transformer 架构可以被分解，并以不同的方式执行一系列任务，它的影响就开始迅速增长:

正是在这一点上，《变形金刚》超越了 NLP。突然间，人工智能的未来不再是有感知能力的机器人或自动驾驶汽车。

如果这些模型可以从文本中学习上下文和意义，并执行广泛的语言任务，这是否意味着它们理解文本？他们会写诗吗？他们会开玩笑吗？如果它们在某些 NLP 任务中表现得比人类更好，这是一般智力的一个例子吗？

像这样的问题意味着这些模型不再局限于聊天机器人和机器翻译的狭窄领域，而是它们现在是人工智能一般智能更大辩论的一部分。

![NLP BERT article conscious AI](img/e09c04ce183dddf90afa4a594d306805.png)

*“… in a lecture published Monday, Bengio expounded upon some of his earlier themes. One of those was attention — in this context, the mechanism by which a person (or algorithm) focuses on a single element or a few elements at a time. It’s central both to machine learning model architectures like Google’s Transformer and to the bottleneck neuroscientific theory of consciousness, which suggests that people have limited attention resources, so information is distilled down in the brain to only its salient bits. Models with attention have already achieved state-of-the-art results in domains like natural language processing, and they could form the foundation of enterprise AI that assists employees in a range of cognitively demanding tasks”. | Source: [VentureBeat](https://web.archive.org/web/20230308023938/https://venturebeat.com/2020/04/28/yoshua-bengio-attention-is-a-core-ingredient-of-consciousness-ai/)*

像任何范式转换技术一样，了解它是被夸大还是被低估是很重要的。最初，人们认为电力不是一项变革性的技术，因为它需要时间来重新定位工作场所和城市环境，以利用电力提供的优势。

铁路也是如此，互联网诞生之初也是如此。关键是，不管你同意还是不同意，你至少需要对手边快速发展的技术有一些看法。

这就是我们现在对伯特这样的模型的看法。即使你不使用它们，你仍然需要理解它们可能对人工智能的未来产生的潜在影响，以及如果它让我们更接近开发普遍智能的人工智能——对社会的未来产生的影响。

## 2.伯特之前存在什么？

BERT 和 Transformer 架构本身，都可以在他们试图解决的问题的上下文中看到。像其他商业和学术领域一样，机器学习和 NLP 的进展可以被视为试图解决当前技术的缺点或不足的技术进化。亨利·福特使汽车更便宜、更可靠，因此它们成为了马的可行替代品。电报改进了以前的技术，能够与人交流而不需要亲自到场。

在 BERT 之前，NLP 的最大突破是:

*   **2013** : Word2Vec 论文， [*向量空间中单词表征的高效估计*](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1301.3781.pdf) 发表。连续单词嵌入开始被创建，以更准确地识别单词的语义和相似性。
*   **2015** : Sequence to sequence 的文本生成方法 [*一篇论文发布了一个神经对话模型*](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1301.3781.pdf) 。它建立在 Word2Vec 中首次展示的一些技术之上，即深度学习神经网络从大量非结构化文本中学习语义和句法信息的潜在能力。
*   **2018** :来自语言模型的嵌入(ELMo)论文 [*深度语境化词语表征*](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1802.05365.pdf) 发布。ELMo(这是整个布偶命名的开始，不幸的是，它并没有停止，见[厄尼](https://web.archive.org/web/20230308023938/https://arxiv.org/abs/1905.07129)、[大鸟](https://web.archive.org/web/20230308023938/https://arxiv.org/abs/2007.14062)和 [KERMIT](https://web.archive.org/web/20230308023938/https://arxiv.org/abs/1906.01604) )在单词嵌入方面是一个飞跃。它试图解释使用一个单词的上下文，而不是静态的，一个单词，一个意思的 Word2Vec 的限制。

在 Word2Vec 之前，单词嵌入要么是使用 [one hot](https://web.archive.org/web/20230308023938/https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) 编码技术的具有大量稀疏向量的简单模型，要么是我们使用 [TF-IDF](https://web.archive.org/web/20230308023938/https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/) 方法来创建更好的嵌入，以忽略常见的低信息量单词，如“*”、“ *this* ”、“ *that* ”。*

*![One hot embeddings](img/a2aa0b1600d46feedb2d10e6c0bfc2aa.png)

*One hot embeddings are not really useful since they fail to show any relationship between words, source:* [*FloydHub blog*](https://web.archive.org/web/20230308023938/https://blog.floydhub.com/automate-customer-support-part-one/)

这些类型的方法在它们的向量中编码很少的语义意义。我们可以用它们来分类文本和识别文档之间的相似性，但是训练它们是困难的，并且它们的整体准确性是有限的。

Word2Vec 通过设计两种新的神经网络架构改变了这一切；跳跃式语法和连续词袋(CBOW)让我们能够在大量文本中训练单词嵌入。这些方法迫使神经网络在给出句子中其他单词的一些例子的情况下尝试并预测正确的单词。

这种方法背后的理论是[常用词将一起使用](https://web.archive.org/web/20230308023938/https://en.wikipedia.org/wiki/Distributional_semantics)。例如，如果你在谈论“*手机*”，那么我们很可能也会看到类似“*手机*”、 *iPhone* 、 *Android* 、*电池*、*触摸屏*等词汇。这些词可能会在足够频繁的基础上同时出现，以至于该模型可以开始设计一个具有权重的大向量，这将有助于它预测当它看到“*电话*”或“*移动*”等时可能会发生什么。然后，我们可以使用这些权重或嵌入来识别彼此相似的单词。

![Word2Vec_embeddings](img/e9a62f6c68dd012561c0bea9d532b501.png)

*Word2Vec showed that embeddings could be used to show relationships between words like capital cities and their corresponding countries | Source:* [*Semantic Scholar Paper*](https://web.archive.org/web/20230308023938/https://www.semanticscholar.org/paper/Distributed-Representations-of-Words-and-Phrases-Mikolov-Sutskever/87f40e6f3022adbc1f1905e3e506abad05a9964f)

通过对更大的文本序列(如句子)做类似的事情，这种方法通过文本生成的例子得到了扩展。这就是所谓的顺序对顺序方法。它扩展了深度学习架构的范围，以执行越来越复杂的 NLP 任务。

这里要解决的关键问题是语言是一个连续的单词流。一个句子没有标准长度，每个句子都不一样。通常，这些深度学习模型需要知道它们正在处理的数据序列的固定长度。但是，这在文本中是不可能的。

因此，序列对序列模型使用了一种称为递归神经网络(RNNs)的技术。有了它，这些架构可以执行“循环”,并连续处理文本。这使他们能够创建响应输入提示产生文本的 LMs。

![RNNs as looping output](img/566c53d2f4bdb6f644367dc6e0829871.png)

*An example of how to think of RNNs as “looping” output from one part of the network to the input of the next step in a continuous sequence | Source:* [*Chis Olah’s (amazing) post on RNNs*](https://web.archive.org/web/20230308023938/https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

尽管像 Word2Vec 这样的模型和 RNNs 这样的体系结构在 NLP 方面取得了令人难以置信的进步，但它们仍然有一些缺点。Word2Vec 嵌入是静态的——每个单词都有一个固定的嵌入，即使单词根据上下文有不同的含义。RNN 架构的训练速度很慢，这限制了它可以训练的数据量。

正如我们所提到的，每一个新的模型都可以被看作是在以前的基础上进行改进的尝试。ELMo 试图解决 Word2Vec 的静态单词嵌入的缺点，采用 RNN 方法来训练模型识别单词含义的动态性质。

这是通过尝试根据单词所在的句子，动态地给单词分配一个向量来实现的。ELMo 模型允许用户向模型中输入文本，并基于该句子生成嵌入，而不是像 Word2Vec 那样使用带有单词和嵌入的查找表。因此，它可以根据上下文为一个单词产生不同的含义。

![NLP ELMo](img/0054c9b3d22a20f99b70450ecd0a2478.png)

*ELMo uses two separate networks to try and process text “bidirectionally” | Source* [*Google AI Blog*](https://web.archive.org/web/20230308023938/https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

这里需要注意的另一个要点是，ELMo 是第一个尝试非顺序处理文本的模型。以前的模型如 Word2Vec 一次读取一个单词，并按顺序处理每个单词。埃尔莫试图复制人类阅读文本的方式，他用两种方式处理文本:

1.  **开始到结束**:架构的一部分正常读取文本，从开始到结束。
2.  **反过来，end to start** :架构的另一部分从后往前读文本。理想的情况是，模型可以通过阅读“未来”来学习其他东西。
3.  **组合**:在文本被阅读后，两个嵌入被连接起来“组合”意思。

这是一种双向阅读文本的尝试。虽然它不是“真正”双向的(它更像是一种反向的单向方法)，但它可以被描述为“浅”双向的。

在很短的时间内，我们已经看到了一些快速的进步，从 Word2Vec 到更复杂的神经网络架构，如用于文本生成的 RNNs，再到通过 ELMo 基于上下文的单词嵌入。然而，使用这些方法训练大量数据的限制仍然存在问题。这严重阻碍了这些模型提高在一系列 NLP 任务中表现良好的能力的潜力。这就是预训练的概念为伯特这样的模型的到来奠定了基础，以加速进化。

## 3.预训练的 NLP 模型

简单地说，如果没有预训练模型的出现，BERT(或任何其他基于 Transformer 的模型)的成功是不可能的。预训练模型的理想在深度学习中并不新鲜。在图像识别方面已经实践了很多年。

![NLP transfer learning](img/171203e85a6917a7b8004aed6954b49d.png)

*Training a model on a massive amount of data and then making it available pre-trained enabled innovation in machine vision |* [*Source*](https://web.archive.org/web/20230308023938/https://madhuramiah.medium.com/deep-learning-using-resnets-for-transfer-learning-d7f4799fa863)

ImageNet 是一个巨大的标签图像数据集。多年来，它一直是训练图像识别模型的基础。这些模型学会了从这些大型数据库中识别图像识别的关键方面。这是识别图像边界、边缘和线条以及常见形状和物体的能力。

这些通用的训练模型可以下载，并用于训练你自己的，小得多的数据集。假设你想训练它识别你公司的人的脸。你不需要从零开始，让模型了解图像识别的一切，而是只需建立一般训练的模型，并根据你的数据调整它们。

像 Word2Vec 这样的模型的问题是，虽然它们是在大量数据上训练的，但它们不是通用的语言模型。您可以根据您的数据从头开始训练 Word2Vec，或者您可以简单地使用 Word2Vec 作为网络的第一层来初始化您的参数，然后添加层来为您的特定任务训练模型。因此，您可以使用 Word2Vec 来处理您的输入，然后为您的情感分类或 POS 或 NER 任务设计您自己的模型层。这里的主要限制是，每个人都在训练自己的模型，很少有人有资源(无论是数据还是计算成本)来训练任何非常大的模型。

正如我们在第 2 节中提到的那样，直到 2018 年像 ELMo 这样的模型出现。在那段时间，我们看到其他模型，如 [ULMFit](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1801.06146.pdf) 和[Open AIs first transformer model](https://web.archive.org/web/20230308023938/https://ruder.io/nlp-imagenet/)，也创建预训练模型。

这就是领先的 NLP 研究人员 Sebastian Ruder 所说的 NLPs[ImageNet moment](https://web.archive.org/web/20230308023938/https://ruder.io/nlp-imagenet/)——NLP 研究人员开始在预训练模型的强大基础上构建新的更强大的 NLP 应用程序。他们不需要大量的资金或数据来做这件事，这些模型可以“开箱即用”。

这对于像 BERT 这样的模型至关重要的原因有两个:

1.  **数据集大小**:语言杂乱、复杂，对计算机来说比识别图像要难学得多。他们需要更多的数据来更好地识别语言模式，识别单词和短语之间的关系。像最新的 GPT-3 这样的模型是在 45TB 的数据上训练的，包含 1750 亿个参数。这些是巨大的数字，所以很少有人甚至组织有资源来训练这些类型的模型。如果每个人都必须训练自己的 BERT，如果研究人员不利用这些模型的力量，我们将会看到很少的进展。进展将是缓慢的，并且仅限于几个大玩家。
2.  **微调:**预先训练的模型具有双重优势，即它们可以“现成”使用，即无需任何更改，企业只需将 BERT 插入他们的管道，并与聊天机器人或其他应用程序一起使用。但这也意味着这些模型可以针对具体任务进行微调，而无需太多数据或模型调整。对于 BERT 来说，你所需要的只是几千个例子，你可以根据你的数据进行微调。预训练甚至使得像 GPT-3 这样的模型可以在如此多的数据上进行训练，以至于他们可以采用一种被称为[零或少量拍摄学习](https://web.archive.org/web/20230308023938/https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122)的技术。这意味着他们只需要看一些例子就能学会执行一项新任务，比如[编写一个计算机程序](https://web.archive.org/web/20230308023938/https://analyticsindiamag.com/open-ai-gpt-3-code-generator-app-building/)。

随着预训练模型的兴起，以及从 Word2Vec 到 ELMo 的训练和架构的进步，现在是 BERT 登场的时候了。在这一点上，我们知道我们需要一种方法来处理更多的数据并从这些数据中学习更多的上下文，然后使其在预训练的模型中可用，供其他人在他们自己的领域特定的应用中使用。

## 4.变压器架构

如果你从这篇文章中学到了什么，那就是对 Transformer 架构的总体理解，以及它与 BERT 和 GPT-3 等模型的关系。这将让您查看不同的 Transformer 模型，了解他们对 vanilla 架构做了哪些调整，并了解他们试图解决什么问题或任务。这提供了一个关键的洞察力，它可能更适合什么任务或领域。

据我们所知，最初的变形金刚论文叫做“[注意力是你所需要的全部](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1706.03762.pdf)”。名称本身很重要，因为它指出了它与以前的方法有什么不同。在第 2 节中，我们注意到 ELMo 等模型采用 RNNs 以类似循环的方式顺序处理文本。

![ELMo RNN](img/49df65ce14a91f8666d07dfa70ae9107.png)

*RNNs with sequence to sequence approaches processed text sequentially until they reached an end of sentence token (<eos>). In this example an request, “ABC” is mapped to a reply “WXYZ”. When the model receives the <eos> token the hidden state of the model stores the entire context of the preceding text sequence. Source: [A Neural Conversation Model](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/1506.05869.pdf)*

现在想一个简单的句子，比如“*狗在街上追着猫跑*”。对于一个人来说，这是一个容易理解的句子，但是如果你想按顺序处理它，实际上有很多困难。一旦你到了“ *it* 部分，你怎么知道它指的是什么？您可能需要存储一些状态来识别这个句子中的主角是“*猫*”。然后，当你继续阅读这个句子时，你必须找到某种方法将" *it* "和" *cat* "联系起来。

现在想象这个句子的长度可以是任意数量的单词，试着想想当你处理越来越多的文本时，你将如何跟踪被引用的内容。

这就是序列模型遇到的问题。

它们是有限的。他们只能优先考虑最近处理过的单词的重要性。随着他们继续沿着句子前进，前面单词的重要性或相关性开始减弱。

你可以把它想象成在处理每个新单词时向列表中添加信息。您处理的单词越多，就越难引用列表开头的单词。本质上，你需要一次一个元素，一个单词一个单词地往回移动，直到你找到更早的单词，然后看看那些实体是否相关。

“*它*是指“*猫*”吗？这就是所谓的“[消失梯度](https://web.archive.org/web/20230308023938/https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem)”问题，ELMo 使用了一种称为长短期记忆网络(LSTMs)的特殊网络来缓解这种现象的后果。LSTMs 确实解决了这个问题，但没有消除它。

最终，他们无法创造一种有效的方式来“聚焦”每个句子中的重要单词。这就是变压器网络通过使用我们已知的“注意”机制来解决的问题。

![Attention in Transformers](img/95c811dbefc6f65306909b2c75ce9d46.png)

*This gif is from a great blog post about understanding attention in Transformers. The green vectors at the bottom represent the encoded inputs, i.e. the input text encoded into a vector. The dark green vector at the top represents the output for input 1\. This process is repeated for each input to generate an output vector which has attention weights for the “importance” of each word in the input which are relevant to the current word being processed. It does this via a series of multiplication operations between the Key, Value and Query matrices which are derived from the inputs. Source: [Illustrated Self-Attention](https://web.archive.org/web/20230308023938/https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a).*

“注意力是你所需要的全部”这篇论文使用注意力来提高机器翻译的性能。他们创建了一个包含两个主要部分的模型:

1.  **编码器**:这部分“注意力是你所需要的全部”模型处理输入文本，寻找重要部分，并根据与句子中其他单词的相关性为每个单词创建嵌入。
2.  **解码器**:它接收编码器的输出，这是一个嵌入，然后将这个嵌入转换回文本输出，即输入文本的翻译版本。

然而，论文的关键部分不是编码器或解码器，而是用于创建它们的层。具体来说，编码器和解码器都不像传统的 RNNs 那样使用任何递归或循环。取而代之的是，他们使用“注意力”层，信息通过它线性传递。它没有多次循环输入——相反，转换器通过多个注意层传递输入。

你可以把每一个注意力层看作是“学习”更多的输入信息，也就是看着句子的不同部分，试图发现更多的语义或句法信息。这在我们之前提到的渐变消失问题中是很重要的。

随着句子长度的增加，rnn 处理它们和学习更多信息变得越来越困难。每一个新单词都意味着要存储更多的数据，并且更难检索这些数据来理解句子中的上下文。

![Attention_diagram_transformer](img/c94ef156bd017b01c5796468a6c8f966.png)

*This looks scary, and in truth it is a little overwhelming to understand how this works initially. So don’t worry about understanding it all right now. The main takeaway here is that instead of looping the Transformer uses scaled dot-product attention mechanisms multiple times in parallel, i.e. it adds more attention mechanisms and then processes input in each in parallel. This is similar to looping over a layer multiple times in an RNN. Source:* [*Another great post on attention*](https://web.archive.org/web/20230308023938/https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

变压器可以解决这个问题，只需添加更多的“注意头”，或层。由于没有循环，它不会遇到渐变消失的问题。转换器在处理更长的文本时仍然有问题，但它不同于 RNN 的问题，我们不需要在这里讨论。作为比较，最大的 BERT 模型由 24 个注意层组成。GPT 2 号有 12 个关注层，GPT 3 号有 96 个关注层。

在这里，我们不会详细讨论注意力是如何工作的。我们可以在另一个帖子中查看。同时，您可以查看上图中链接的博客文章。它们是关于注意力及其运作方式的极好资源。这篇文章的重点是要理解 Transformer 架构是如何消除我们在 NLP 中使用 RNNs 时遇到的许多问题的。我们前面提到的另一个问题是，在研究 RNNs 的局限性时，以无序方式处理文本的能力。通过注意力机制，转换器使这些模型能够精确地做到这一点，并双向处理文本。

## 5.双向性的重要性

我们前面提到过，rnn 是在 Transformer 之前用来处理文本的架构。rnn 使用递归或循环来处理文本输入序列。以这种方式处理文本会产生两个问题:

1.  速度慢:单向顺序处理文本的成本很高，因为这会造成瓶颈。这就像高峰时段的单车道，那里有长长的车龙，而非高峰时段的道路上几乎没有车。我们知道，一般来说，如果这些模型根据更多的数据进行训练，它们会表现得更好，所以如果我们想要更好的模型，这个瓶颈是一个大问题。
2.  **它错过了关键信息**:我们知道人类并不是以绝对纯顺序的方式阅读文本。正如心理学家丹尼尔·威林厄姆在他的书“[《阅读的头脑》](https://web.archive.org/web/20230308023938/https://www.amazon.co.uk/Reading-Mind-Cognitive-Approach-Understanding/dp/1119301378)”、“*我们不是一个字母一个字母地读，我们是一个字母一个字母一个字母地读，一次找出几个字母*”。原因是我们需要知道一点未来的事情来理解我们现在正在读的东西。NLP 语言模型也是如此。单向处理文本限制了他们从数据中学习的能力

![Bidirectionality](img/a2cc7330421e5c1064874d0076f049ac.png)

*This text went viral in 2003 to show that we can read text when it is out of order. While there is some controversy around this, see* [*here*](https://web.archive.org/web/20230308023938/https://www.mrc-cbu.cam.ac.uk/people/matt.davis/cmabridge/) *for more detail, it still shows that we do not read text strictly in a letter by letter format*

我们看到 ELMo 试图通过我们称之为“浅层”双向的方法来解决这个问题。它在一个方向上处理文本，然后反转文本，即从末尾开始，并以这种方式处理文本。通过连接这两种嵌入，希望这将有助于捕捉句子中的不同含义，如:

1.  鼠标放在桌子上，靠近*笔记本电脑*
2.  那只*老鼠*在桌子上，靠近那只*猫*

这两个句子中的“*鼠标*”指的是一个非常不同的实体，这取决于句子中的最后一个词是“*笔记本电脑*还是“*猫*”。通过颠倒句子，从单词“ *cat* ”开始，ELMo 试图学习上下文，以便能够对单词“ *mouse* ”的不同含义进行编码。通过首先处理单词“cat ”, ELMo 能够将不同的意思结合到“反向”嵌入中。这就是 ELMo 如何改进传统的、静态的 Word2Vec 嵌入的方法，后者只能对每个单词的一个含义进行编码。

如果不理解这一过程的细微差别，我们会发现 ELMo 方法并不理想。

我们希望有一种机制能够让模型在编码过程中查看句子中的其他单词，这样它就可以知道我们是否应该担心桌子上有一只鼠标！而这正是 Transformer 架构的注意力机制让这些模型能够做到的。

*这个来自* [*谷歌博客*](https://web.archive.org/web/20230308023938/https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) *的例子展示了变形金刚网络中的注意力机制如何将注意力“集中”在“它”所指的东西上，在这种情况下是街道，同时也认识到“动物”一词的重要性，即动物没有穿过它，因为街道太宽了。*

能够双向读取文本是像 BERT 这样的 Transformer 模型能够在传统的 NLP 任务中取得如此令人印象深刻的结果的关键原因之一。正如我们从上面的例子中看到的，当你只从一个方向阅读文本时，要知道“它”指的是什么是困难的，并且必须顺序存储所有的状态。

我想这并不奇怪，这是 BERT 的一个关键特性，因为 BERT 中的 B 代表“双向”。Transformer 架构的注意机制允许像 BERT 这样的模型通过以下方式双向处理文本:

1.  **允许并行处理**:基于 Transformer 的模型可以并行处理文本，因此它们不会像基于 RNN 的模型那样受到必须顺序处理文本的瓶颈的限制。这意味着，在任何时候，模型都能够查看它正在处理的句子中的任何单词。但是这引入了其他问题。如果你在并行处理所有的文本，你怎么知道原文中单词的顺序呢？这是至关重要的。如果我们不知道顺序，我们就只有一个单词袋类型的模型，无法从句子中完全提取意思和上下文。
2.  **存储输入的位置**:为了解决排序问题，Transformer 架构将单词的位置直接编码到嵌入中。这是一个“标记”，让模型中的注意力层识别他们正在查看的单词或文本序列位于何处。这个漂亮的小技巧意味着这些模型可以并行处理大量不同长度的文本序列，并且仍然知道它们在句子中出现的顺序。
3.  **使查找变得容易**:我们之前提到过，RNN 类型模型的一个问题是，当它们需要顺序处理文本时，这使得检索早期的单词变得困难。因此，在我们的“鼠标”例句中，RNN 人希望理解句子中最后一个词的相关性，即“笔记本电脑”或“猫”，以及它与句子的前一部分的关系。要做到这一点，它必须从 N-1 个单词，到 N-2，到 N-3 等等，直到它到达句子的开头。这使得查找变得困难，这就是为什么单向模型很难发现上下文的原因。相比之下，基于 Transformer 的模型可以在任何时候简单地查找句子中的任何单词。这样，它在注意力层的每一步都有一个序列中所有单词的“视图”。所以它在处理句子的前半部分时可以“向前看”到句子的末尾，反之亦然。(根据注意力层的实现方式，这有一些细微差别，例如，编码器可以查看任何单词的位置，而解码器仅限于“回顾”它们已经处理过的单词。但是我们现在不需要担心这个)。

由于这些因素，能够并行处理文本，在嵌入中嵌入输入的位置，并且能够方便地查找每个输入，像 BERT 这样的模型可以双向“读取”文本。

从技术上来说，它不是双向的，因为这些模型实际上是一次查看所有文本，所以它是非定向的。但是最好把它理解为一种尝试双向处理文本的方法，以提高模型从输入中学习的能力。

能够以这种方式处理文本带来了一些问题，BERT 需要用一种叫做“屏蔽”的巧妙技术来解决，我们将在第 8 节中讨论。但是，现在我们对 Transformer 架构有了更多的了解，我们可以看看在最初的“注意力是你所需要的”论文中 BERT 和普通 Transformer 架构之间的区别。

## 6.伯特和变形金刚有什么不同？

当你读到最新的模型时，你会看到它们被称为“变形金刚”模型。有时这个词会被不严格地使用，像伯特和 GPT-3 这样的模型都会被称为“变形金刚”模型。但是，这些模型在一些重要方面非常不同。

*如“注意力是你所需要的”一文中所述的变压器网络。注意，左边是编码器，右边是解码器，这就是我们的网络。|来源:* [*关注是你所需要的*](https://web.archive.org/web/20230308023938/https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776)

理解这些差异将有助于您了解针对您自己独特的用例使用哪种模型。理解不同模型的关键是了解它们如何以及为什么偏离最初的 Transformer 架构。一般来说，需要注意的主要事项有:

1.  **是否使用了编码器？**最初的 Transformer 架构需要翻译文本，因此它以两种不同的方式使用注意机制。一个是对源语言进行编码，另一个是将编码后的嵌入解码回目标语言。查看新型号时，检查它是否使用了编码器。这意味着它涉及以某种方式使用输出来执行另一项任务，即作为另一层的输入来训练分类器，或类似的事情。
2.  **解码器用了吗？**或者，一个模型可能不使用编码器部分，而只使用解码器。解码器实现的注意机制与编码器略有不同。它的工作方式更像传统的语言模型，在处理文本时只查看以前的单词。这将适用于语言生成等任务，这就是为什么 GPT 模型使用转换器的解码器部分，因为它们主要关心的是生成文本以响应文本的输入序列。
3.  **增加了哪些新的训练层？**最后要看的是模型为了执行训练增加了哪些额外的层(如果有的话)。正如我们前面提到的，注意力机制通过平行和双向处理文本打开了一系列的可能性。不同的层可以在此基础上构建，并为不同的任务训练模型，如问答或文本摘要。

![bert_encoder](img/3b58733ea15eef3f51bad1c2507abd6c.png)

*BERT only uses the encoder part of the original Transformer network*

现在我们知道要找什么了，BERT 和普通的变形金刚有什么不同呢？

1.  **BERT 使用编码器** : BERT 使用转换器的编码器部分，因为它的目标是创建一个执行许多不同 NLP 任务的模型。因此，使用编码器使 BERT 能够在嵌入中对语义和语法信息进行编码，这是许多任务所需要的。这已经告诉了我们很多关于伯特的事情。首先，它不是为文本生成或翻译等任务设计的，因为它使用编码器。它可以在多种语言上训练，但它本身不是一个机器翻译模型。同样，它仍然可以预测单词，所以它可以用作文本生成模型，但这不是它优化的目的。
2.  **BERT 不使用解码器**:如前所述，BERT 不使用普通变压器架构的解码器部分。所以，BERT 的输出是嵌入的，而不是文本输出。这很重要——如果输出是嵌入的，这意味着无论你用 BERT 做什么，你都需要做一些嵌入的事情。例如，您可以使用余弦相似性等技术来比较嵌入并返回相似性得分。相比之下，如果您使用解码器，输出将是一个文本，所以您可以直接使用它，而不需要执行任何进一步的行动。
3.  **BERT 使用创新的训练层:** BERT 采用编码器的输出，并将其用于执行两种创新训练技术的训练层，即掩蔽和下一句预测(NSP)。这些是解锁包含在 BERT 嵌入中的信息的方法，以使模型从输入中学习更多信息。我们将在第 8 节详细讨论这些技术，但要点是 BERT 让 Transformer 编码器尝试并预测隐藏或屏蔽的单词。通过这样做，它迫使编码器尝试并“学习”更多关于周围文本的信息，并能够更好地预测隐藏或“屏蔽”的单词。然后，对于第二种训练技术，它让编码器在给定前一个句子的情况下预测整个句子。BERT 引入了这些“调整”来利用转换器，特别是注意力机制，并创建了一个为一系列 NLP 任务生成 SOTA 结果的模型。在当时，它超越了以前做过的任何事情。

现在我们知道了 BERT 与普通变压器架构的不同之处，我们可以仔细看看 BERT 模型的这些部分。但首先，我们需要了解伯特是如何“阅读”文本的。

## 7.记号赋予者:伯特如何阅读

当我们考虑像 BERT 这样的模型时，我们经常忽略这个过程的一个重要部分:这些模型如何“读取”输入，以便能够从他们接受训练的大量文本中学习？

你可能认为这是容易的部分。一次处理一个单词，用一个空格把每个单词分开，然后把它传递给注意力层，让他们施展魔法。

![Tokenizers](img/f07f1684fa03dadf99ffccd5a3076511.png)

*Tokenization seems straightforward, it just breaks the sentence up into words. But it turns out this is not as easy as it seems. | Source:* [*FloydHub*](https://web.archive.org/web/20230308023938/https://blog.floydhub.com/tokenization-nlp/)

然而，当我们尝试通过单词或其他简单方法(如标点符号)来标记文本时，会出现一些问题，例如:

1.  一些语言不通过空格分隔单词:使用单词级方法意味着该模型不能用于像中文这样的语言，在这些语言中单词分隔不是一项简单的任务。
2.  你将需要大量的词汇:如果我们按单词来划分事物，我们需要为我们可能遇到的每一个可能的单词做相应的嵌入。这是个很大的数字。你怎么知道你在训练数据集中看到了每一个可能的单词？如果你没有，模型将无法处理一个新单词。这是过去发生的事情，当遇到一个未知单词时，模特们被迫向< UNK >令牌进行识别。
3.  更大的词汇量会降低你的模型的速度:记住，我们需要处理更多的数据，让我们的模型学习更多的语言知识，更好地完成自然语言处理任务。这是 Transformer 的主要好处之一——我们可以处理比以前任何模型都多得多的文本，这有助于使这些模型变得更好。然而，如果我们使用单词级标记，我们需要大量的词汇，这增加了模型的大小，并限制了它对更多文本进行训练的能力。

![Huggingface library](img/f6277ecb6f4a878c6e1f11ab6dca7d51.png)

HuggingFace 有一个很棒的[标记器库](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/main_classes/tokenizer.html)。它包括关于不同方法如何工作的[优秀范例教程](https://web.archive.org/web/20230308023938/https://huggingface.co/transformers/tokenizer_summary.html)。例如，这里显示了如何根据训练数据中单词的频率创建基本词汇。这显示了像“hug”这样的单词是如何被标记为“hug”的，而“pug”是由两个子单词部分“p”和“ug”标记的。

伯特是如何解决这些问题的？它实现了一种新的标记化方法， [WordPiece](https://web.archive.org/web/20230308023938/https://paperswithcode.com/method/wordpiece) ，它将子词方法应用于标记化。WordPiece 通过以下方式解决了许多以前与标记化相关的问题:

1.  **使用子词代替单词**:wordpartie 不是查看整个单词，而是将单词分解成更小的部分，或者构建单词块，这样它就可以使用它们来构建不同的单词。比如想一个“学”这样的词。你可以把它分成三个部分，“lea”，“rn”和“ing”。这样，你就可以用积木造出各种各样的单词:
    *   学习= lea + rn + ing
    *   lean = lea+rn

然后，您可以将它们与其他子单词单元组合在一起，组成其他单词:

*   Burn = bu + rn
*   流失= chu +rn
*   Turning = tur + rrn + ing
*   圈数= tu + rn + s

您可以为最常用的词创建完整的词，但为不经常出现的其他词创建子词。然后，您可以确信您将通过使用您创建的构建块来标记任何单词。如果你遇到一个全新的单词，你总是可以一个字符一个字符地把它拼凑起来，例如 LOL = l + o + l，因为子单词标记库也包括每个单独的字符。即使你从未见过，你也可以造出任何单词。但是一般来说，你应该有一些可以使用的子词单元，并根据需要添加一些字符。

2.  创建一个小词汇量:因为我们不需要为每个单词准备一个令牌，我们可以创建一个相对较小的词汇量。例如，BERT 在其库中使用了大约 30，000 个令牌。这可能看起来很大，但是想想曾经发明和使用过的每一个可能的单词，以及它们所有不同的用法。要涵盖这一点，您需要包含数百万个标记的词汇表，但您仍然无法涵盖所有内容。能够用 30，000 个代币做到这一点是不可思议的。这也意味着我们不需要使用< UNK >令牌，仍然有一个小的模型大小和大量的数据训练。
3.  **它仍然假设单词由空格分隔，但是…** 虽然 wordpartie 确实假设单词由空格分隔，但新的子单词令牌库，如[sentepiece](https://web.archive.org/web/20230308023938/https://github.com/google/sentencepiece)(与基于 Transformer 的模型一起使用)或[多语言通用句子编码器](https://web.archive.org/web/20230308023938/https://arxiv.org/abs/1907.04307) (MUSE)，是这种子单词方法的进一步增强，也是新模型最常用的库。

如果你想更深入地了解这个主题，我在这里写了一篇关于记号赋予者的深入评论。

## 8.掩盖:努力工作对聪明工作

正如我们在第 6 节中提到的，当你在看不同的基于 Transformer 的模型时，看看它们在训练方法上的不同会很有趣。就 BERT 而言，培训方法是最具创新性的方面之一。Transformer 模型提供了足够的改进，你可以使用传统的语言模型方法训练它们，并看到巨大的好处。

![Transformer models challenge](img/24240d1e0991413f38d04466d11dbfe4.png)

*One problem with BERT and other Transformer based models which use the encoder is that they have access to all the words at input time. So asking it to predict the next words is too easy. It can “cheat” and just look it up. This was not an issue with RNNs based models which could only see the current word and not the next one, so they couldn’t “cheat” this way. Source:* [*Stanford NLP*](https://web.archive.org/web/20230308023938/https://nlp.stanford.edu/seminar/details/jdevlin.pdf)

这是从以前的标记预测未来标记的地方。以前的 rnn 使用自回归技术来训练它们的模型。GPT 模型同样使用自回归方法来训练它们的模型。除了变压器架构(即我们之前提到的解码器)，他们可以训练比以往更多的数据。而且，正如我们现在所知，模型可以通过注意力机制和双向处理输入的能力更好地学习上下文。

![BERT masking](img/e9a8f1c2653a1c7a4a10d7c107223c9c.png)

*BERT uses a technique called masking to prevent the model from “cheating” and looking ahead at the words it needs to predict. Now it never knows whether the word it’s actually looking at is the real word or not, so it forces it to learn the context of all the words in the input, not just the words being predicted.*

相反，BERT 使用了一种创新技术，试图“强迫”模型从给定的数据中学习更多。这项技术还提出了许多关于深度学习模型如何与给定的训练技术进行交互的有趣方面:

1.  **为什么需要掩蔽？**记住，转换器允许模型双向处理文本。本质上，模型可以同时看到输入序列中的所有单词。以前，对于 RNN 模型，他们只看到输入中当前的工作，不知道下一个单词是什么。让这些模型预测下一个单词很容易，模型不知道它，所以必须尝试预测它，并从中学习。然而，有了 Transformer，BERT 可以“欺骗”并查看下一个单词，因此它什么也学不到。这就像参加一个考试，并得到答案。如果你知道答案永远在那里，你就不会学习。伯特使用掩蔽来解决这个问题。
2.  **什么是掩蔽？**掩蔽([也称为完形填空](https://web.archive.org/web/20230308023938/https://en.wikipedia.org/wiki/Cloze_test))简单地说就是我们隐藏或“掩蔽”一个单词，然后强迫模型预测这个单词，而不是预测下一个单词。对于 BERT 来说，15%的输入令牌在模型看到之前就被屏蔽了，所以它不可能作弊。为此，随机选择一个单词，简单地用“[MASK]”标记替换，然后输入到模型中。
3.  **不需要标签**:对于这些任务，要记住的另一件事是，如果你正在设计一种新的训练技术，最好是你不需要为训练标记或手动构造数据。屏蔽通过要求一种简单的方法来实现对大量非结构化数据的训练来实现这一点。这意味着它可以在完全无人监督的情况下进行训练。
4.  15%的 80%:虽然蒙版看起来是一个简单的技术，但它有很多细微的差别。在被选择用于屏蔽的 15%的令牌中，只有 80%实际上被替换为屏蔽令牌。取而代之的是，10%用一个随机的单词替换，10%用正确的单词替换。
5.  **蒙版**有什么好玩的？为什么不直接屏蔽 100%选择的 15%输入？这是个有趣的问题。如果你这样做，模型将知道它只需要预测被屏蔽的单词，而不需要了解输入中的其他单词。不太好。你需要模型去了解所有的输入，而不仅仅是 15%的屏蔽输入。为了“强迫”模型学习非屏蔽单词的上下文，我们需要用随机单词替换一些标记，用正确的单词替换一些标记。这意味着 BERT 永远不知道它被要求预测的非屏蔽单词是否是正确的单词。作为一种替代方法，如果我们在 90%的时间里使用掩码标记，然后在 10%的时间里使用不正确的单词，BERT 会在预测非掩码标记时知道它总是错误的单词。类似地，如果我们只在 10%的时间里使用正确的单词，BERT 会知道它总是正确的，所以它会继续重用它为那个单词学习的静态单词。它永远也学不会这个单词的上下文嵌入。这是对 Transformer 模型如何在内部表示这些状态的一个有趣的洞察。毫无疑问，我们会在其他模型中看到更多类似的训练调整。

## 9.伯特:微调和迁移学习

现在应该很清楚了，当它在几年前出版时，BERT 在很多方面打破了这个模式。另一个区别是能够适应特定的领域。这建立在我们已经讨论过的许多事情的基础上，例如作为一个预先训练的模型，这意味着人们不需要访问大型数据集来从头训练它。您可以构建从更大的数据集中学习的模型，并将该知识“转移”到特定任务或领域的模型中。

![BERT transfer learning](img/ad2170635af1cc3218c5cda1870e2ae7.png)

*In Transfer Learning we take knowledge learned in one setting, usually via a very large dataset, and apply it to another domain where, generally, we have much less data available. | Source* [*Sebastian Ruder blog post*](https://web.archive.org/web/20230308023938/https://ruder.io/state-of-transfer-learning-in-nlp/)

BERT 是一个预先训练好的模型，这意味着我们可以根据我们的领域对它进行微调，因为:

1.  **BERT 可以进行迁移学习**:迁移学习是一个 power 概念，最早是为机器视觉实现的。然后，在 ImageNet 上训练的模型可用于其他“下游”任务，在这些任务中，它们可以基于在更多数据上训练的模型的知识。换句话说，这些预先训练的模型可以将它们在大型数据集上学习的知识“转移”到另一个模型，后者需要更少的数据来出色地完成特定任务。对于机器视觉，预先训练的模型知道如何识别图像的一般方面，如线条、边缘、面部轮廓、图片中的不同对象等等。他们不知道像个人面部差异这样的细微细节。一个小模型可以很容易地被训练来将这种知识转移到它的任务中，并识别特定于它的任务的人脸或物体。如果你想让[识别一株患病的植物](https://web.archive.org/web/20230308023938/https://heartbeat.fritz.ai/plantvillage-helping-farmers-in-east-africa-identify-and-treat-plant-disease-9a26b167b400)，你不需要从头开始。
2.  **你可以选择相关的层来调优**:虽然这仍然是[正在进行的研究](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/2002.12327.pdf)的问题，但似乎 BERT 模型的较高层学习更多的上下文或语义知识，而较低层往往在句法相关的任务上表现更好。较高层通常与特定任务的知识更相关。对于微调，您可以在 BERT 的基础上添加自己的层，并使用少量的数据来训练一些任务，如分类。在这些情况下，您可以冻结后面层的参数，只允许您添加的层参数发生变化。或者，您可以“解冻”这些更高层，并通过更改这些值来微调 BERT。
3.  伯特需要更少的数据:既然伯特似乎已经学习了一些关于语言的“一般”知识，你需要更少的数据来微调它。这意味着你可以使用带标签的数据来训练分类器，但你需要标记的数据要少得多，或者你可以使用 BERT 的原始训练技术，如 NSP，来训练它对未标记的数据。如果你有 3000 个句子对，你可以用你的未标记数据来微调 BERT。

所有这些意味着 BERT 和许多其他 Transformer 模型可以很容易地适应您的业务领域。虽然微调似乎确实存在一些问题，而且许多模型已经从零开始为独特的领域训练 BERT，如 [Covid19 信息检索](https://web.archive.org/web/20230308023938/https://www.aclweb.org/anthology/2020.coling-main.59.pdf)，但这些模型的设计仍然有很大的转变，它们能够在无人监督和相对少量的数据上进行训练。

## 10.鳄梨椅:伯特和变形建筑的下一步是什么

![Avocado chairs](img/bc0ebae5346813d150eb81a5f0343dd0.png)

*Avocado chairs designed by a decoder based Transformer model which was trained with images as well as text. | Source:* [*OpenAI*](https://web.archive.org/web/20230308023938/https://openai.com/blog/dall-e/)

现在，我们已经结束了对 BERT 和 Transformer 架构的旋风式回顾，让我们期待一下深度学习这一令人兴奋的领域的未来:

1.  关于这些模型局限性的问题:围绕着试图仅从文本中提取意义的潜在局限性，有一场引人入胜、近乎哲学的讨论。一些[最近发表的论文](https://web.archive.org/web/20230308023938/https://arxiv.org/pdf/2004.10151.pdf)展示了关于深度学习模型可以从语言中学习什么的一种新的思维形式。在对越来越多的数据进行训练方面，这些模型注定会达到收益递减的点吗？这些都是令人兴奋的问题，它们正在超越自然语言处理的界限，进入更广泛的通用人工智能领域。关于这些模型的更多限制，请看我们之前关于这个话题的 [neptune.ai 帖子](/web/20230308023938/https://neptune.ai/blog/ai-limits-can-deep-learning-models-like-bert-ever-understand-language)。
2.  模型的可解释性:人们常说我们开始使用一项技术，然后才弄清楚它是如何工作的。例如，我们在真正理解湍流和空气动力学的一切之前就学会了驾驶飞机。同样，我们使用这些模型，我们真的不明白他们在学习什么或如何学习。对于这些工作，我强烈推荐[莉娜·沃伊塔的博客](https://web.archive.org/web/20230308023938/https://lena-voita.github.io/)。她对伯特、GPT 和其他变压器模型的研究令人难以置信，是我们开始逆向工程这些模型如何工作的越来越多的工作的一个例子。
3.  **一幅图像胜过千言万语**:变形金刚模型的一个有趣的潜在发展是[已经在进行一些工作](https://web.archive.org/web/20230308023938/https://openai.com/blog/dall-e/)将文本和视觉结合起来，使模型能够根据句子提示生成图像。类似地，卷积神经网络(CNN)一直是机器视觉的核心技术，直到最近，基于[变压器的模型](https://web.archive.org/web/20230308023938/https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)开始被用于该领域。这是一个令人兴奋的发展，因为视觉和文本的结合有可能改善这些模型的性能，而不是单独使用文本。正如我们前面提到的，最初的机器翻译模型已经演变成一种技术，似乎比人工智能的任何其他领域都更接近于实现一般智能的圣杯。仔细观察这个空间，它可能是目前人工智能最令人兴奋的领域。

## 结论

仅此而已。如果你做到了这一步，非常感谢你的阅读！我希望这篇文章能帮助您理解 BERT 和 Transformer 架构。

这确实是目前人工智能中最有趣的领域之一，所以我鼓励你继续探索和学习。您可以从我在本文中留下的各种不同的链接开始。

祝你的人工智能之旅好运！*