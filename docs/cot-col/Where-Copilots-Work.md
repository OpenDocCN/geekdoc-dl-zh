<!--yml

类别：COT 专栏

日期：2024-05-08 11:10:56

-->

# 共同开发者的工作场所

> 来源：[`every.to/chain-of-thought/where-copilots-work`](https://every.to/chain-of-thought/where-copilots-work)

#### 由 Lever 赞助

使用[Lever](https://www.lever.co/demo/?utm_medium=third_party_email&utm_source=other&utm_campaign=none&utm_content=every_divinations_sponsored_newsletter)招聘更聪明——提供现代人才招聘领导者完整的 ATS 和强大的 CRM 功能的唯一完整招聘解决方案：LeverTRM

卢克·天行者有 R2-D2 的吹哨声和蜂鸣声。飞行员有古斯。伯蒂有他的管家 Jeeves，他会在甚至没有被要求之前就在房间里闪现来执行任务。

这些故事之所以受欢迎，是因为每个人都想要一个共同开发者——一个能让你变得更好的伙伴，以及（有时）成为在困难时可以依靠的朋友。

这种事情正是现在很多人工智能领域的人正在构建的。

GitHub 共同开发者是第一个有显著影响力的大规模人工智能用例，据说为使用它的开发者编写了[40%的代码](https://github.blog/2022-06-21-github-copilot-is-generally-available-to-all-developers/)。里德·霍夫曼认为，每个职业都会有一个[共同开发者](https://fortune.com/2022/12/07/linkedin-founder-reid-hoffman-on-ai-human-amplification/)。微软正在将一个[人工智能共同开发者集成到 Office 中](https://techcrunch.com/2023/03/06/microsoft-dynamics-copilot/)。Diagram 正在为设计师构建一个[共同开发者](https://genius.design/)。名单还在继续。

这些系统就像是超级强大的自动补全。它们预测你即将要做的事情，然后在你自己做之前就将其提供给你。这样能节省时间和精力。

如果你是一个在做副业项目的开发者，你可能正在考虑也要构建一个共同开发者。GPT-3 使这种事情在周末能够相当容易地实现。我知道因为我也一直在做：

我为自己建了一个小共同开发者。我希望它能帮助我变得更聪明：在我写作时建立思维之间的联系，为我所做的观点提供支持性证据的片段，并建议我可以使用的引用。

它接收任何文本块，然后尝试使用我在[Readwise](https://readwise.io/)数据库中找到的引用来完成这个文本块。

这是一个很酷的演示，但它离成为一个可用产品还有很长一段距离。它并不总是为我提供很好的引用，也不总是以实际支持我要表达的观点的方式来完成它们。它也没有展示足够的理解我的写作，或者它正在提取的作者的写作，以便有用。

正如我在[组织的终结](https://every.to/chain-of-thought/the-end-of-organizing)中所写的，对于这类技术的未来，我感到非常乐观。我发现自己读的纸质书越来越少，记笔记的次数也减少了。我越来越确信，未来一年左右，我所做的每个数字标记都将因这些工具而变得更加有用。

对我和其他像我一样的开发者而言，问题是：在当今的技术下，这类副驾驶体验*实际上*能够为哪些领域带来价值？以及需要解决哪些瓶颈，以使其对更多使用案例更有用？

让我们一一来看这些事项。

[Lever](https://www.lever.co/demo/?utm_medium=third_party_email&utm_source=other&utm_campaign=none&utm_content=every_divinations_sponsored_newsletter)是首屈一指的人才获取套件，它让人才团队轻松达成招聘目标，并将公司与顶尖人才联系在一起。

利用 LeverTRM，人才领导者可以扩大和培养他们的人才储备，建立真诚持久的关系，并找到合适的人才来招聘。而且，多亏了[LeverTRM](https://www.lever.co/demo/?utm_medium=third_party_email&utm_source=other&utm_campaign=none&utm_content=every_divinations_sponsored_newsletter)的分析功能，你可以得到定制报告与数据可视化、完成的招聘报价、面试反馈等等——从而可以做出更好、更明智、更战略的招聘决策。

## 如今可以构建副驾驶的领域

基于 AI 的副驾驶在当前情况下，使用现成的技术在小段轻微转换的样板文本中提供了大量的价值。

特别是在以下领域：

1.  文本可以被快速准确地检查，用户几乎不需要付出太多努力

1.  不准确所带来的成本很低

1.  可以可靠地利用[嵌入搜索](https://platform.openai.com/docs/guides/embeddings)找到相关的文本。

GitHub CoPilot 是一个很好的例子。但其他例子包括像[撰写拨款申请](https://grantable.co/)、[合同撰写](https://motionize.io/)、税收准备、许多类型的[电子邮件回复](https://chrome.google.com/webstore/detail/ellie-your-ai-email-assis/mhcnlcilgicfodlpjcacgglchmpoojcp)、[RFP 响应](https://www.userogue.com/)、向医生提供医疗建议等。

如果你正在构建一个副驾驶（或者考虑构建一个），我准备了一个小清单供你查看，以便确定是否有可能利用当今的技术获得良好的结果。

## 你能为此构建一个副驾驶吗？一个清单。

如果你想利用当今的技术为特定领域构建一个副驾驶，这里是你需要考虑的事项清单：

1.  是否有一套相关的文本补全语料库可供这个副驾驶使用？

1.  可以在这个文本语料库中可靠地通过嵌入搜索找到相关的文本补全吗？

1.  这些文本片段，无需更多上下文，能够轻松转换并作为准确的完成插入吗？

1.  完成可以在几乎不费吹灰之力的情况下检查其准确性吗？

#### **有一个相关文本完成的语料库可以供这个合作伙伴使用吗？**

你希望你的合作伙伴聪明，不要凭空想象。它应该能够访问一些知识来源，在用户需要时将其带给用户。理想情况下，这个知识来源应该是准确的、最新的，甚至可能是用户个人的，例如，它可能包括所有他们的电子邮件或他们公司的内部维基。

如果你有这个，你就可以进行下一步了。

#### **可以可靠地通过这个文本语料库进行嵌入搜索以找到相关的文本完成吗？**

一旦你有了合作伙伴使用的知识库，你需要合作伙伴能够准确地识别这个知识库的片段，当用户需要时将其返回给用户。例如，在我的思考合作伙伴演示中，我需要我的合作伙伴找到我的 Readwise 中与我当前所写内容相关的引用。

这样做的标准方法是使用嵌入搜索。嵌入是文本的一种紧凑的数学表示。就像纬度和经度可以帮助你确定地图上两个城市的距离一样，嵌入也可以为文本片段做同样的事情。如果你想知道两个文本片段是否相似，就计算它们的嵌入并进行比较。嵌入在“接近”一起的文本片段是相似的。

嵌入很有用，因为当用户正在输入合作伙伴想要自动完成的内容时，它只需浏览其知识库，找到与用户正在输入的内容“接近”的文本片段。

但是嵌入不是完美的，这也是目前许多合作伙伴使用案例失败的地方。你的合作伙伴质量将受限于你在知识库中找到相关信息的能力，以帮助用户。如果你没有得到相关的结果，完成的准确性将受到影响。

如果你*能够*获得相关的结果，那么你可以进行下一步了。

#### **这些文本片段，无需更多上下文，能够轻松转换并作为准确的完成插入吗？**

一旦你可以从嵌入搜索中找到知识库中最相关的信息，你的合作伙伴就需要以智能的方式将它们打包成用户的完成。

这种方式在它们只需要在建议之前稍微转换时效果最佳。例如，你经常希望重新排列文本，使其传达相同的信息，但重新措辞以完成用户的句子。这种转换在 GPT-3 中很容易做到，但更高级的转换则更难做到。

#### **完成可以在几乎不费吹灰之力的情况下检查其准确性吗？**

一旦你的合伙人提出一个完成方案，最好让用户知道这个完成方案是否准确，而不需要花费太多功夫。如果用户必须花费大量时间来判断完成方案是否准确，他们就会忽视它。

这是合伙人的一个重要杠杆之一。如果您能够轻松检查一个完成方案而不需要花费太多精力，那么您的合伙人可以返回许多错误答案，因为用户不需要花费太多精力来考虑它们。我认为这是 GitHub Copilot 成功的一部分：您只需运行代码来查看它是否正确，因此计算机生成代码然后为您检查它。

需要更多用户输入的其他用例将需要相应更高的准确率，以使用户感到有动力进行检查。

## 什么可能改变这个列表？

合伙人的限制是人工智能的上下文窗口的限制。上下文窗口是您可以将标记馈送到提示中的数量，以及它可以在完成中返回的标记数量。

因为上下文窗口是有限的，所以您必须使用嵌入搜索来找到可以馈送给人工智能以生成合伙人完成的小片段信息。这意味着，在上下文窗口仍然很小的情况下，您的合伙人的质量受到嵌入搜索质量的限制。

GPT-3 当前的上下文窗口是 4,096 个标记，大约是 3,000 个单词。有传言说 OpenAI 即将发布一个上下文窗口为 32K 标记的模型版本——大约是当前尺寸的 8 倍。我认为，这将是对合伙人使用情况返回的响应质量的巨大改变。

您将能够返回更多信息以供人工智能推理，并将其转换为可用的响应，这将直接影响准确性。

这里的另一个重要限制因素是推理成本、推理速度、嵌入成本和可用数据的访问。我预计成本会下降，速度会显著提高，以至于我不担心它们成为真正的瓶颈。但是可用数据的访问是一件大事。

现在，我正在使用 Readwise 作为我的数据来源。但是，如果我的合伙人可以访问我提取的书籍，我的完成方案会好得多。一本书中的平均标记数量约为 80,000 个标记。因此，为了提高我的响应质量，我需要想办法让这些数据对人工智能可用，并且对其进行清理，以便它能够找到相关的段落。

## 建议给建造者

如果您正在构建或投资于这个领域，我对创建更好的合伙人体验的建议如下：

#### **加紧您的反馈循环**

你可以把一个合伙人完成视为一个连续的链条：

1.  获取用户输入

1.  查询相关文档

1.  使用文档提示模型

1.  返回一个结果

当你在开发助手体验时，你希望尽可能快地迭代链条中的每一部分，并且尽可能少的代码。我建议建立工具来帮助你快速完成这些工作。

当我建立我的笔记助手时，我建立了一个小的用户界面来可视化并快速更换链条的每一部分：

这对我来说效果很好，但你应该探索自己的解决方案。

#### **发挥创意进行嵌入搜索**

目前，你的完成质量受限于你的嵌入式搜索的质量。因此，我建议花时间专注于提高您的嵌入式搜索的质量。

有许多方法可以增强嵌入式搜索，以帮助您获取更相关的文档。例如，请查看[HyDE](https://github.com/texttron/hyde)以从查询方面解决此问题的创意解决方案。或者，尝试使用 GPT-3 对您的知识库中的数据进行总结，以使嵌入能够找到可用的文本片段更容易。

#### **降低检查准确性的成本**

在这里，创建与现有技术一致的良好体验的另一个重要杠杆是降低用户的成本，如果完成的准确性低。一个简单的方法：在向用户显示任何内容之前，使用 GPT-3 检查它是否认为完成是好的。如果不是，那就不要显示它。

但是还有很多其他方法可以做到这一点。例如，请确保完成的内容非常简短。另一个例子：确保用户需要检查准确性的所有上下文信息都包含在完成中，这样他们就不必进行研究或思考太多。

## 总结

在我看来，这是理想的助手：

每次您触摸键盘时，它都会调用您的整个笔记存档和您曾经阅读过的所有内容，以帮助您完成下一句话。

它将帮助你在思想之间建立联系，提出支持性证据的片段，并建议使用的引用。它还可能提出你喜欢的作家，他们对你正在提出的观点持不同意见——这样你可以改变主意，或者根据他们的反驳来锋利地调整你的论点。

理想情况下，它会以一种无缝、高度准确且易于检查的方式完成这些工作。换句话说，通常如果它完成了某事，它是在表达一个好的观点，并且你很容易判断观点是否好，而无需付出太多额外的努力。

目前的情况远非如此。如果我们想要将这些工具推进到不仅仅是有趣的演示，我们必须自己来建立它们。

我希望这篇文章能推动一些人朝着这个方向前进。随着我不断发现更多内容，我会继续通知你们。