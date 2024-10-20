# 支持跨浏览器的语音到文本的语音搜索

> 原文：<https://www.algolia.com/blog/product/supporting-speech-to-text-across-browsers-for-voice-search/>

直到今天，如果你想在桌面和移动的浏览器中使用语音到文本，你要么被限制在提供原生支持的浏览器中(目前只有 Chrome 和 Edge，加上 Firefox)，要么你必须投入大量的前期时间来实现第三方语音到文本提供商。然而，今天，我们很兴奋地宣布，我们在 [AssemblyAI](https://www.assemblyai.com/algolia-voice-search) 的朋友已经开发了一个用于语音识别的 [InstantSearch.js 小部件](https://github.com/AssemblyAI/assemblyai-algolia-voice-widget)，它将 [InstantSearch](https://www.algolia.com/doc/guides/building-search-ui/what-is-instantsearch/js/) 以易于实现而闻名的特性与他们自己的高质量语音转文本功能结合在一起。

## [](#a-little-bit-of-history-on-instantsearch-and-voice-search)关于即时搜索和语音搜索的一点历史

当我们的客户找到我们，询问他们的客户如何进行免提搜索时，我们发布了针对 [Android](https://github.com/algolia/voice-overlay-android/) 和 [iOS](https://github.com/algolia/voice-overlay-ios/) 的语音覆盖，随后是一个 [IntantSearch.js 小部件，用于使用浏览器的原生功能在移动网站上进行语音搜索](https://www.algolia.com/doc/api-reference/widgets/voice-search/js/)。从那以后，我们从客户那里得到了很好的反馈，他们喜欢在几分钟内就能在他们的网站和应用程序中进行语音搜索。

algolia instant search 的目标是让伟大的搜索变得容易，同时给其他人提供工具来构建支持新形式搜索的新部件。语音也是如此，这就是为什么我们很高兴看到 AssemblyAI 的 InstantSearch.js 小部件支持跨平台浏览器的语音到文本转换。

## [](#where-can-you-use-this)这个用在哪里？

很简单:你可以在任何地方搜索！语音搜索不仅仅是“Alexa，一条鲸鱼有多重？”(这取决于鲸鱼的种类。)您的用户希望在他们喜欢的购物平台、内容网站和市场上进行语音搜索。普华永道的一项研究显示，71%的用户更喜欢通过语音搜索，而不是使用键盘。语音搜索通常比打字更快，用户发现他们犯的错误更少:我们需要给他们更简单的搜索方式。

如果你正在寻找更多的灵感和它是如何工作的，你可能想看我们关于[语音搜索能为零售](https://resources.algolia.com/ecommerce/how-to-harness-voice-search-in-the-retail-sector-2)做什么的网上研讨会，阅读我们关于媒体语音搜索的[电子书](https://resources.algolia.com/ebooks/the-guide-to-delightful-voice-search-in-media)，或者查看我们关于 [WW 如何使用 Algolia 来为他们的用户实现食品跟踪的案例研究](https://resources.algolia.com/ecommerce/ww-voice-search)。

## [](#how-long-does-it-take-to-implement)实施需要多长时间？

用 Algolia 实现语音搜索只需要不到一个下午的时间。你可以在我们的文档中看到所有的步骤[。这是一个很好的开始使用语音的方式，并且可以在用户想去的地方与他们见面。](https://www.algolia.com/doc/guides/getting-started/quick-start/tutorials/build-voice-search/)

请记住，语音搜索就是让你的用户能够在他们想到你卖的东西时，随时随地询问你的产品。没有这个简单的出口，他们可能会转而去谷歌或决定推迟到另一个时间——永远不会到来。声音确保你不会被遗忘。

想进一步了解 Algolia 能为您的搜索需求(网络、移动、语音)做些什么？[安排一次演示](https://www.algolia.com/schedule-demo/)。