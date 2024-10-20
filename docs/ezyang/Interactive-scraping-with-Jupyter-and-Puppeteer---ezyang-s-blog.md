<!--yml

类别：未分类

日期：2024-07-01 18:16:57

-->

# 使用 Jupyter 和 Puppeteer 进行交互式抓取：ezyang 的博客

> 来源：[`blog.ezyang.com/2021/11/interactive-scraping-with-jupyter-and-puppeteer/`](http://blog.ezyang.com/2021/11/interactive-scraping-with-jupyter-and-puppeteer/)

## 使用 Jupyter 和 Puppeteer 进行交互式抓取

抓取网站的讨厌之处之一是在浏览器和实际抓取脚本之间来回跳转，您在浏览器中使用开发工具来确定应该使用哪些选择器来抓取数据，而您的实际抓取脚本通常是一些批处理程序，可能在调试之前需要执行几个步骤。一旦您的抓取器运行起来了，批处理脚本就没问题了，但在开发过程中，在某个页面上暂停抓取过程并在 DOM 中摆弄一下以查看该做什么，这真的非常方便。

这种交互式开发正是 Jupyter 笔记本擅长的；当与基于浏览器的抓取库如 Puppeteer 结合使用时，您可以正好达到这种工作流程。以下是设置步骤：

1.  Puppeteer 是一个 JavaScript 库，因此您需要一个支持 JavaScript 的 Jupyter 内核来运行它。作为额外的复杂性，Puppeteer 也是异步的，因此您需要一个支持异步执行的内核。幸运的是，[ijavascript-await](https://www.npmjs.com/package/ijavascript-await) 正好提供了这一功能。请注意，在最近的 node 版本上，此软件包无法编译；您可以安装此 PR 来解决这个问题：[`github.com/n-riesco/ijavascript/pull/257`](https://github.com/n-riesco/ijavascript/pull/257) 假设我们能够在 node 支持顶级 await 时使用原始 ijavascript，但目前尚不支持：[`github.com/nodejs/node/issues/40898`](https://github.com/nodejs/node/issues/40898)

1.  在存储 snotebooks 的目录内，您需要 `npm install puppeteer`，以便在笔记本中使用。

1.  使用 `let puppeteer = require('puppeteer'); let browser = await puppeteer.launch({headless: false});` 启动 Puppeteer，并获利！

会有一个实时浏览器实例，您可以使用开发工具进行操作，然后在 Jupyter 笔记本中键入命令，查看它们如何影响浏览器状态。

我 [在推特上发过一条推文](https://twitter.com/ezyang/status/1462199995923378179)，评论员提供了一些关于可以尝试的其他好建议：

+   您不必使用 Puppeteer；Selenium 也可以驱动浏览器，并且它还有 Python API（因此无需与替代 Jupyter 内核打交道）。我个人更喜欢在 JavaScript 中工作，因为页面脚本本身也是 JavaScript，但这主要是个人偏好的问题。

+   对于简单的交互，如果你只是想进行几次交互并记录它们，[Headless Recorder](https://github.com/checkly/headless-recorder) 提供了一个很好的扩展，可以直接在你的浏览器中记录操作，然后以可执行形式导出。我还没有试过它，但看起来使用起来会非常方便。
