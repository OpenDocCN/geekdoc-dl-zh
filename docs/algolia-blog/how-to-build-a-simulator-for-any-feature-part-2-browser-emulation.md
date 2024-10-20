# 如何为任何功能构建模拟器—第 2 部分:浏览器仿真

> 原文：<https://www.algolia.com/blog/engineering/how-to-build-a-simulator-for-any-feature-part-2-browser-emulation/>

在本系列的[上一篇文章中，我们谈到了单元测试如何帮助我们轻松地测试我们的功能，以及进入使用单元测试的思维模式如何首先帮助我们编写更纯粹、更简单、更模块化的功能。但是当我们需要测试一个不符合功能的系统时会发生什么呢？如果我们不能依靠一个函数的输出来告诉我们它是否在工作呢？如果我们的函数依赖于 UI 设置的状态会怎样？在这种情况下，很难准确地模拟用户如何导航我们的应用程序的复杂性——如果没有一个有用的抽象或简化可以测试，我们可能只需要构建一个现场实例并假装成用户。](https://www.algolia.com/blog/engineering/how-to-build-a-simulator-for-any-feature-part-1-unit-testing/)

当然，我们不能自动化*那个*过程，对吗？这样做似乎太依赖人工和人脑了。事实证明，解决方案不在于自动化应用程序内部的任何东西，而在于*自动化浏览器*的输入，模拟你的鼠标和键盘，这样你正在测试的应用程序就无法区分模拟器和真人。这个过程被称为浏览器自动化。还有什么比浏览器本身的无头版本更好的工具来自动化浏览器呢 Chromium 项目维护了一个名为[puppet er](https://pptr.dev/)的 API，我们将为此使用它。

## [](#puppeteer-%e2%80%94-a-headless-browser)木偶师——无头浏览器

第一步是用一个 dev 实例启动并运行我们的站点，因为我们想在投入生产之前测试这些东西。由于我们只是在这里演示，我将在 Algolia 的生产主页上运行该测试。

接下来，我们将设置一个节点。JS 程序。为此选择一个逻辑位置(可能是项目根目录中的一个文件夹，以便您可以将它与您的构建过程集成)。在这个文件夹中，您将想要创建一个调用所有测试的`index.js`文件。这样，只要从 repo 的根目录运行`node test`就会运行所有的测试，并且您可以隐藏到文件夹中运行更多的特定测试。

现在，在一些情况下，在构建过程的最后让它成为一个后构建命令可能是有意义的，但是这些将是更复杂的测试，您可能想要手动运行，所以如果您正在本地开发和测试，最好就把它留在那里，让您的开发人员手动运行`node test`或`node test/specific-test.js`，当他们实际上想要运行 Puppeteer 测试时。如果您的应用程序有一个更复杂的构建过程，包括更少的本地测试和更多的云中开发实例，那么建立一个无服务器项目(这是 Vercel 和 Netlify 擅长的)，将它设置为 lambda 函数读取测试，然后修改测试以在 ping 该端点时运行，可能会更容易。测试实际上不需要从与应用程序相同的地方运行(因为测试是完全在客户机上运行的)，所以您可以制作一个带有一些按钮的小 GUI 来触发云中的每个测试。

运行完`npm install puppeteer`后，你需要一些样板代码。下面是一个示例测试的结构，它以我的身份登录到 Algolia 并获取某个应用程序的凭证:

```
const puppeteer = require('puppeteer');

const login = async ({width, height}) => {
	const browser = await puppeteer.launch();
	const page = await browser.newPage();
	await page.goto('<https://algolia.com>');
	await page.setViewport({
		width,
		height,
		deviceScaleFactor: 1
	});

	// test logic here

	await browser.close();
};

module.exports = async () => {
	login({
		width: 375,
		height: 667
	});
	login({
		width: 1150,
		height: 678
	});
};; 
```

快速概述:

1.  第一行只是导入了木偶师。木偶师是由 Chromium DevTools 团队运行的，所以它本质上是该生态系统的一部分。在这里导入它会链接到 Chromium 的一个 headless 版本(那个版本只是假装它有一个 GUI)，因为它是与 Chromium 一起开发的，所以那个版本保证能与 Puppeteer 一起工作。
2.  第一个函数叫做`login`，它运行我们的测试。我们现在在这里没有任何测试逻辑，只有启动 headless Chrome 浏览器的样板文件，转到 Algolia 网站，并将视窗设置为我们传递给函数的任何内容。最后，我们关闭浏览器，这样它就不会在内存中保持打开状态。
3.  然后，我们导出一个函数，在不同的屏幕尺寸上运行相同的测试。

* * *

要从`test`文件夹中的`index.js`内部运行这个测试，我们只需要运行`require("./login")();`。有了这种一行一测试的设置，我们甚至可以将测试分组为批次，并在命令行上用标志来触发这些批次，例如`node test --dashboard`。这里的定制没有限制，所以可以根据应用程序的需要进行调整。

在我的例子中，每当测试失败时，我都会遇到`throw`错误，而不担心批处理任何东西——我没有足够的测试来让我把精力集中在那里。不过，在您的应用程序中，这些测试函数返回测试结果可能会有所帮助，而不仅仅是在失败时显示错误，因为这将使您能够基于从`index.js`运行的测试打印出更详细的消息。现在，我的目标是让一切都安静地运行，除非有我们需要解决的问题。

让我们更深入地研究一下实际的测试逻辑，它可能比您预期的更简单:

```
// step 0\. accept cookies so the site works like normal
await page.waitForSelector('a[href="/policies/cookies/"] + button');
await page.click('a[href="/policies/cookies/"] + button');

// step 1\. wait for the log in link to appear and click on it
if (width < 960) {
	await page.waitForSelector('[data-mobile-menu-button]');
	await page.click('[data-mobile-menu-button]');
}
await page.waitForSelector('a[href="/users/sign_in"]');
await page.click('a[href="/users/sign_in"]');

// step 2\. wait for the email input to appear on the new page, focus on it, and type the email from .env
await page.waitForSelector('input[type=email]');
await page.focus('input[type=email]')
await page.keyboard.type(process.env.email);

// step 3\. focus on the password input, and type the password from .env
await page.focus('input[type=password]')
await page.keyboard.type(process.env.password);

// step 4\. click the log in button
await page.click("form#new_user button[type=submit]");

// step 5\. if we're on mobile, we'll need to click one more button to indicate we'd like to stay on mobile
if ((await page.url()).includes("mobile")) {
	await page.waitForSelector('.continue-button');
	await page.click('.continue-button');
}

// step 6\. wait for the application selector to appear and click it
await page.waitForSelector('#application-select');
await page.click("#application-select");

// step 7\. click on the application with the name in .env
await page.$$eval(
	'div > span.options',
	(options, applicationName) => {
		const matchedOptions = options.filter(option => option.innerText.split("\\n")[1] == applicationName);
		if (matchedOptions.length) {
			matchedOptions[0].click();
		} else {
			console.log(`Application ${applicationName} does not exist`);
		}
	},
	process.env.applicationName
);

// step 8\. wait for the api keys button to appear on the new page, and click it
await page.waitForSelector('#overview-api-keys-link');
await page.click("#overview-api-keys-link");

// step 9\. wait for the page to load and get the application id and public api key for this project
await page.waitForSelector('#api-keys-page-heading');
const [applicationID, publicAPIKey] = await page.$$eval(
	'input[readonly]',
	inputs => inputs.slice(0, 2).map(input => input.value)
);

// step 10\. log the results of our test
if (applicationID != process.env.applicationID)
	throw `Application ID is not correct. Test produced "${applicationID}", but it should have been "${process.env.applicationID}"`;
if (publicAPIKey != process.env.publicAPIKey)
	throw `Public API key is not correct. Test produced "${publicAPIKey}", but it should have been "${process.env.publicAPIKey}"`;

// step 11\. close the browser
await browser.close(); 
```

这不是一个微不足道的例子——如果你读了这个过程中每一步上面的评论，这些都不会太令人惊讶。它只是在真实的浏览器中模仿用户的行为，考虑到 Algolia 的主页处理移动和桌面的不同。伴随它的是一个`.env`文件，我在其中定义了用于登录的`email`和`password`、目标应用程序的`name`(用户将使用它来找到正确的应用程序)，以及测试应该返回的`publicAPIKey`和`applicationID`。如果一切顺利，这不会输出任何内容。但是如果有什么东西坏了，它会抛出一个描述性的错误，这样我们就可以在构建完成之前修复它。

那么我们从这个系列中学到了什么呢？有两种主要的测试方式:

1.  **单元测试** — *(正如本系列的[第 1 部分所讨论的)](https://www.algolia.com/blog/engineering/how-to-build-a-simulator-for-any-feature-part-1-unit-testing/)*这对于不与外界交互的简单、纯粹的函数来说是完美的。甚至隔离不纯函数的纯部分也是一个很好的实践，这样你就可以用简单的断言单独测试它们，这些断言和你测试的代码放在同一个文件中。
2.  **浏览器仿真** —对于更大的工作流，你需要你的测试实际上假装成一个用户，所以我们启动了 Chromium 的无头版本 Puppeteer 来“操纵”一个假用户，并遍历整个过程，浏览器甚至不知道它正在与另一个程序交互。

应该注意的是，这些方法并不相互排斥——在 Algolia 仪表板中向开发人员显示正确的 API 凭证这一特殊功能非常重要，足以用这两种方法进行测试。我们已经编写了一个木偶程序，从用户的角度来遍历整个工作流程，但是除此之外，分解我们搜集的 API 关键页的水合过程也是有帮助的。如果我们在该页面中隔离尽可能多的纯函数，我们可以用单元测试来测试它们，以确保绝大多数的进程不会无声无息地中断。因为它是从数据库中提取数据来显示在视图中，所以不可能使整个事情变得纯粹，但是这就是使用两种类型的测试的好处:我们不需要用任何一种方法来获得 100%的覆盖率。只要它们涵盖了所有内容，我们就可以对应用程序的持久性和长期可持续性充满信心。测试愉快！