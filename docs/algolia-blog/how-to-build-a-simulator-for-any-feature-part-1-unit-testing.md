# 如何为任何特性构建一个模拟器——第 1 部分:单元测试

> 原文：<https://www.algolia.com/blog/engineering/how-to-build-a-simulator-for-any-feature-part-1-unit-testing/>

不久前，我的时间线上出现了一条推文，要求 Twitterverse 用三个词吓唬一位产品经理。在精彩的回答中有“还有一件事…”，“差不多完成了”，“那是遗留代码”，以及可能被低估的“我测试过了”。最后一条让我难以忘怀，因为它可能是最恐怖的一条。

这引出了更多的问题:你是如何测试的？你涵盖了所有的边缘案例吗？测试是否准确地反映了该产品的实际使用情况？测试是否只检查了预期的结果？测试是否涵盖了变更是否会破坏应用程序中的其他东西？令人欣慰的是，美国开发人员已经提出了一个很好的系统来系统而彻底地回答这些问题，所以让我们在本文中探索两种主要类型的自动化测试。

## [](#unit-tests)单元测试

在我们开始构建单元测试之前，最好后退一步，问问我们为什么需要它们。总的来说，我们希望确保我们的应用程序正常工作！单元测试只是一些小的代码片段，用来验证事情是否正常运行。然而，为了理解单元测试是如何做到这一点的，重要的是首先要认识到现代开发涉及模块化(将功能分割成尽可能小的部分)。一旦我们把我们的大任务分成这些小函数，它们中的许多将碰巧是[纯函数](https://en.wikipedia.org/wiki/Pure_function):给定相同的输入，确定性函数总是返回相同的输出，不影响函数之外的任何东西，也不受函数之外的任何东西的影响。

这些属性使得纯函数易于测试，因为我们可以模拟这些函数将如何被使用，并根据它的预期来检查输出。这就是我们所说的“单元测试”的含义:使用我们想要用假数据测试几次的函数，并确保我们得到正确的结果。例如，以用 Python 编写的著名的 Sigmoid 函数为例，它下面有一个简单的单元测试:

```
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def test_sigmoid():
	assert sigmoid(0) == 0.5
	assert 0.9999 < sigmoid(10) < 1
	assert 0.2689 < sigmoid(-1) < 0.269
	assert sigmoid(-100000000000000000) == 0 
```

果然，最后一行揭示了我们代码中的一个错误。谁能想到 Python 中的`math`库不能处理`-100000000000000000`这样的数字？事后看来，对我来说很明显——我们输入到`math.exp`的正面版本[稍微超出了双精度](https://stackoverflow.com/a/4050933/10593368)的范围，但是我以前没有想到这一点！这个小错误很好地证明了我们不可能知道关于我们的工具的所有事情。尽管这里的公式在技术上是正确的，但我使用的工具无法处理处理如此大的数字的边缘情况。幸运的是，如果我们给它一个负数(这是对`sigmoid`的一个正输入，因为我们给`math.exp`和`-x`)，那么`math.exp`会处理好这个问题，因为结果会舍入到 0，但是在`x`为正数的情况下，我们需要编写稍微不同的函数:

```
def sigmoid(x):
	if x >= 0:
		return 1 / (1 + math.exp(-x))
	else:
		ex = math.exp(x)
		return ex / (1 + ex) 
```

现在，我们的单元测试返回了，没有触发任何警报！如果我们的项目经理碰巧想到了一个我们没有想到的边缘情况，那么向测试函数添加另一个`assert`是微不足道的。

但是，如果我们要测试的函数不纯呢？这里有一个想法:尽你所能重构一个新的纯函数，使不纯的函数包含尽可能少的内容。例如，以这个 JavaScript 函数为例，它在运行 Caesar 密码之前从 DOM 获取一个字符串值:

```
const runCaesarCipherOnInput = () => {
	const input = document.getElementById("input");
	const result = document.getElementById("result");
	const key = 4;

	result.innerText = input
		.value
		.toUpperCase()
		.split("")
		.map((letter, i) => {
	    const code = input.charCodeAt(i);
	    if (65 <= code && code <= 90) {
				let n = code - 65 + key;
				if (n < 0) n = 26 - Math.abs(n) % 26;
				return String.fromCharCode((n % 26) + 65);
	    } else return letter;
		})
		.join("");
}; 
```

这个管用！但是很难进行单元测试，因为正在发生的一切。它不是确定性的(也就是说，对于给定的输入，函数并不总是返回相同的输出)，并且它包含一些超出函数范围的副作用。甚至`.map`调用中的内部循环函数也不纯粹！我们如何解决这个问题？

现在让我们试着去掉所有使函数不纯的东西，给我们留下一些易于测试的功能，然后我们将把这些部分重新构建到原始函数中。

```
const caesar = (msg, key, rangeStart=65, rangeEnd=91) => msg
	.split("")
	.map(letter => shiftLetter(letter, key, rangeStart, rangeEnd))
	.join("");

const shiftLetter = (letter, key, rangeStart, rangeEnd) => {
	const rangeLength = rangeEnd - rangeStart;
	const code = letter.charCodeAt(0);
	if (rangeStart <= code && code < rangeEnd) {
		let n = code - rangeStart + key;
		if (n < 0) n = rangeLength - Math.abs(n) % rangeLength;
		return String.fromCharCode((n % rangeLength) + rangeStart);
	} else return letter;
};

const runCaesarCipherOnInput = () => {
	const input = document.getElementById("input");
	const result = document.getElementById("result");
	const key = 4;

	result.innerText = caesar(input.value.toUpperCase(), key);
}; 
```

注意，在这个例子中，我们仍然有两个不纯的函数:第 3 行的函数被提供给`.map`和`runCaesarCipherOnInput`。但是，这些功能足够简单，不需要测试！它们实际上不做任何逻辑(它们主要是为了提供纯函数参数而存在)，常量声明和 DOM 读取非常容易理解，所以我们可以放心地认为它们会做我们想要做的事情。如果我使用一个`for`循环而不是`map`，我甚至可以走得更远，但是这里的要点是，创建单元测试通常不需要作为开发人员的你做很大的改变。这个小的重写实现了相同的目的，它更容易阅读，并且现在是可测试的——像`caesar`和`shiftLetter`这样的重要逻辑位是纯函数，所以我们可以像以前一样为它们编写断言测试:

```
const testCaesar = () => {
	if (caesar("HELLO", 1) != "IFMMP") throw "Assertion error"; // test basic case
	if (caesar("these are all lowercase letters, so nothing should happen", 8) != "these are all lowercase letters, so nothing should happen") throw "Assertion error"; // test lowercase
	if (caesar(caesar("DECRYPTED TEXT", 19), -19) != "DECRYPTED TEXT") throw "Assertion error"; // test negative keys for decryption
};

const testShiftLetter = () => {
	if (shiftLetter("L", 3, 65, 91) != "O") throw "Assertion error"; // test basic case
	if (shiftLetter("s", 14, 65, 122) throw "Assertion error"; // test wrap around, and custom ranges
}; 
```

这些函数应该去它们的逻辑对应的地方！为了使批处理测试更容易运行，将单元测试打包到逻辑组中，或者甚至使用专门的单元测试框架来为您处理样板文件可能会有所帮助。

但是，当我们试图测试一些不容易封装在函数中的东西时，会发生什么呢？用简单的单元测试来测试 UI 元素和整个工作流并不简单。接下来我们能做什么？查看本系列的第 2 部分:[模拟整个浏览器](https://www.algolia.com/blog/engineering/how-to-build-a-simulator-for-any-feature-part-2-browser-emulation/)。