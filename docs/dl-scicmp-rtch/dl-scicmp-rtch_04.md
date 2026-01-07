# 1  概述

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/basics_overview.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/basics_overview.html)

本书分为三个部分。第二部分和第三部分将分别探讨各种深度学习应用和必要的科学计算技术。然而，在这第一部分，我们将学习`torch`的基本构建块：张量、自动微分、优化器和模块。如果这不是一个可能产生的错误印象，我会称这部分为“`torch`基础”，或者按照一个常见的模板，称为“开始使用`torch`”。这些确实是基础，但在这里是指*基础*：在阅读接下来的章节后，你将对`torch`的工作原理有坚实的理解，并且将看到足够的代码，以便在后续章节中舒适地尝试更复杂的示例。换句话说，你将在一定程度上对`torch`*精通*。

此外，你将从头开始编写一个神经网络——甚至两次：一个版本将仅涉及原始张量和它们的内置功能，而另一个版本将利用专门的`torch`结构，以面向对象的方式封装神经网络训练所必需的功能。因此，你将为第二部分做好准备，在那里我们将探讨如何将深度学习应用于不同的任务和领域。
