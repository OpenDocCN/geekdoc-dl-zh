# 人工智能为用户测试和用户界面设计提供的建议

> 原文：<https://www.algolia.com/blog/engineering/ai-powered-recommendations-for-user-testing-and-ui-design/>

我们继续关于在后台使用人工智能推荐的系列报道。当我们建立 Algolia 推荐时，我们最初的目标是为电子商务客户推荐产品。但是我们推荐的 API 涵盖了许多其他用例。 这篇文章是关于如何改进你的在线和后台表单的设计，就像在一个典型的网站或 Salesforce UI 上看到的那样。

## [](#filling-out-painstakingly-complicated-online-forms)填写繁琐复杂的在线表格

对于客户和员工来说，填写电子表格需要耐心和专注。在最糟糕的情况下，它还需要大量的点击和屏幕变化，以及想象力的飞跃，才能找到适当的行动。

员工是错综复杂的电子表格的主要受害者。他们每天都要与表单进行交互，不同公司、不同系统、不同职业、不同任务的用户界面(UI)也各不相同。如果员工幸运的话，他们有开发人员和设计人员来创建 UI，但是众多的需求和系统约束阻碍了良好的设计。常用的解救方法是 *用户测试*——但是这既费钱又费时，而且用户的行为并不总是容易解释的。

## [](#how-ai-driven-recommendations-can-improve-electronic-form-ui-design)AI 驱动推荐如何改进电子表单 UI 设计

在这篇文章中，我们设想 Algolia 推荐自动化用户测试和 UI 设计过程。它可以捕捉动作的组合，并围绕这些组合建立模型。当这些模型定期输入员工打字和按钮点击(作为分析数据捕获)时，可以非常快速地生成如下推荐:

*   在同一屏幕上放置哪些字段
*   哪些字段应该相邻
*   在一个屏幕上问多少个问题
*   如何减少给定任务的屏幕和动作数量
*   是使用引导式导航(向导)还是允许自由进入

### [](#two-api-calls)**两个 API 调用**

在本系列的前一篇文章中，我们讨论了推荐如何依赖于两个 API:

1.  我们的**Insights API**发送事件并捕获用户活动和分析，如点击和转化。当发送了足够多的事件后，Algolia Recommend 就可以构建模型来执行第二步。
2.  我们的**推荐 API** 了解最频繁的组合。推荐可以帮助您发现用户在使用任何用户界面(尤其是表单)时采取的操作组合。

*注意推荐可以找到比频繁组合更多的组合。同样强大的是相关条目字段的概念。结合相关的领域打开了用户设计的新的可能性，这在本文中没有讨论。*

## [](#rearrange-and-reduce-the-number-of-fields-using-recommendations)使用建议重新排列并减少字段数量

### [](#the-general-idea)**大意**

首先，需要将 **字段中的** 用户倾向一起修改。然后，Algolia 对这些数据进行建模。该模型允许推荐引擎将 **常用字段与** 一起编译成一个完整的列表。有了这份报告，您的设计人员可以开始在单个屏幕上重新排列或删除字段，或者重新思考字段在工作流和系统中的分布方式。

例如，航空公司已经知道如何只问三个问题就开始预订:机场、日期和乘客数量。不幸的是，大多数公司不知道该问哪三个问题，所以他们在屏幕上散布了 10 个问题，希望用户自己找出合适的领域。只有 10 个字段的内部网应用程序很少见——大概有 15 到 30 个。

让我们看看推荐是如何简化表单交互性的。

### [](#the-code-to-send-user-activity-and-system-events)**代码发送用户活动和系统事件**

发送一个洞察事件，告知推荐引擎文本字段 1、3 和 8 被一起修改:

```
convertedObjectIDs(
  'on_save_form',
  'ui_forms_index',
  ["field1_id", "field3_id", "field8_id"]
);

```

### [](#getting-the-recommendations)**获取推荐**

返回与屏幕 1 上的字段 1 相关的字段:

```
$recommendations = $recommendClient->getFrequentlyBoughtTogether([
  [
    'indexName' => 'ui_forms_index',
    'objectID' => 'field1_id',
  ],
]);

```

这里要注意两件事:

1.  虽然 API 方法的名称经常是*一起购买，但是“购买”一词可以被认为是在任何给定事件 期间捕获两个或更多 objectIDs 的任何动作。我们在这里定义的事件是“on_save_form”。然而，如下所示，我们可以使用“在用户会话期间”。*
**   这个特殊的代码是关于只跟踪一个屏幕。接下来，我们将把它扩展到多个屏幕。*

 *### [](#what-you-can-learn-from-these-recommendations)**从这些推荐中你能学到什么**

您可以从这个用例中生成一个报告:

| 目标字段 | 经常一起编辑 |
| 字段 1 | 字段 3、8、11 |
| 字段 2 | 字段 3、9、20 |
| 场 X | 字段 a、b、c、.. |

通过一点旋转和交叉引用，你可以发现暗示字段不同位置的模式。也许您会将频繁的字段配对放在彼此相邻的位置。

## [](#improve-the-distribution-of-fields-over%c2%a0multiple%c2%a0screens-using-recommendations)改善字段在 **多个** 屏幕上的分布使用建议

继续前面的例子，让我们添加几个屏幕。您需要更改数据以反映不止一个屏幕的更大上下文。为此，你只需要做一个改变:

*   添加该字段的屏幕信息

### [](#the-code-to-send-user-activity-and-system-events%c2%a0)**代码发送用户活动和系统事件**

```
convertedObjectIDs(
  'on_save_form',
  'ui_forms_index',
  ["screen1.field1_id", "screen2.field1_id", "screen2.field2_id"]
);

```

### [](#getting-the-recommendations)**获取推荐**

返回与屏幕 1 上的字段 1 相关的字段:

```
$recommendations = $recommendClient->getFrequentlyBoughtTogether([
  [
    'indexName' => 'ui_forms_index',
    'objectID' => 'screen1.field1_id',
  ],
]);

```

### [](#what-you-can-learn-from-these-recommendations)**从这些推荐中你能学到什么**

如您所见，现在我们知道屏幕 1 上的字段 1 通常与屏幕 2 上的字段 1 和字段 2 的修改相结合。您需要执行额外的解析来分配正确的屏幕分类。

更进一步:什么会阻止我们跨多个 *系统* 捕获模式？您只需要指定字段所在的“系统”。然而，真正的技巧是事件:“保存”事件通常不能跨多个系统工作。我们可以在系统工作流中使用一个已知的端点，但是我们将使用时间来代替。

## [](#streamlining-the-workflow-and-efficiency-of-filling-out-forms-across-multiple-systems)简化跨多个系统填表的工作流程和效率

这里提出的解决方案是使用一个 **会话超时** 。

### [](#the-code-to-send-user-activity-and-system-events)**代码发送用户活动和系统事件**

```
convertedObjectIDs(
  'session_time_out',
  'ui_forms_index',
  ["system1.screen1.field1_id", "system2.screen1.field1_id", "system2.screen2.field2_id"]
);

```

### [](#getting-the-recommendations)**获取推荐**

返回与系统 1 屏幕 1 上的字段 1 相关的字段:

```
$recommendations = $recommendClient->getFrequentlyBoughtTogether([
  [
    'indexName' => 'ui_forms_index',
    'objectID' => 'system1.screen1.field1_id',
  ],
]);

```

## [](#conclusion)结论

这仅仅是如何使用人工智能驱动的推荐来分析用户活动的开始。我相信你能想到更多。这就是它的美妙之处:推荐是一个通用的工具，通过想象和一些思考就可以很容易地适应它。*