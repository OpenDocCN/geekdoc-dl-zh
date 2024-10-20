# 不要做傲慢的模型

> 原文：<https://web.archive.org/web/20230101103301/https://www.datacamp.com/blog/dont-make-arrogant-models>

[![](img/e6af7b319ae4b8bdf9120c0e93f52219.png)](https://web.archive.org/web/20221210082657/https://www.datacamp.com/groups/business)

## 傲慢对你的模特来说可不是什么好品质。

一个鲜为人知的事实是，科学家生产的模型数据通常不够健壮或容错，无法真正投入生产。当然，当输入数据与您的定型数据和维持数据相似时，您可以相信您的预测。任何数据科学家都可以构建模型对象，以与训练数据相同的格式传入数据，并获得看似可靠的预测。

在现实世界中，事情总是比你想象的更糟糕。当数据科学家将模型扔出围栏时，ML 工程师或 IT 部门通常必须用护栏重建模型，以确保它们在生产中有用。作为数据科学家，我们应该做得更好——并摒弃那种认为我们的模型在每个生产场景中都将如我们所愿发挥作用的傲慢态度。我们不应该说，“这是我的 Jupyter 笔记本；我的工作完成了！”至少，我们应该从记录传入变量的预期行为开始，以帮助 ML 工程师更容易地编写运行时测试。更好的是，我建议增加一层单元测试来调整模型预测。

在本文中，我将带您浏览一个简单的错误处理示例，该示例使用 R 和包装模型预测函数的逻辑条件。Python 用户很可能熟悉`assert`、`try except`和常用的逻辑操作符来完成本文中涉及的许多相同的事情。本文使用 R 作为例子，因为许多使用 R 的数据科学家并不公开他们的模型作为生产的端点，这意味着这些模型“谦逊”的方面对他们来说可能是新的。

最终，我们的目标是为您的模型添加一层保护，以强制执行预期的行为，使其能够承受异常值，具有容错能力，并且在某些情况下，可以覆盖安全值的预测。这些运行时测试和编码护栏有助于使模型像 opencpu 服务器中的 POST 请求一样安全地用于生产。在您的模型函数中包含这些额外的预测行为将在风险承担者之间建立信任，您的模型不会傲慢地行事，并且尽管有异常值或意外的输入，也会交付价值。

## 示例设置

在本例中，您将[使用这个小样本数据集](https://web.archive.org/web/20221210082657/https://s3.amazonaws.com/assets.datacamp.com/email/other/final_Small_Customer_Data.csv)来构建一个客户倾向模型。这些虚假数据是汽车贷款营销活动的结果。输入变量包括当前的汽车制造和最近的储蓄账户余额。我们的分类模型将学习哪种汽车制造和账户余额有助于接受营销提议。当然，在现实世界中，你会有更多的数据，并遵循更严格的数据科学实践，如分区-但在这个例子中，我们将采取一些捷径，因为我们侧重于预测层。

[rpart]库用于递归分区来构建我们的决策树。类似地，`[rpart.plot]`([https://www . rdocumentation . org/packages/rpart . plot/versions/3 . 1 . 0](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/rpart.plot/versions/3.1.0))库将帮助我们快速构建一个看起来不错的树。接下来我们使用`[yardstick]`([https://www . rdocumentation . org/packages/scale/versions/0 . 0 . 9](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/yardstick/versions/0.0.9))轻松获得模型度量，使用`[ggplot2]`([https://www . rdocumentation . org/packages/gg plot 2/versions/3 . 3 . 5](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/ggplot2/versions/3.3.5))构建一个镶嵌图。下面的代码简单地用`[read.csv()]`([https://www . rdocumentation . org/packages/utils/versions/3 . 6 . 2/topics/read . table](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/read.table))加载数据，并用`[head()]`([https://www . rdocumentation . org/packages/utils/versions/3 . 6 . 2/topics/head](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/head))检查前六行，这样您就对输入有所了解了。

```py
# Libs
library(rpart)
library(rpart.plot)
library(yardstick)
library(ggplot2)

# Read in the data
fakeCustomers <- read.csv('final_Small_Customer_Data.csv')

# EDA
head(fakeCustomers) 
```

## 让我们建立一个简单的决策树

现在我们应用`[rpart()]`([https://www . rdocumentation . org/packages/RP art/versions/4.1-15/topics/RP art](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/rpart/versions/4.1-15/topics/rpart))函数来构建我们的决策树。因为您接受所有默认的模型参数，所以您只需要传入模型公式 Y_AcceptedOffer ~。和制作树的数据。但是，在模型公式中使用句点(`Y_AcceptedOffer~.,`)会增加模型行为的风险。假设稍后基础定型数据更改为包含其他列。通过使用句点，模型将简单地继承所有未定义为 Y 变量的列。因此，如果您在没有显式声明 x 变量的情况下，通过将已经更改的数据作为代码的来源来重建模型，您甚至会在不知道的情况下导致目标泄漏或过度拟合。因此，在公式中明确声明 x 变量通常是个好主意。最终，产生的`fit`对象是一个模型，我们不想简单地将它传递给 IT 部门。让我们也定义一个当 fit 得到一个未知值时的安全模型响应！

```py
# Fit the model
fit <- rpart(Y_AcceptedOffer~., fakeCustomers) 
```

## 做一些预测

让我们确保我们的模型在完美的输入下如预期的那样运行。在工作流的这一点上，您应该根据训练和验证集来评估模型性能。这里我们对原始数据使用`[predict()]`([https://www . rdocumentation . org/packages/stats/versions/3 . 6 . 2/topics/predict](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict))，用`[tail()]`([https://www . rdocumentation . org/packages/utils/versions/3 . 6 . 2/topics/head](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/head))检查其中的一部分，然后构造一个简单的混淆矩阵。最后，您用`table()`创建一个混淆矩阵，然后将`yardstick`的`[conf_mat()]`([https://www . rdocumentation . org/packages/码尺/versions/0 . 0 . 6/topics/conf _ mat](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/yardstick/versions/0.0.6/topics/conf_mat))嵌套在  中，以获得包括准确性在内的 13 个模型度量。请记住，营销人员没有无限的预算，所以你应该更关心前 1%或 5%的潜在客户的准确性，而不仅仅是准确性。

```py
# Get predictions
pred <- predict(fit, fakeCustomers, type ='class')

# Examine
results <- data.frame(preds   = pred,
                      actuals = fakeCustomers$Y_AcceptedOffer)
tail(results, 10) 
```

```py
# Simple Confusion Matrix
(confMat <-table(results$preds, results$actuals)) 
```

```py
# Obtain model metrics
summary(conf_mat(confMat)) 
```

## 外观检验

除了数字 KPI，您还可以通过马赛克图直观地检查混淆矩阵。在这个例子中，镶嵌图将具有代表混淆矩阵的每个部分的矩形，例如真阳性和假阳性。每个矩形的面积对应于混淆矩阵的值。这个视图可以让你很容易地理解你的课堂作业与实际情况相比有多平衡。下面的代码将原始混淆矩阵嵌套在 [conf_mat](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/yardstick/versions/0.0.6/topics/conf_mat) 和 ggplot2 的 [autoplot](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/ggplot2/versions/3.3.0/topics/autoplot) 函数中，以创建一个基本的镶嵌图。

```py
autoplot(conf_mat(confMat)) 
```

使用简单模型的一个好处是您可以询问模型的行为。对于决策树，您可以使用 [rpart.plot()](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.8/topics/rpart.plot) 函数来可视化结果。该图将让您了解每个节点中变量的分割值和重要性。

```py
rpart.plot(fit, roundint = F) 
```

## 没事吧。没那么快。

不要把这个模型代码发送给它，并期待热烈的回应！当然，对于这些虚假的潜在客户来说，它工作得很好——因为它们和训练数据一模一样。即使在正常的模型构建中，您通常会传入一个具有相似分布和相同因子级别的分区。但在现实中，数据完整性和其他因素可能是真正传入数据的问题，它们可能会破坏您的模型。

```py
fakeNew <- fakeCustomers[c(6:8),]
fakeNew

# Make a prediction
predict(fit, fakeNew, type = 'prob') 
```

## 为你的预测增加一层保护。

在本节中，您将探索当汽车品牌从 lexus 雷克萨斯更换为 Lexus 雷克萨斯时会发生什么。数据输入错误和输入错误时有发生，因为人们参与其中。键入错误的因素和调换数字输入经常会破坏生产中的模型，正如您在下面运行 predict(fit，fakeRecord)时看到的那样。

错误:因素 carMake 有新的水平雷克萨斯

```py
# Entry Form Error
fakeRecord <- fakeNew[1,]
fakeRecord[,2] <- as.factor('lexus')

# Uh-Oh!; Error: factor carMake has new level lexus
#predict(fit, fakeRecord) 
```

## 给你的模型添加一个不起眼的图层

让我们通过检查输入是否有意义来添加一个保护性的预测层，如果有意义，那么调用 [predict()](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict) 。在这段代码中，您编写了一个名为 humblePredict()的包装器函数，它接受要评分的新观察值。在 for 循环中，该函数检查:

1.  使用`is.data.frame`表示每一行都是数据帧的一部分
2.  使用 match `%in%`运算符确保数据帧的列与模型训练公式相匹配。
3.  `carMake`列中的观察值是来自模型训练数据的预期水平。这是另一个使用`%in%`的匹配操作符调用
4.  最后，使用`is.numeric`函数确定`'RecentBalance'`列是一个数值。

如果这四个逻辑条件都满足，那么 If 语句就像往常一样简单地调用`[predict()]`([https://www . rdocumentation . org/packages/stats/versions/3 . 6 . 2/topics/predict](https://web.archive.org/web/20221210082657/https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict))。但是，逻辑条件出现在 if-else 语句中。因此，如果这些条件中的任何一个返回 false，则执行 FALSE 代码块。在本例中，默认响应是“DidNotAccept”的“安全”模型响应。这个水平是安全的，因为它意味着公司不会花钱向这个潜在客户营销。当然，在您自己的工作中，您可能会有一个更明确的错误，使用不同的模型或简单地从您的训练集中返回平均 Y 值。关键是您可以完全控制错误代码的行为，并且应该确保您的模型具有与业务需求相对应的防护栏。这种类型的函数包装有助于您决定模型如何处理错误输入。当模型面临不良输入时，您想要错误、安全值、NA 还是其他输出？

```py
humblePredict <- function(x){
  classifications <- list()
  for(i in 1:nrow(x)){
    if(is.data.frame(x[i,]) == T &
     all(all.vars(formula(fit)[-2]) %in% names(x[i,])) == T &
     x[i,grep('carMake',names(x[i,]))] %in%
     unlist(attributes(fit)$xlevels) == T &
     is.numeric(x[i,grep('RecentBalance', names(x[i,]))])==T){
    response  <- predict(fit, x, type = 'class')
    classifications[[i]] <- response
  } else {
    response  <- 'DidNotAccept’'
    classifications[[i]] <- response
    }
  }

  return(unlist(classifications))
}
humblePredict(fakeRecord) 
```

这只是让模型在生产中更加健壮的冰山一角。您可以在 humblePredict 中编写代码来更改异常值数值输入，或者在级别未知的情况下将因子级别更改为最频繁的级别。如果您想了解更多，请从分别用于单元测试和运行时测试的`testthat()`和`assertive()`库开始。**没有断言或者至少是安全行为的文档，任何模型都不应该发送给 IT 部门。**

了解如何使用 [DataRobot](https://web.archive.org/web/20221210082657/https://www.datarobot.com/) 构建你可以信任的人工智能。