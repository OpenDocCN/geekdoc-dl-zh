# 14  使用 luz 进行训练

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/training_with_luz.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/training_with_luz.html)

到本书的这一部分，你已经知道如何训练神经网络。说实话，但确实需要一些认知努力来记住像 `optimizer$zero_grad()`、`loss$backward()` 和 `optimizer$step()` 这样的步骤的正确执行顺序。此外，在比我们运行示例更复杂的场景中，需要主动记住的事情列表会更长。

例如，我们还没有讨论过如何处理机器学习的通常三个阶段：训练、验证和测试。另一个问题是数据在 *设备*（CPU 和 GPU，如果你有的话）之间的流动问题。这两个主题都需要在训练循环中引入额外的代码。编写此代码可能是乏味的，并可能导致错误。

你可以在本章末尾的附录中看到我确切指的是什么。但现在，我想专注于补救措施：一种高级、易于使用、简洁的方式来组织和仪表化训练过程，这是由建立在 `torch` 之上的包 `luz` 贡献的。

## 14.1 光之存在 - 光之存在 - 让有光

一个 *torch* 已经带来了一些光明，但有时在生活中，没有 *太亮* 的东西。`luz` 被设计成使使用 `torch` 进行深度学习尽可能轻松，同时允许轻松定制。在本章中，我们关注整体过程；定制的示例将在后续章节中呈现。

为了便于比较，我们采用我们的运行示例，并添加第三个版本，现在使用 `luz`。首先，“只是”直接移植示例；然后，我们将其适应到一个更现实的场景中。在那个场景中，我们

+   使用单独的训练、验证和测试集；

+   让 `luz` 在训练/验证期间计算 *metrics*；

+   展示如何使用 *回调* 在训练期间执行自定义操作或动态更改超参数；

+   解释上述 *设备* 发生了什么。

## 14.2 将玩具示例移植

### 14.2.1 数据

`luz` 不仅实质性地转换了训练神经网络所需的代码；它还在数据方面增加了灵活性。除了对 `dataloader()` 的引用外，其 `fit()` 方法还接受 `dataset()`、张量，甚至 R 对象，我们很快就能验证这一点。

我们首先生成一个 R 矩阵和一个向量，就像之前一样。不过，这次我们还将它们包装在 `tensor_dataset()` 中，并实例化一个 `dataloader()`。现在我们生成的不是 100 个，而是 1000 个观测值。

```r
library(torch)
library(luz)

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 1000

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

ds <- tensor_dataset(x, y)

dl <- dataloader(ds, batch_size = 100, shuffle = TRUE)
```

*### 14.2.2 模型*

要使用 `luz`，不需要对模型定义进行任何更改。注意，尽管如此，我们只是 *定义* 模型架构；我们从未自己实际 *实例化* 模型对象。

```r
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_module(
 initialize = function(d_in, d_hidden, d_out) {
 self$net <- nn_sequential(
 nn_linear(d_in, d_hidden),
 nn_relu(),
 nn_linear(d_hidden, d_out)
 )
 },
 forward = function(x) {
 self$net(x)
 }
)
```

*### 14.2.3 训练*

要训练模型，我们不再编写循环。`luz`用一种*声明式*风格取代了熟悉的*迭代*风格：你告诉`luz`你想要发生什么，就像一个温顺的魔法学徒一样，它会启动机器。

具体来说，指令发生在两个 - 必要的 - 调用中。

1.  在`setup()`中，你指定要使用的损失函数和优化器。

1.  在`fit()`中，你传递训练（和可选的验证）数据的引用，以及要训练的周期数。

如果模型是可配置的 - 意味着，它接受传递给`initialize()`的参数 - 那么第三个方法就派上用场了：`set_hparams()`，在另外两个方法之间调用。（这是`hparams`，代表超参数。）使用这个机制，你可以轻松地尝试不同的层大小，或其他可能影响性能的因素。

```r
fitted <- net %>%
 setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(dl, epochs = 200)
```

*运行此代码，你应该看到大约如下输出：

```r
Epoch 1/200
Train metrics: Loss: 3.0343                                                                               
Epoch 2/200
Train metrics: Loss: 2.5387                                                                               
Epoch 3/200
Train metrics: Loss: 2.2758                                                                               
...
...
Epoch 198/200
Train metrics: Loss: 0.891                                                                                
Epoch 199/200
Train metrics: Loss: 0.8879                                                                               
Epoch 200/200
Train metrics: Loss: 0.9036 
```

在上面，我们传递给`fit()`的是`dataloader()`。让我们检查引用`dataset()`是否同样可行：

```r
fitted <- net %>%
 setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(ds, epochs = 200)
```

*甚至，`torch`张量：

```r
fitted <- net %>%
 setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(list(x, y), epochs = 200)
```

*最后，R 对象，当我们在不处理张量时可能很有用。

```r
fitted <- net %>%
 setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(list(as.matrix(x), as.matrix(y)), epochs = 200)
```

*在接下来的章节中，我们始终会使用`dataloader()`；但在某些情况下，这些“快捷方式”可能很有用。

接下来，我们扩展玩具示例，说明如何解决更复杂的要求。
  
## 14.3 更现实的场景

### 14.3.1 集成训练、验证和测试

在深度学习中，训练和验证阶段是交替进行的。每个训练周期之后都跟着一个验证周期。重要的是，两个阶段中使用的数据必须严格不重叠。

在每个训练阶段，都会计算梯度并改变权重；在验证期间，不会发生这些。那么为什么要有一个验证集呢？如果我们对每个周期，都为两个部分计算与任务相关的度量标准，我们就可以看到我们是否对训练数据*过拟合*：也就是说，基于描述我们想要建模的总体人群的具体样本的结论。我们只需要做两件事：指示`luz`计算一个合适的度量标准，并传递一个指向验证数据的额外`dataloader`。

前者是在`setup()`中完成的，对于回归任务，常见的选项是均方误差或平均绝对误差（MSE 或 MAE）。既然我们已经在使用 MSE 作为我们的损失，那么让我们选择 MAE 作为度量标准：

```r
fitted <- net %>%
 setup(
 loss = nn_mse_loss(),
 optimizer = optim_adam,
 metrics = list(luz_metric_mae())
 ) %>%
 fit(...)
```

*验证`dataloader`是在`fit()`中传递的 - 但为了能够引用它，我们首先需要构建它！所以现在（预计我们还需要测试集），我们将原始的 1000 个观测值分成三个部分，为每个部分创建一个`dataset`和`dataloader`。

```r
train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(
 setdiff(1:length(ds), train_ids),
 size = 0.2 * length(ds)
)
test_ids <- setdiff(
 1:length(ds),
 union(train_ids, valid_ids)
)

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds,
 batch_size = 100, shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 100)
test_dl <- dataloader(test_ds, batch_size = 100)
```

*现在，我们已经准备好开始增强的工作流程：

```r
fitted <- net %>%
 setup(
 loss = nn_mse_loss(),
 optimizer = optim_adam,
 metrics = list(luz_metric_mae())
 ) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(train_dl, epochs = 200, valid_data = valid_dl)
```

```r
Epoch 1/200
Train metrics: Loss: 2.5863 - MAE: 1.2832                                       
Valid metrics: Loss: 2.487 - MAE: 1.2365
Epoch 2/200
Train metrics: Loss: 2.4943 - MAE: 1.26                                          
Valid metrics: Loss: 2.4049 - MAE: 1.2161
Epoch 3/200
Train metrics: Loss: 2.4036 - MAE: 1.236                                         
Valid metrics: Loss: 2.3261 - MAE: 1.1962
...
...
Epoch 198/200
Train metrics: Loss: 0.8947 - MAE: 0.7504
Valid metrics: Loss: 1.0572 - MAE: 0.8287
Epoch 199/200
Train metrics: Loss: 0.8948 - MAE: 0.7503
Valid metrics: Loss: 1.0569 - MAE: 0.8286
Epoch 200/200
Train metrics: Loss: 0.8944 - MAE: 0.75
Valid metrics: Loss: 1.0579 - MAE: 0.8292
```

尽管训练集和验证集来自完全相同的分布，但我们确实看到了一点过拟合。这是我们将在下一章中更多讨论的话题。

一旦训练完成，上面的 `fitted` 对象将包含 epoch 级别的指标历史，以及涉及训练过程中的许多重要对象的引用。其中后者包括拟合的模型本身——这为在测试集上获得预测提供了一个简单的方法：

```r
fitted %>% predict(test_dl)
```

```r
torch_tensor
 0.7799
 1.7839
-1.1294
-1.3002
-1.8169
-1.6762
-0.7548
-1.2041
 2.9613
-0.9551
 0.7714
-0.8265
 1.1334
-2.8406
-1.1679
 0.8350
 2.0134
 2.1083
 1.4093
 0.6962
-0.3669
-0.5292
 2.0310
-0.5814
 2.7494
 0.7855
-0.5263
-1.1257
-3.3117
 0.6157
... [the output was truncated (use n=-1 to disable)]
[ CPUFloatType{200,1} ]
```

我们还希望评估测试集上的性能：

```r
fitted %>% evaluate(test_dl)
```

```r
A `luz_module_evaluation`
── Results 
loss: 0.9271
mae: 0.7348
```

这种工作流程：同步进行训练和验证，然后检查并提取测试集上的预测，是我们在这本书中会多次遇到的事情。
  
### 14.3.2 使用回调“挂钩”到训练过程

在这个阶段，你可能觉得我们在代码效率上取得的成果，可能以灵活性为代价。自己编写训练循环，你可以安排各种事情发生：保存模型权重、调整学习率……无论你需要什么。

实际上，没有失去任何灵活性。相反，`luz` 提供了一种标准化的方式来实现相同的目标：回调。回调是可以在以下时间点执行任意 R 代码的对象：

+   当整体训练过程开始或结束时（`on_fit_begin()` / `on_fit_end()`）；

+   当一个 epoch（包括训练和验证）开始或结束时（`on_epoch_begin()` / `on_epoch_end()`）；

+   当在一个 epoch 期间，训练（验证，相应地）阶段开始或结束时（`on_train_begin()` / `on_train_end()`；`on_valid_begin()` / `on_valid_end()`）；

+   当在训练（验证，相应地）期间，一个新批次即将被处理或已经被处理时（`on_train_batch_begin()` / `on_train_batch_end()`；`on_valid_batch_begin()` / `on_valid_batch_end()`）；

+   甚至在“最内层”训练/验证逻辑中的特定里程碑，例如“损失计算后”、“`backward()` 后”或“`step()` 后”。

虽然你可以使用回调实现任何你想要的逻辑（我们将在后面的章节中看到如何做到这一点），但 `luz` 已经配备了一个非常有用的集合。例如：

+   `luz_callback_model_checkpoint()` 在每个 epoch 后（或根据指示仅在改进的情况下）保存模型权重。

+   `luz_callback_lr_scheduler()` 激活 `torch` 的一个 *学习率调度器*。存在不同的调度器对象，每个对象都遵循自己的逻辑，在动态更新学习率。

+   `luz_callback_early_stopping()` 一旦模型性能停止改进，就会终止训练。究竟“停止改进”意味着什么，用户可以配置。

回调作为列表传递给 `fit()` 方法。例如，增强我们最近的流程：

```r
fitted <- net %>%
 setup(
 loss = nn_mse_loss(),
 optimizer = optim_adam,
 metrics = list(luz_metric_mae())
 ) %>%
 set_hparams(d_in = d_in,
 d_hidden = d_hidden,
 d_out = d_out) %>%
 fit(
 train_dl,
 epochs = 200,
 valid_data = valid_dl,
 callbacks = list(
 luz_callback_model_checkpoint(path = "./models/",
 save_best_only = TRUE),
 luz_callback_early_stopping(patience = 10)
 )
 )
```

*使用这种配置，权重将被保存，但仅当验证损失下降时。如果十次 epoch 内没有改进（再次，在验证损失上），训练将停止。使用这两个回调，你可以选择任何其他指标作为决策依据，相关的指标也可能指的是训练集。

这里，我们看到在 111 个 epoch 后发生了早期停止：

```r
Epoch 1/200
Train metrics: Loss: 2.5803 - MAE: 1.2547
Valid metrics: Loss: 3.3763 - MAE: 1.4232
Epoch 2/200
Train metrics: Loss: 2.4767 - MAE: 1.229
Valid metrics: Loss: 3.2334 - MAE: 1.3909
...
...
Epoch 110/200
Train metrics: Loss: 1.011 - MAE: 0.8034
Valid metrics: Loss: 1.1673 - MAE: 0.8578
Epoch 111/200
Train metrics: Loss: 1.0108 - MAE: 0.8032
Valid metrics: Loss: 1.167 - MAE: 0.8578
Early stopping at epoch 111 of 200
```*  *### 14.3.3 `luz` 如何帮助设备

最后，让我们简要提及`luz`如何帮助进行设备放置。在通常的环境中，设备是 CPU，也许如果可用，还有 GPU。对于训练，数据和模型权重需要位于同一设备上。这可能会引入复杂性，并且至少需要额外的代码来保持所有组件同步。

使用`luz`，相关动作对用户来说是透明的。让我们以上面的预测步骤为例：

```r
fitted %>% predict(test_dl)
```

*如果这段代码是在具有 GPU 的机器上执行的，`luz`将检测到这一点，并且模型的权重张量已经移动到那里。现在，对于上面的`predict()`调用，“幕后”发生的情况如下：

+   `luz`将模型置于评估模式，确保权重不会被更新。

+   `luz`将测试数据逐批次移动到 GPU 上，并获取模型预测。

+   这些预测随后被移回 CPU，以便调用者可以使用 R 进一步处理它们。（如`as.numeric()`、`as.matrix()`等转换函数只能作用于 CPU 上的张量。）

在下面的附录中，您可以找到如何手动实现训练-验证-测试工作流程的完整指南。您可能会发现这比我们上面所做的方法要复杂得多——而且它甚至没有涉及到指标，或者`luz`回调提供的任何功能。

在下一章中，我们将讨论现代深度学习的基本要素，这些要素我们尚未涉及；随后，我们将探讨专门用于处理不同任务和领域的特定架构。
  


为了清晰起见，我们在此重复两件不依赖于您是否使用`luz`的事情：`dataloader()`准备和模型定义。

```r
# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 1000

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

ds <- tensor_dataset(x, y)

dl <- dataloader(ds, batch_size = 100, shuffle = TRUE)

train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(setdiff(
 1:length(ds),
 train_ids
), size = 0.2 * length(ds))
test_ids <- setdiff(1:length(ds), union(train_ids, valid_ids))

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds,
 batch_size = 100,
 shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 100)
test_dl <- dataloader(test_ds, batch_size = 100)

# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_module(
 initialize = function(d_in, d_hidden, d_out) {
 self$net <- nn_sequential(
 nn_linear(d_in, d_hidden),
 nn_relu(),
 nn_linear(d_hidden, d_out)
 )
 },
 forward = function(x) {
 self$net(x)
 }
)
```

*回想一下，使用`luz`后，您与观察训练和验证损失演变的唯一区别就是一个这样的片段：

```r
fitted <- net %>%
 setup(
 loss = nn_mse_loss(),
 optimizer = optim_adam
 ) %>%
 set_hparams(
 d_in = d_in,
 d_hidden = d_hidden, d_out = d_out
 ) %>%
 fit(train_dl, epochs = 200, valid_data = valid_dl)
```

*然而，没有`luz`，需要注意的事情可以分为三个不同的类别。

首先，实例化网络，如果已安装 CUDA，则将其权重移动到 GPU 上。

```r
device <- torch_device(if
(cuda_is_available()) {
 "cuda"
} else {
 "cpu"
})

model <- net(d_in = d_in, d_hidden = d_hidden, d_out = d_out)
model <- model$to(device = device)
```

*其次，创建一个优化器。

```r
optimizer <- optim_adam(model$parameters)
```

*第三，最大的部分：在每个 epoch 中，迭代训练批次以及验证批次，当处理前者时执行反向传播，而当处理后者时只是被动地报告损失。

为了清晰起见，我们将训练逻辑和验证逻辑各自封装到它们自己的函数中。`train_batch()`和`valid_batch()`将从遍历相应批次的循环内部调用。这些循环反过来将为每个 epoch 执行。

虽然`train_batch()`和`valid_batch()`本身会触发通常的按顺序执行的动作，但请注意设备放置调用：为了模型能够接收数据，它们必须位于同一设备上。然后，为了进行均方误差计算，目标张量也需要位于那里。

```r
train_batch <- function(b) {
 optimizer$zero_grad()
 output <- model(b[[1]]$to(device = device))
 target <- b[[2]]$to(device = device)

 loss <- nn_mse_loss(output, target)
 loss$backward()
 optimizer$step()

 loss$item()
}

valid_batch <- function(b) {
 output <- model(b[[1]]$to(device = device))
 target <- b[[2]]$to(device = device)

 loss <- nn_mse_loss(output, target)
 loss$item()
}
```

*在遍历 epoch 的循环中，有两行值得特别注意：`model$train()`和`model$eval()`。前者指示`torch`将模型置于训练模式；后者则相反。在我们这里使用的简单模型中，如果你忘记了这些调用，可能不会有什么问题；然而，当我们稍后使用正则化层如`nn_dropout()`和`nn_batch_norm2d()`时，正确调用这些方法至关重要。这是因为这些层在评估和训练期间的行为不同。

```r
num_epochs <- 200

for (epoch in 1:num_epochs) {
 model$train()
 train_loss <- c()

 # use coro::loop() for stability and performance
 coro::loop(for (b in train_dl) {
 loss <- train_batch(b)
 train_loss <- c(train_loss, loss)
 })

 cat(sprintf(
 "\nEpoch %d, training: loss: %3.5f \n",
 epoch, mean(train_loss)
 ))

 model$eval()
 valid_loss <- c()

 # disable gradient tracking to reduce memory usage
 with_no_grad({ 
 coro::loop(for (b in valid_dl) {
 loss <- valid_batch(b)
 valid_loss <- c(valid_loss, loss)
 }) 
 })

 cat(sprintf(
 "\nEpoch %d, validation: loss: %3.5f \n",
 epoch, mean(valid_loss)
 ))
}
```

*这完成了我们对手动训练的概述，并且应该使我的断言更加具体，即使用`luz`可以显著减少偶然（例如，复制粘贴）错误的可能性。

