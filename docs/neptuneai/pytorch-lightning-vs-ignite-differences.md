# PyTorch 闪电 vs 点燃:有什么区别？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pytorch-lightning-vs-ignite-differences>

Pytorch 是使用最广泛的深度学习库之一，仅次于 Keras。它为任何在开发和研究中使用深度学习方法的人提供了敏捷性、速度和良好的社区支持。

Pytorch 比 Tensorflow 有一定优势。作为一名人工智能工程师，我非常喜欢的两个关键特性是:

1.  Pytorch 有动态图(Tensorflow 有静态图)，这使得 Pytorch 实现更快，并增加了 pythonic 的感觉。
2.  Pytorch 很容易学，而 Tensorflow 有点难，主要是因为它的图形结构。

我在 Pytorch 上遇到的唯一问题是，当模型按比例放大时，它缺少结构。随着算法中引入越来越多的函数，模型变得复杂，很难跟踪细节。类似 Keras 的东西将有利于提供一个具有简单调用功能的高级接口。

今天，Pytorch 社区已经相当大了，不同的群体已经创建了解决相同问题的高级库。在本文中，我们将探索两个库: [Pytorch Lighting](https://web.archive.org/web/20221206042755/https://www.pytorchlightning.ai/) 和 [Pytorch Ignite](https://web.archive.org/web/20221206042755/https://pytorch.org/ignite/) ，它们为您的深度学习代码提供了灵活性和结构。

## 对比:Pytorch 照明与 Pytorch 点火

|  | 闪电 | 燃烧 |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
|  | 

TensorBoard，海王星，MLflow，Wandb，

彗星， | 

TensorBoard，Neptune，MLflow，Wandb，Polyaxon

 |
|  |  |  |
|  |  |  |
|  |  |  |
| 

制作
样书

 |  |  |
|  | 

功能指标和模块指标界面

 | 

如果不定义指标，则根据任务选择。在本文中，我们将其定义为(autosklearn . metrics . roc _ AUC)

 |

Pytorch 闪电是什么？

## Lightning 是构建在 Pytorch 之上的高级 python 框架。它是威廉·法尔孔在攻读博士学位时发明的。它是为研究人员创建的，专门用于尝试新的深度学习模型，其中涉及研究规模、多 GPU 训练、16 位精度和 TPU。

为什么是闪电？

### 创建 Lightning 的目的是通过消除低级代码，同时保持代码的可读性、逻辑性和易于执行，来扩展和加速研究过程。

Lightning 为 pytorch 函数提供了一种结构，在这种结构中，函数的排列方式可以防止模型训练过程中的错误，这种错误通常发生在模型放大时。

关键特征

### Pytorch Lightning 附带了许多功能，可以为专业人员以及研究领域的新手提供价值。

**在任何硬件**上训练模型:CPU、GPU 或 TPU，无需更改源代码

*   **16 位精度支持**:通过将内存使用减半来加快训练模型的速度
*   **可读性**:减少不想要的或样板代码，把重点放在代码的研究方面
*   **删除不需要的或样板代码**
*   **界面**:简洁、整洁、易于导航
*   **更容易复制**
*   **可扩展**:你可以使用多个数学函数(优化器、激活函数、损失函数等等)
*   **可重用性**
*   **与可视化框架**集成，如 Neptune.ai、Tensorboard、MLFlow、Comet.ml、Wandb
*   使用 PyTorch Lightning 的好处

## 闪电 API

### Lightning API 提供了与原始 Pytorch 相同的功能，只是更加结构化。在定义模型的时候，你不用修改任何代码，完全一样，你需要做的就是继承 **LightningModule** 而不是 **nn.module** 。LightningModule 负责建模深度学习网络的所有重要方面，例如:

定义模型的架构( *init*

*   定义训练、验证和测试循环(分别为训练 _ 步骤、验证 _ 步骤和测试 _ 步骤)
*   定义优化器(*configure _ optimizer*
*   Lightning 自带**lightning data module**；您可以创建自己的培训、验证和测试数据集，然后将其传递给培训师模块。
*   让我们看看在 Lightning 中定义模型是什么样子的。

如您所见，LightningModule 很简单，类似于 Pytorch。它负责所有需要定义的重要方法，比如:

```py
class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
```

**__init__** 负责模型和相关权重

*   **forward** :与 Pytorch Forward 相同，连接架构的所有不同组件，并向前传递
*   **训练 _ 步骤**:定义训练循环及其功能
*   **配置优化器**:定义优化器
*   还有其他功能:

**测试 _ 步骤**

*   **测试 _ 结束**
*   **配置优化器**
*   **验证 _ 步骤**
*   **验证 _ 结束**
*   **训练器**方法负责配置训练标准(时期数、训练硬件:CPU、GPU 和 TPU、GPU 数量等)。培训师的主要工作是将**工程代码**从研究代码中分离出来。
    最后，你需要做的就是调用**。从训练器实例中拟合**方法，传递定义好的模型和数据加载器，并执行。

韵律学

```py
mnist_model = MNISTModel()

train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)

trainer.fit(mnist_model, train_loader)
```

#### 指标的目的是允许用户使用某种数学标准来监控和测量训练过程，如:准确性、AUC、RMSE 等。它不同于损失函数；损失函数测量预测值和实际值之间的差异，并同时使用参数更新权重，而指标则用于监控模型在训练集和验证测试中的表现。这为模型的性能提供了有洞察力的行为。

Lightning 有两种指标:

功能度量

1.  模块度量接口
2.  **功能指标**

功能性指标允许您根据自己的需求创建自己的指标作为功能。Pytorch 为您提供了一个 tensor_metric decorator，它主要负责将所有输入和输出转换为张量，以便在所有 DDP 节点上同步度量的输出(如果 DDP 已初始化)。

Pytorch 还提供:

```py
import torch

from pytorch_lightning.metrics import tensor_metric

@tensor_metric()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  return torch.sqrt(torch.mean(torch.pow(pred-target, 2.0)))

```

numpy_metric:用 numpy 实现的度量函数的包装器

*   tensor_collection_metric:其输出不能转换为 torch 的度量的包装器。张量完全
*   **模块指标接口**

模块指标接口允许您为指标提供模块化接口。它负责张量转换，并处理 DDP 同步和 i/o 转换。

使用模块度量接口的另一种方法是使用普通 pytorch 创建一个度量函数，并从 lightning 基类派生一个类，然后在 forward 中调用您的度量:

```py
import torch

from pytorch_lightning.metrics import TensorMetric

class RMSE(TensorMetric):
    def forward(self, x, y):
        return torch.sqrt(torch.mean(torch.pow(x-y, 2.0)))
```

钩住

```py
import torch

from pytorch_lightning.metrics import TensorMetric

def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  return torch.sqrt(torch.mean(torch.pow(pred-target, 2.0)))
class RMSE(TensorMetric):
  def forward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rmse(pred, target)
```

#### Lightning 也有被称为**钩子**的处理程序。挂钩帮助用户在训练期间与教练互动。基本上，它让用户在训练中采取某种行动。

例如:

**on_epoch_start** :这是在 epoch 最开始的训练循环中调用的:

为了启用挂钩，请确保您覆盖了您的 **LightningModule** 中的方法，并定义了需要在期间完成的操作或任务，培训师将在正确的时间调用它。

```py
def on_epoch_start(self):

```

**on_epoch_end** :这是在 epoch 结束时的训练循环中调用的:

要了解更多关于钩子的信息，请点击[链接](https://web.archive.org/web/20221206042755/https://pytorch-lightning.readthedocs.io/en/0.4.9/Trainer/hooks)。

```py
def on_epoch_end(self):

```

分布式培训

### Lightning 提供多 GPU 培训和 5 种分布式后端培训:

数据并行' dp '

*   分布式数据并行' ddp '
*   分布式数据并行-2 'ddp2 '
*   分布式数据并行分片' dpp_sharded '
*   极速'极速'
*   这些设置可以在调用。拟合方法。

注意，为了使用分片发行版，您需要从 plugins 参数中调用它。

```py
Trainer = Trainer(distributed_backend = None)

Trainer = Trainer(distributed_backend ='dp')

Trainer = Trainer(distributed_backend ='ddp')
```

```py
trainer = Trainer(gpus=4, plugins='ddp_sharded')
```

查看这篇[文章](https://web.archive.org/web/20221206042755/https://towardsdatascience.com/sharded-a-new-technique-to-double-the-size-of-pytorch-models-3af057466dba)以更深入地了解**分片发行版**。

deepspeed 也是如此:

要了解更多关于 **deepspeed** 的信息，请查看这篇[文章](https://web.archive.org/web/20221206042755/https://pytorch-lightning.medium.com/pytorch-lightning-v1-2-0-43a032ade82bhttps://pytorch-lightning.medium.com/pytorch-lightning-v1-2-0-43a032ade82b)。

```py
trainer = Trainer(gpus=4, plugins='deepspeed', precision=16)
```

再现性

### 这样，**再现性变得非常容易**。为了一次又一次地重现相同的结果，你需要做的就是设置伪随机发生器的种子值，并确保训练器中的确定性参数为真**。**

有了上面的配置，您现在可以放大模型，甚至不用担心模型的工程方面。请放心，一切都由闪电模块负责。

```py
from pytorch_lightning import Trainer, seed_everything

seed_everything(23)

model=Model()
Trainer = Trainer(deterministic = True)
```

它规范了代码。

你所需要做的就是关注研究方面，包括操纵数学函数，增加一层神经元，甚至改变训练硬件。

它将工程与研究分离开来。

与海王星整合

### Lightning 提供了与 Neptune 的无缝集成。你需要做的就是调用 **NeptuneLogger** 模块:

如上所示设置所有需要的参数，然后将其作为一个参数传递给 trainer 函数，您就可以通过 Neptune 仪表盘监视您的模型了。

```py
from pytorch_lightning.loggers.neptune import NeptuneLogger

neptune_logger = NeptuneLogger(
   api_key="ANONYMOUS",
   project_name="shared/pytorch-lightning-integration",
   close_after_fit=False,
   experiment_name="train-on-MNIST",
   params=ALL_PARAMS,
   tags=['1.x', 'advanced'],
)
```

生产

```py
trainer = pl.Trainer(logger=neptune_logger,
                    checkpoint_callback=model_checkpoint,
                    callbacks=[lr_logger],
                    **Trainer_Params)
```

### 将 lightning 模型部署到生产环境中也非常简单，就像使用。to_torchscript，。to_onnx 并且有三种方法可以保存用于生产的模型:

将模型保存为 PyTorch 检查点

1.  将模型转换为 ONNX
2.  将模型导出到 Torchscript
3.  要获得关于模型部署和生产的更深入的知识，请查看这篇[文章](https://web.archive.org/web/20221206042755/https://towardsdatascience.com/how-to-deploy-pytorch-lightning-models-to-production-7e887d69109f)。

社区

### 闪电社区正在成长。几乎有 **390 名贡献者**、**11 名研究科学家**组成的核心团队、博士生，以及超过 **17k 的活跃用户**。因为社区正在快速发展，文档非常重要。

如果你发现自己有什么问题，可以在 Lightning 的 Slack 或者 Github 上寻求帮助。

Lightning 的[文档](https://web.archive.org/web/20221206042755/https://pytorch-lightning.readthedocs.io/)非常简洁、易读、易懂。还包括视频解释。

何时使用 PyTorch 闪电

## 研究和创造新的建筑。

*   寻找分布式并行培训。
*   寻找 CPU，GPU 和 TPU 培训。在 PyTorch 中，您可以轻松地从训练器本身更改硬件。
*   它提供了 SOTA 架构，因此您可以根据自己的需要调整它的设置。
*   何时不使用 PyTorch 闪电

## 如果你不知道 PyTorch，那就先学 PyTorch 再用闪电。可以去看看[闪电](https://web.archive.org/web/20221206042755/https://github.com/PyTorchLightning/lightning-flash)。

*   代码对比:Pytorch vs 闪电

## 从上面的例子中，您可以看到，Lightning 为每个操作提供了更专用的功能:构建模型、加载数据、配置优化器等等，此外，它还负责样板代码，如配置训练循环。它更侧重于研究方面，而不是工程方面。

什么是 Ignite？

## Ignite 是在 PyTorch 基础上开发的另一个高级库。它有助于神经网络训练。和闪电一样，它也是为研究人员创造的。它需要更少的纯 PyTorch 代码，这增加了界面的灵活性和简单性。

为什么点燃？

### Ignite 为用户提供了一个界面，将架构、标准和损耗整合到一个函数中，用于培训和评估(可选)。这个特性使 Ignite 基于 Pytorch 的基础，**，同时也使用户意识到从工程术语中分离出来的高级抽象**(可以稍后在培训之前配置)。这给了用户很大的灵活性。

关键特征

### Ignite 提供了三个高级功能:

**引擎**:这允许用户构造不同的配置用于训练和评估；

*   **现成的**指标:允许用户轻松评估模型；
*   **内置处理程序:**这允许用户创建*训练管道，记录*，或者简单地说——*与引擎交互*。**
**   使用 PyTorch Ignite 的好处*

 *## 点燃 API

### **引擎:**引擎让用户在每一批数据集上运行给定的**抽象**，在它经历时期、日志记录等时发出事件。

让我们看看发动机是如何工作的:

正如你所看到的，用于训练深度学习模型的抽象或基础被包含在函数 *update_model* 中，然后被传递到引擎中。这只是包含反向传播的训练函数。没有定义额外的参数或事件。

```py
def update_model(engine, batch):
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(update_model)
```

要开始训练，您只需调用**。从训练器运行**方法，根据要求定义 *max_epochs* 。

```py
trainer.run(data_loader, max_epochs=5)
```

事件和处理程序

#### 事件和处理程序帮助用户在训练期间与引擎进行交互。基本上，它让用户监控模型。我们将看到两种与模型互动的方式:借助**装饰者、**和*的帮助。添加事件处理程序。*

下面的函数使用@trainer.on decorator 打印评估器在训练数据集上运行的结果。

如您所见，函数中的主要元素是 train_evaluator，它主要对训练数据执行评估并返回指标。可以使用相同的度量来发现准确性、损失等。您所要做的就是给出一个打印件或一个 return 语句，以便获得值。

```py
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['nll']
    last_epoch.append(0)
    training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))
```

另一种方法是使用**。训练器的 add_event_handler** 。****

上面的代码使用验证数据集来操作指标。这个和上一个一模一样。唯一的区别是我们在**中传递这个函数。训练器**、**的 add_event_handler** 方法，它将像前面的函数一样工作。

```py
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['nll']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
```

有相当多的**内置**事件可以让你在训练期间或训练结束后与教练互动。

例如，**事件。EPOCH_COMPLETED** 会在 EPOCH 完成后执行某个功能。**事件。另一方面，完成的**将在训练完成后执行。

韵律学

#### Ignite 提供准确度、精确度、召回率或混淆矩阵等指标，以计算各种质量。

例如，下面我们计算训练数据集的准确度。

从上面的代码中，您可以看到用户必须**将度量实例**连接到引擎。然后使用引擎的 **process_function** 的输出来计算度量值。

```py
from ignite.metrics import Accuracy

def predict_on_batch(engine, batch)
    model.eval()
    with torch.no_grad():
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

    return y_pred, y

evaluator = Engine(predict_on_batch)
Accuracy().attach(evaluator, "val_acc")
evaluator.run(val_dataloader)
```

Ignite 还允许用户通过算术运算创建自己的指标。

分布式培训

### Ignite 支持分布式培训，但用户必须对其进行相应的配置，这可能需要很大的努力。用户需要正确设置分布式过程组、分布式采样器等。如果你不熟悉如何设置它，它会非常乏味。

再现性

### Ignite 的可再现性在于:

Ignite 自动处理随机状态的能力，这可以确保批处理在不同的运行时间具有相同的数据分布。

*   Ignite 不仅允许集成 Neptune，还允许集成 MLflow、Polyaxon、TensorBoard 等等。
*   与海王星整合

### 这个[海王星整合](https://web.archive.org/web/20221206042755/https://docs.neptune.ai/integrations/ignite/)非常容易。你所需要做的就是 [pip 安装 neptune-client 库](https://web.archive.org/web/20221206042755/https://docs.neptune.ai/setup/installation/)，然后你只需从**ignite . contrib . handlers . Neptune _ logger**中调用 **NeptuneLogger** 。

有趣的是，您可以附加许多事件处理程序，这样所有数据都将显示在 Neptune 仪表盘中，这将有助于您监控训练。

```py
from ignite.contrib.handlers.neptune_logger import *

npt_logger = NeptuneLogger(api_token="ANONYMOUS",
                           project_name='shared/pytorch-ignite-integration',
                           name='ignite-mnist-example',
                           params={'train_batch_size': train_batch_size,
                                   'val_batch_size': val_batch_size,
                                   'epochs': epochs,
                                   'lr': lr,
                                   'momentum': momentum})

```

下面你会发现一些例子，说明如何用 Neptune 附加事件处理程序。

社区

```py
npt_logger.attach(trainer,
                  log_handler=OutputHandler(tag="training",
                                            output_transform=lambda loss: {'batchloss': loss},
                                            metric_names='all'),
                  event_name=Events.ITERATION_COMPLETED(every=100))

npt_logger.attach(train_evaluator,
                  log_handler=OutputHandler(tag="training",
                                            metric_names=["loss", "accuracy"],
                                            another_engine=trainer),
                  event_name=Events.EPOCH_COMPLETED)

npt_logger.attach(validation_evaluator,
                  log_handler=OutputHandler(tag="validation",
                                            metric_names=["loss", "accuracy"],
                                            another_engine=trainer),
                  event_name=Events.EPOCH_COMPLETED)

```

### Ignite 社区正在成长；在撰写本文时，几乎有 **124 个贡献者**和超过 **391 个活跃用户**。

何时使用 PyTorch Ignite

## 具有优秀界面的高级库，具有根据需求定制 Ignite API 的附加属性。

*   当您想分解代码，但不想牺牲灵活性来支持复杂的训练策略时

*   提供了一个丰富的实用工具支持环境，如指标、处理程序和记录器，可用于轻松地评估/调试您的模型，它们可以单独配置。

*   何时不使用 PyTorch 点火

## 如果你不熟悉 Pytorch。

*   如果你不精通分布式培训，只是想轻松使用它。

*   如果你不想花很多时间去学习一个新的库。

*   代码比较:Pytorch 与 Ignite

## **纯 PyTorch**

**PyTorch-Ignite**

```py
model = Net()
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = torch.nn.NLLLoss()

max_epochs = 10
validate_every = 100
checkpoint_every = 100

def validate(model, val_loader):
    model = model.eval()
    num_correct = 0
    num_examples = 0
    for batch in val_loader:
        input, target = batch
        output = model(input)
        correct = torch.eq(torch.round(output).type(target.type()), target).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    return num_correct / num_examples

def checkpoint(model, optimizer, checkpoint_dir):
    filepath = "{}/{}".format(checkpoint_dir, "checkpoint.pt")
    obj = {"model": model.state_dict(), "optimizer":optimizer.state_dict()}
    torch.save(obj, filepath)

iteration = 0

for epoch in range(max_epochs):
    for batch in train_loader:
        model = model.train()
        optimizer.zero_grad()
        input, target = batch
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if iteration % validate_every == 0:
            binary_accuracy = validate(model, val_loader)
            print("After {} iterations, binary accuracy = {:.2f}"
                  .format(iteration, binary_accuracy))

        if iteration % checkpoint_every == 0:
            checkpoint(model, optimizer, checkpoint_dir)
        iteration += 1
```

[*来源*](https://web.archive.org/web/20221206042755/https://colab.research.google.com/drive/1gFIPXmUX73HWlLSxFvvYEweQBD_OPx1t#scrollTo=FnUEHqN9lPcb)

```py
model = Net()
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = torch.nn.NLLLoss()

max_epochs = 10
validate_every = 100
checkpoint_every = 100

trainer = create_supervised_trainer(model, optimizer, criterion)
evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()})

@trainer.on(Events.ITERATION_COMPLETED(every=validate_every))
def validate(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("After {} iterations, binary accuracy = {:.2f}"
          .format(trainer.state.iteration, metrics['accuracy']))

checkpointer = ModelCheckpoint(checkpoint_dir, n_saved=3, create_dir=True)
trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every),
                          checkpointer, {'mymodel': model})

trainer.run(train_loader, max_epochs=max_epochs)
```

如您所见，Ignite 压缩了 pytorch 代码，使您在研究领域更加高效，您可以在跟踪和操作工程方面(即模型训练)的同时练习不同的技术。

结论

## Lightning 和 Ignite 各有各的好处。如果您正在寻找灵活性，那么 Ignite 是不错的选择，因为您可以使用传统的 Pytorch 来设计您的架构、优化器和整体实验。Ignite 将帮助您组装特定功能的不同组件。

如果你正在为一个新的**设计**寻找**快速**原型，或者研究最先进的 ML 方法，那么就用闪电吧。这将有助于你专注于研究方面，并有助于你更快地扩大模型，从而减少误差。此外，它还提供 TPU 和并行分布。

我希望你喜欢这篇文章。如果您想尝试一些实际的例子，请访问 Lightning 和 Ignite 的笔记本链接(在本文开头的比较表中)。

感谢阅读！

资源

### [闪电文件](https://web.archive.org/web/20221206042755/https://pytorch-lightning.readthedocs.io/en/latest/index.html)

1.  [点燃文档](https://web.archive.org/web/20221206042755/https://pytorch.org/ignite/)
2.  [Neptune 记录器文档](https://web.archive.org/web/20221206042755/https://docs.neptune.ai/integrations/)
3.  [8 位创作者和核心贡献者谈论他们来自 PyTorch 生态系统的模型训练库](/web/20221206042755/https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem)
4.  [8 Creators and Core Contributors Talk About Their Model Training Libraries From PyTorch Ecosystem](/web/20221206042755/https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem)*