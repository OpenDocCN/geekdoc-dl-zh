# 如何跟踪海王星的 PyTorch 闪电实验

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pytorch-lightning-neptune-integration>

使用 PyTorch Lightning 并想知道您应该选择哪个记录器来[跟踪您的实验](/web/20221206015925/https://neptune.ai/experiment-tracking)？

想要找到保存超参数、指标和其他建模元数据的好方法吗？

考虑使用 PyTorch Lightning 来构建您的深度学习代码，并且不介意了解它的日志记录功能？

不知道闪电有一个相当可怕的海王星积分？

这篇文章(很可能)适合你。

## 为什么是 PyTorch 闪电和海王星？

如果你从未听说过，PyTorch Lightning 是 PyTorch 之上的一个非常轻量级的包装器，它更像是一个编码标准而不是框架。这种格式可以让你摆脱大量的样板代码，同时保持简单易懂。

其结果是一个框架，为研究人员、学生和生产团队提供了尝试疯狂想法的终极灵活性，而不必学习另一个框架，同时自动化掉所有的工程细节。

您可以获得的一些出色功能包括:

*   在不改变代码的情况下，在 CPU、GPU 或 TPUs 上进行训练，
*   琐碎的多 GPU 和多节点训练
*   微不足道的 16 位精度支持
*   内置性能分析器(训练器(profile=True))

以及一大堆其他伟大的功能。

但是，伴随着这种轻松运行实验的强大功能和随意调整的灵活性，出现了一个问题。

如何跟踪所有变化，例如:

*   损失和指标，
*   超参数
*   模型二进制
*   验证预测

和其他能帮助你组织实验过程的东西？

### PyTorch 闪电记录器

幸运的是，PyTorch Lightning 为您提供了一个将记录器轻松连接到 pl 的选项。训练器和一个[支持的记录器](https://web.archive.org/web/20221206015925/https://pytorch-lightning.readthedocs.io/en/latest/api_references.html#loggers)可以跟踪之前提到的所有东西(和许多其他东西)是 NeptuneLogger，它保存你的实验在…你猜对了，[海王星](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning)。

海王星不仅跟踪你的实验文物，而且:

最好的部分是，这种集成使用起来真的很简单。

让我给你看看它是什么样子的。

你也可以看看这个 [colab 笔记本](https://web.archive.org/web/20221206015925/https://colab.research.google.com/drive/1EThGG9EbN6rjMITFUKNtUwQdth_Zuu36?usp=sharing)，玩玩我们将要谈到的你自己的例子。

## PyTorch 闪电日志:基本集成(保存超参数、指标等)

在最简单的情况下，您只需创建`NeptuneLogger`:

```py
from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration")
```

并将其传递给`Trainer`的 logger 参数，以符合您的模型。

```py
from pytorch_lightning import Trainer

trainer = Trainer(logger=neptune_logger)
trainer.fit(model)

```

通过这样做，您可以自动:

*   记录指标和损失(并创建图表)，
*   记录并保存超参数(如果通过 lightning hparams 定义)，
*   记录硬件利用率
*   记录 Git 信息和执行脚本

看看这个实验。

你可以监控你的实验，比较它们，并与他人分享。

对一辆四缸车来说还不错。

但是只要多一点努力，你就能得到更多。

## PyTorch 闪电测井:高级选项

Neptune 为您提供了许多定制选项，您可以简单地记录更多特定于实验的内容，如图像预测、模型权重、性能图表等等。

所有这些功能对 Lightning 用户都是可用的，在下一节中，我将向您展示如何充分利用 Neptune。

### 创建 NeptuneLogger 时记录额外信息

创建记录器时，您可以记录其他有用的信息:

*   [代码](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#code):快照脚本、jupyter 笔记本、配置文件等等，
*   超参数:记录学习率、历元数和其他东西(如果你正在使用 lightning 的 lightning `hparams`对象，它将被自动记录)
*   [属性](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#data-versions):日志数据位置、数据版本或其他
*   [标签](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#tags):添加“resnet50”或“无增强”等标签来组织您的跑步。

只需将这些信息传递给你的记录器:

```py
neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project="shared/pytorch-lightning-integration",
    tags=["pytorch-lightning", "mlp"],
)

```

### 用 PyTorch Lightning 记录训练中的额外事情

训练中可以记录很多有趣的信息。

您可能对监控以下内容感兴趣:

*   每个时期后的模型预测(考虑预测遮罩或覆盖的边界框)
*   诊断图表，如 ROC AUC 曲线或混淆矩阵
*   [模型检查点](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#model-checkpoints)，或其他对象

这真的很简单。只需转到您的`LightningModule`并调用作为`self.logger.experiment`可用的 Neptune 实验的方法。

例如，我们可以记录每个时期后的损失直方图:

```py
class CoolSystem(pl.LightningModule):

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        fig = plt.figure()
        losses = np.stack([x['val_loss'].numpy() for x in outputs])
        plt.hist(losses)
        neptune_logger.experiments['loss_histograms'].log(File.as_image(fig))
        plt.close(fig)

        return {'avg_val_loss': avg_loss}
```

[自己探索它们](https://web.archive.org/web/20221206015925/https://app.neptune.ai/shared/pytorch-lightning-integration/e/PYTOR-173293/all?path=imgs&attribute=loss_histograms)。

[在培训期间，您可能希望记录的其他事情](https://web.archive.org/web/20221206015925/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)有:

*   `neptune_logger.experiment["your/metadata/metric"].log(metric)` #记录自定义指标
*   `neptune_logger.experiment["your/metadata/text"].log(text)` #日志文本值
*   `neptune_logger.experiment["your/metadata/file"].upload(artifact)` #日志文件
*   `neptune_logger.experiment["your/metadata/figure"].upload(File.as_image(artifact))` #日志图片、图表
*   `neptune_logger.experiment["properties/key"] = value` #添加键值对
*   `neptune_logger.experiment["sys/tags"].add(['tag1', 'tag2'])` #为组织添加标签

很酷吧？

但是…这不是你能做的全部！

### PyTorch 闪电训练结束后记录东西

跟踪你的实验不一定要在你做完后才结束。安装循环末端。

您可能想要跟踪`trainer.test(model)`的指标，或者计算一些额外的验证指标并记录下来。

要做到这一点，你只需要告诉`NeptuneLogger`不要在安装后关闭:

```py
neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration",
    ...
)

```

…您可以继续记录🙂

**测试指标:**

```py
trainer.test(model)
```

**其他(外部)指标:**

```py
from sklearn.metrics import accuracy_score
...
accuracy = accuracy_score(y_true, y_pred)
neptune_logger.experiment['test/accuracy'].log(accuracy)
```

**测试集上的性能图表:**

```py
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
...
fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
neptune_logger.experiment['test/confusion_matrix'].upload(File.as_image(fig))
```

**整个模型检查点目录:**

```py
neptune_logger.experiment('checkpoints').upload('my/checkpoints')

```

[转到本实验](https://web.archive.org/web/20221206015925/https://app.neptune.ai/shared/pytorch-lightning-integration/e/PYTOR-173293/all?path=&attribute=confusion_matrix)查看这些对象是如何被记录的:

但是…还有更多！

海王星让你在训练后获取实验。

让我告诉你怎么做。

### 把你的 PyTorch 闪电实验信息直接拿到笔记本上

您可以在实验完成后获取实验，分析结果，并更新度量、工件或其他东西。

例如，让我们将实验仪表板提取到熊猫数据帧:

```py
import neptune.new as neptune

project = neptune.init('shared/pytorch-lightning-integration')
project.fetch_runs_table().to_pandas()

```

或者获取一个单独的实验并用训练后计算的一些外部度量来更新它:

```py
exp = neptune.init(project='shared/pytorch-lightning-integration', id='PYTOR-63')
exp['some_external_metric'].log(0.92)
```

或者获取一个单独的实验并用训练后计算的一些外部度量来更新它:

```py
exp = project.get_experiments(id='PYTOR-63')[0]
exp.log_metric('some_external_metric', 0.92)

```

如你所见，你可以从 Pytorch Lightning 将很多东西记录到 Neptune。

如果你想深入了解这个问题:

## 最后的想法

Pytorch Lightning 是一个很棒的库，可以帮助您:

*   组织你的深度学习代码，让其他人容易理解，
*   将开发样板外包给经验丰富的工程师团队，
*   访问大量最先进的功能，几乎不需要修改您的代码

借助 Neptune integration，您可以免费获得一些额外的东西:

*   你可以监控和跟踪你的深度学习实验
*   你可以很容易地与其他人分享你的研究
*   您和您的团队可以访问实验元数据并更有效地协作。

希望有了这种力量，你将确切地知道你(和其他人)尝试了什么，你的深度学习研究将以闪电般的速度前进

## 完整的 PyTorch 闪电追踪脚本

```py
pip install --upgrade torch pytorch-lightning>=1.5.0
    neptune-client
    matplotlib scikit-plot

```

```py
import os

import numpy as np
import neptune.new as neptune
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl

MAX_EPOCHS=15
LR=0.02
BATCHSIZE=32
CHECKPOINTS_DIR = 'my_models/checkpoints'

class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()

        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train/loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val/loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        fig = plt.figure()
        losses = np.stack([x['val_loss'].numpy() for x in outputs])
        plt.hist(losses)
        neptune_logger.experiment['imgs/loss_histograms'].upload(neptune.types.File.as_image(fig))

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test/loss', loss)
        return {'test_loss': loss}

    def test_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=LR)

    def train_dataloader(self):

        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)

    def val_dataloader(self):

        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)

    def test_dataloader(self):

        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)
from pytorch_lightning.loggers.neptune import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration",
    tags=["pytorch-lightning", "mlp"],
)
model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)

from pytorch_lightning import Trainer

model = CoolSystem()
trainer = Trainer(max_epochs=MAX_EPOCHS,
                  logger=neptune_logger,
                  checkpoint_callback=model_checkpoint,
                  )
trainer.fit(model)
trainer.test(model)

import numpy as np

model.freeze()
test_loader = DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=256)

y_true, y_pred = [],[]
for i, (x, y) in enumerate(test_loader):
    y_hat = model.forward(x).argmax(axis=1).cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    y_true.append(y)
    y_pred.append(y_hat)

    if i == len(test_loader):
        break
y_true = np.hstack(y_true)
y_pred = np.hstack(y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
neptune_logger.experiment['test/accuracy'].log(accuracy)

from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
neptune_logger.experiment['confusion_matrix'].log(File.as_image(fig))

neptune_logger.experiment('checkpoints').upload(CHECKPOINTS_DIR)

neptune_logger.experiment.stop()

```