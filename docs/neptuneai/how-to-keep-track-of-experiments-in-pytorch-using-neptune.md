# 如何在 PyTorch 中跟踪实验

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-keep-track-of-experiments-in-pytorch-using-neptune>

机器学习开发看起来很像传统的软件开发，因为它们都需要我们编写大量代码。但其实不是！让我们通过一些要点来更好地理解这一点。

*   机器学习代码不会抛出错误(当然我说的是语义)，原因是，即使你在一个[神经网络](/web/20221206010849/https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications)中配置了一个错误的方程，它仍然会运行，但会打乱你的预期。用[安德烈·卡帕西](https://web.archive.org/web/20221206010849/https://karpathy.github.io/2019/04/25/recipe/)、*的话说，“神经网络无声无息地失灵”。*
*   机器学习代码/项目严重依赖结果的可重复性。这意味着如果一个[超参数](/web/20221206010849/https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020)被推动或者训练数据发生变化，那么它会在许多方面影响模型的性能。这意味着你必须记下超参数和训练数据的每一个变化，以便能够重现你的工作。
    当网络很小的时候，这可以在一个文本文件中完成，但是如果是一个有几十或几百个超参数的大项目呢？文本文件现在不那么容易了吧！
*   机器学习项目复杂性的增加意味着复杂分支的增加，必须对其进行跟踪和存储以供将来分析。
*   机器学习也需要大量的计算，这是有代价的。你肯定不希望你的云成本暴涨。

有组织地跟踪实验有助于解决所有这些核心问题。海王星是一个完整的[工具，帮助个人和团队顺利跟踪他们的实验。](/web/20221206010849/https://neptune.ai/blog/best-ml-experiment-tracking-tools)它提供了许多功能和演示选项，有助于更轻松地跟踪和协作。

[https://web.archive.org/web/20221206010849if_/https://www.youtube.com/embed/he2KqU76ays?feature=oembed](https://web.archive.org/web/20221206010849if_/https://www.youtube.com/embed/he2KqU76ays?feature=oembed)

视频

## 用海王星进行实验跟踪

传统的跟踪过程包括将日志对象保存为文本或 CSV 文件，这非常方便，但是对于与输出日志的混乱结构相关的未来分析没有用处。下图以图片的形式讲述了这个故事:

虽然可读，但你会很快失去兴趣。一段时间后，你也可能会丢失文件——没有人会预料到突然的磁盘故障或过度清理！

所以，总而言之，txt 的方式方便但不推荐。为了解决这个问题，Neptune 跟踪每个超参数，包括模型和训练程序的超参数，以便您可以有效地与您的团队交流，并在未来分析训练程序以进一步优化它们。

下面是一个类似的实验，但使用 Neptune 应用程序进行跟踪:

<https://web.archive.org/web/20221206010849im_/https://neptune.ai/wp-content/uploads/Example-dashboard-metadata-structure.mp4>

*[Example project metadata in Neptune](https://web.archive.org/web/20221206010849/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/e/TFKERAS-14/all)* 

### 在 Pytorch 建立海王星实验

设置过程很简单。首先注册一个帐户 [](https://web.archive.org/web/20221206010849/https://neptune.ai/) [这里](/web/20221206010849/https://neptune.ai/register)，这将创建一个唯一的 ID 和仪表板，在那里你可以看到你所有的实验。您可以随时添加您的团队成员，并在实验中进行合作。按照[这些步骤](https://web.archive.org/web/20221206010849/https://docs.neptune.ai/getting-started/installation)获得您的唯一 id(设置时使用)。

为了在 python 的培训过程中使用这个仪表板，Neptune 开发人员开发了一个易于使用的包，您可以通过 pip 安装它:

```py
pip install neptune-client

```

完成安装后，您需要像这样初始化 Neptune:

```py
import neptune.new as neptune
from neptune.new.types import File

NEPTUNE_API_TOKEN = "<api-token-here>"
run = neptune.init(project='<username>/Experiment-Tracking-PyTorch',
                   api_token=NEPTUNE_API_TOKEN)

```

现在，让我们看看如何从 PyTorch 脚本中利用 Neptune 的仪表板。

#### 基本指标集成

让我们从跟踪通常的度量开始，比如训练/测试损失、时期损失和梯度。为此，您只需将 run['metrics/train_loss']。log(loss ),其中“metrics”是可以存储所需参数的目录,“loss”是被跟踪的度量。在你的 PyTorch 训练循环中会是这样的:

```py
def train(model, device, train_loader, optimizer, epoch):
   model.train()
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(device), target.to(device)

       optimizer.zero_grad()

       output = model(data)

       loss = F.nll_loss(output, target)
       loss.backward()

       optimizer.step()

       run['metrics/train_loss'].log(loss)

       if batch_idx % 100 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))
```

运行上面的代码后，检查您的 Neptune 仪表板，您将看到跟踪和绘制的损失度量供您分析。

为了跟踪超参数(你应该，总是这样做！)你需要做的就是简单的给海王星的 run 对象添加这样的参数。

```py
PARAMS = {'batch_size_train': 64,
         'batch_size_test': 1000,
         'momentum': 0.5,
         'learning_rate': 0.01,
         'log_interval' : 10,
         'optimizer': 'Adam'}

run['parameters'] = PARAMS 

```

使用这些更改再次运行实验后，您将在仪表板中看到所有参数，如下所示:

添加参数和标记的一个最大目的是将所有东西都插入一个仪表板，这样就可以在未来轻松地进行优化或特性更改分析，而不会消耗大量代码。

#### 高级选项

Neptune 为您提供了许多定制选项，您可以简单地记录更多特定于实验的内容，如图像预测、模型权重、性能图表等等。

所有这些功能都可以很容易地集成到您当前的 PyTorch 脚本中，在下一节中，我将向您展示如何充分利用 Neptune。

运行实验时，您可以记录其他有用的信息:

*   代码:快照脚本、jupyter 笔记本、配置文件等等
*   **超参数**:日志学习率，历元数，以及其他
*   **属性**:日志数据位置、数据版本或其他
*   **标签**:添加“resnet50”或“无增强”等标签来组织您的跑步。
*   名称:每个实验都应该有一个有意义的名称，所以我们不要每次都用“默认”

只需将这些作为参数传递给 init()函数，这很简单:

```py
NEPTUNE_API_TOKEN = "<api-token-here>"
run = neptune.init(project='<username>/Experiment-Tracking-PyTorch',
                   api_token=NEPTUNE_API_TOKEN,
                  tags=['classification', 'pytorch', 'neptune'],
                   source_files=["**/*.ipynb", "*.yaml"]  
)

```

上面的代码摘录将上传您的属于正则表达式的代码文件，添加您可以在仪表板中识别的标签。现在，让我们看看如何记录其他实验特定的内容，如图像和模型重量文件:

#### 记录图像

```py
def train(model, device, train_loader, optimizer, epoch):
   model.train()
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(device), target.to(device)

       optimizer.zero_grad()

       output = model(data)

       loss = F.nll_loss(output, target)
       loss.backward()

       optimizer.step()

       run['metrics/train_loss'].log(loss)

       if batch_idx % 50 == 1:
           for image, prediction in zip(data, output):

               img = image.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
               img = Image.fromarray(img.reshape(28,28))
                     run["predictions/{}".format(batch_idx)].upload(File.as_image(img))

       if batch_idx % 100 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))
```

在运行带有日志记录更改的代码后，您将在 Neptune 仪表板的“预测”目录中看到记录的图像。

通过左侧菜单中的“添加新仪表板”，您可以使用许多可用的小部件创建自己的定制仪表板。例如，您可以在一个屏幕上添加图像预测并进行分析！

在这里，我添加了其中的两个，你可以通过选择你更感兴趣的来扩展它。

#### 你可以在实验中记录额外的东西

训练中可以记录很多有趣的信息。您可能对监控以下内容感兴趣:

*   每个时期后的模型预测(考虑预测遮罩或覆盖的边界框)
*   诊断图表，如 ROC AUC 曲线或混淆矩阵
*   模型检查点或其他对象

例如，我们可以使用 torch.save()方法将模型权重和配置保存到本地磁盘以及 Neptune 的仪表板中:

```py
torch.save(model.state_dict(), 'model_dict.ckpt')

run["model_checkpoints"].upload("model_dict.ckpt")
```

至于像 ROC 曲线和混淆矩阵这样的训练后分析，您可以使用您最喜欢的绘图库绘制它，并用 neptune.log_image()记录它

```py
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
...
fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
run["metrics/confusion_matrix"].upload(File.as_image(fig))

```

如果您希望看到这个令人敬畏的 API 的每一个功能，请前往包含代码示例的 Neptune 文档。

## 你已经到达终点了！

我们看到了**为什么** *[实验跟踪](/web/20221206010849/https://neptune.ai/experiment-tracking)* 由于它们无声的脆弱性和未来的分析前景，在机器学习系统中是一种必需品。我们也看到了**Neptune ai 如何证明是完成这项任务的合适工具。使用 Neptune 的 API:**

*   你可以监控和跟踪你的深度学习实验
*   你可以很容易地与其他人分享你的研究
*   您和您的团队可以访问实验元数据并更有效地协作。

你可以在这个笔记本中找到代码[这里](https://web.archive.org/web/20221206010849/https://colab.research.google.com/drive/10CN6HsPcPU9shiPs6y3cdehFr-FWAaCD?usp=sharing)和海王星实验[这里](https://web.archive.org/web/20221206010849/https://app.neptune.ai/theaayushbajaj/Experiment-Tracking-PyTorch/experiments?split=tbl&dash=charts&viewId=standard-view)。

目前就这些，敬请关注更多！再见！