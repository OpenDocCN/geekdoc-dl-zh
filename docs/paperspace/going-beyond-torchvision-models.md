# 超越火炬视觉模型

> 原文：<https://blog.paperspace.com/going-beyond-torchvision-models/>

ResNets、DenseNets 和 Inception networks 无疑是用于执行图像分类和对象识别的一些最强大的模型。这些模型已经在 ImageNet 大规模视觉识别挑战(ILSVRC)中显示出一些有希望的结果，并且已经达到了超越人类的程度。

Pytorch，脸书用于研究和生产的深度学习基础设施，有一个名为 Torchvision 的库，主要用于计算机视觉任务，它为我们提供了所有这些在 ImageNet 数据集上训练的令人难以置信的模型。

我们可以利用这些现有的规范模型，使用一种称为迁移学习的技术来执行图像分类和检测，以适应我们的问题。查看这些模型的评估指标，我们可以发现这些模型虽然强大，但离完美的准确性还有一些距离。计算机视觉研究人员真正推动了尽可能精确的建筑模型的界限，甚至超越了 ResNets 和 DenseNets，但我们还没有看到 Torchvision 模型模块的任何更新。这就是本文试图解决的问题——访问尚未添加到 Torchvision 框架中的模型。

非常感谢 GitHub 知识库的作者[https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)在 Pytorch 中实现所有这些在 torchvision 框架中不可用的模型方面所做的伟大工作。下面是整篇文章的快速概述。

1.  安装必要的库
2.  获取我们的模型。
3.  使用迁移学习在 cifar-10 数据集上训练其中一个模型
4.  将我们的模型与类似的火炬视觉模型进行评估和比较。

### 装置

有两种方法可以安装所需的模块——从 GitHub 库下载或使用 pip install。我们将首先通过 pip 安装来安装该模块。这比你想象的要简单得多。只需启动您的终端并输入命令:

```py
pip install pretrainedmodels 
```

仅此而已。让我们看看如何通过克隆到存储库中来安装 pretrainedmodels 模块。也很简单。只需启动 git cmd 或任何其他终端，并使用以下命令将这些模型的实现克隆到 GitHub 存储库中:

```py
git clone https://github.com/Cadene/pretrained-models.pytorch 
```

在终端中，进入克隆的目录并输入命令:

```py
 python setup.py install 
```

这应该会安装 pretrainedmodels 模块。要验证这一点，请打开任何 python IDE 或更好的 Jupyter notebook，并导入 pretrainedmodels 模块，代码如下:

```py
import pretrainedmodels 
```

如果我们没有得到任何错误，我们的模块已经正确安装。我们应该注意，该模块不包括模型的重量。当我们获得模型时，重量将被自动下载。

### 获取我们的模型

在我们为分类选择首选模型之前，让我们看看 pretrainedmodels 模块中可供我们选择的无穷无尽的模型列表。让我们看看实现这一点的代码。

```py
print(pretrainedmodels.model_names) 
```

这应该会打印出 pretrainedmodels 模块中所有可用模型的单一列表。我们还可以通过运行包含以下代码的 Jupyter 笔记本单元来查看每个型号的配置:

```py
print(pretrainedmodels.pretrained_settings["name of the model"]) 
```

这会以字典形式打印出一些关于模型的信息，比如模型权重的 URL 路径、用于归一化输入图像的平均值和标准偏差、输入图像大小等等。在本文中，我们将使用 se_resnet50 模型。如果你想了解更多关于这些模型的架构和性能，那么你应该看看这篇文章:[https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)。几乎每个机器学习模型都需要数据来进行训练。在本文中，我们将使用 torchvision 框架中包含的 Cifar-10 数据集。我们将通过一个简单的管道从 torchvision 框架加载数据，但在此之前，我们需要导入一些重要的库。

```py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import copy
import time 
```

导入所需的库后，我们可以继续创建管道来加载数据集。

```py
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)
dataset_sizes = {
  'train' : len(trainset),
  'val' : len(testset)
}
dataloaders = {
              'train': trainloader,
              'val' : testloader 
```

在接下来的步骤中，我们将构建和训练我们的模型。我们将使用面向对象的编程风格，这是一种构建 PyTorch 模型的传统方式。在您继续下面的部分之前，我建议您在 Pytorch 官方页面上学习这个 60 分钟的 blitz 教程:[https://py torch . org/tutorials/beginner/deep _ learning _ 60min _ blitz . html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)但是如果您仍然选择继续，不要担心，我会尽最大努力在代码中的评论中解释代码的每一点。让我们看看这样做的代码。

```py
 class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Obtain the desired model from the pretrainedmodels library

        self.model = pretrainedmodels.__dict__['se_resnet50']()

        # Build our classifier and since we are classifying the images into 10 
        # classes, we return a 10-dimensional vector as the output.

        self.classifier = nn.Sequential(
        nn.Linear(self.model.last_linear.in_features,10),
        nn.LogSoftmax(dim=1))

        # Requires_grad = False denies the se_resnet50 base model the ability to 
        # update its parameters hence make it unable to train.

        for params in self.model.parameters():
            params.requires_grad = False

            # We replace the fully connected layers of the base model (se_resnet model) 
            # which served as the classifier with our custom trainable classifier.

        self.model.last_linear= self.classifier

# Every model which inherits from nn.Module requires that we override the  forward
# function which defines the forward pass computation performed at every call.

def forward(self, x):
    # x is our input data
    return self.model(x) 
```

每次调用 init 函数时，都会创建我们的模型。现在我们已经准备好了模型，我们可以开始训练它了。在 Model 类中，我们可以定义另一个名为 fit 的函数，该函数将调用我们在一批图像上覆盖的 forward 函数，然后通过模型反向传播我们的错误以进行权重更新。让我们构建软件管道来执行这个向前传播和向后传播任务。

```py
 def fit(self, dataloaders, num_epochs):

    # We check whether a gpu is enabled for our environment.

    train_on_gpu = torch.cuda.is_available()

    # We define our optimizer and pass in the model parameters (weights and biases) 
    # into the constructor of the optimizer we want. 
    # More info: https://pytorch.org/docs/stable/optim.html

    optimizer = optim.Adam(self.model.last_linear.parameters())

    # Essentially what scheduler does is to reduce our learning by a certain factor 
    # when less progress is being made in our training.

    scheduler = optim.lr_scheduler.StepLR(optimizer, 4)

    # Criterion is the loss function of our model. 
    # We use Negative Log-Likelihood loss because we used log-softmax as the last layer of our model. 
    # We can remove the log-softmax layer and replace the nn.NLLLoss() with nn.CrossEntropyLoss()

    criterion = nn.NLLLoss()
    since = time.time()

    # model.state_dict() is a dictionary of our model's parameters. What we did here 
    # is to deepcopy it and assign it to a variable

    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_acc = 0.0

    # We check if a gpu is enabled for our environment and move our model to the gpu

    if train_on_gpu:
        self.model = self.model.cuda()
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase. 
        # We iterate through the training set and validation set in every epoch.

        for phase in ['train', 'val']:

            # we apply the scheduler to the learning rate in the training phase since 
            # we don't train our model in the validation phase

            if phase == 'train':
                scheduler.step()
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   #Set model to evaluate mode to turn off features like dropout
            running_loss = 0.0
            running_corrects = 0

            # Iterate over batches of train and validation data.

            for inputs, labels in dataloaders[phase]:
                if train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # clear all gradients since gradients get accumulated after every 
                iteration.
                optimizer.zero_grad()

                # track history if only in training phase

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                # calculates the loss between the output of our model and ground-truth
                labels

                loss = criterion(outputs, labels)

                # perform backpropagation and optimization only if in training phase

                if phase == 'train':

                # backpropagate gradients from the loss node through all the 
                parameters
                    loss.backward()

                    # Update parameters(Weighs and biases) of our model using the 
                    gradients.

                    optimizer.step()
              # Statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model if we obtain a better validation accuracy than the previous one.

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model parameters and return it as the final trained model.

    self.model.load_state_dict(best_model_wts)
    return self.model

# We instantiate our model class
model = Model()
# Run 10 training epochs on our model
model_ft = model.fit(dataloaders, 10) 
```

对 se_resnet50 进行 10 个历元的训练，我们达到了 86.22%的验证准确率。

![](img/0be368bfe7ad856afdeaa3b2172567a3.png)

best validation accuracy for se_resnet50

我们还拟合了 torchvision 提供的 resnet50 模型，其中 se_resnet50 是 cifar-10 数据集上 10 个时期的变体。在训练结束时，我们获得了 34.90%的准确率，这是非常差的。很明显，我们可以看到 resnet50 模型并不适合这个问题，但它的一个变体(已经在 pretrainedmodels 库中实现)在这个问题上表现得非常好。

### 后续步骤

下一步，我鼓励读者尝试 pretrainedmodels 库中提供的所有模型，看看是否有任何 pretrainedmodels 能够带来性能上的改进。

### 关于作者

我是一名本科生，目前在读电气电子工程。我也是一个深度学习爱好者和作家。我的工作主要集中在计算机视觉在医学图像分析中的应用。我希望有一天能打入自动驾驶汽车领域。你可以在推特(@henryansah083)上关注:[https://twitter.com/henryansah083?s=09](https://twitter.com/henryansah083?s=09)LinkedIn:[https://www.linkedin.com/in/henry-ansah-6a8b84167/](https://www.linkedin.com/in/henry-ansah-6a8b84167/)