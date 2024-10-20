# Pix2pix:关键模型架构决策

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pix2pix-key-model-architecture-decisions>

[生成对抗网络或 GANs](/web/20221117203630/https://neptune.ai/blog/generative-adversarial-networks-gan-applications) 是一种属于无监督学习类的神经网络。它用于深度生成建模的任务。

在深度生成建模中，深度神经网络学习一组给定数据点上的概率分布，并生成相似的数据点。由于这是一个无人监督的学习任务，它在学习过程中不使用任何标签。

自 2014 年发布以来，深度学习社区一直在积极开发新的 gan，以改善生成建模领域。本文旨在提供关于 GAN 的信息，特别是 Pix2Pix GAN，它是最常用的生成模型之一。

## 甘是什么？

GANs 由 Ian Goodfellow 在 2014 年设计。GANs 的主要意图是生成不模糊且具有丰富特征表示的样本。判别模型在这方面做得很好，因为它们能够在不同的类别之间进行分类。另一方面，深度生成模型的效率要低得多，因为在自动编码器中，很难近似许多棘手的概率计算。

自动编码器及其变体是显式似然模型，这意味着它们显式计算给定分布的概率密度函数。gan 及其变体是隐式似然模型，这意味着它们不计算概率密度函数，而是学习潜在的分布。

gan 通过将整个问题作为一个二进制分类问题来处理，来学习底层分布。在这种方法中，问题模型由两个模型表示:生成器和鉴别器。生成器的工作是生成新的样本，鉴别器的工作是分类或鉴别生成器生成的样本是真是假。

这两个模型在零和游戏中一起训练，直到生成器可以产生与真实样本相似的样本。或者换句话说，他们被训练到生成器可以骗过鉴别器。

### 香草甘的架构

先简单了解一下 GANs 的架构。从这一节开始，大多数主题将使用代码进行解释。首先，让我们定义所有需要的依赖项。

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
```

#### 发电机

生成器是 GAN 中的一个组件，它接收定义为高斯分布的噪声，并产生与原始数据集相似的样本。随着 GANs 多年来的发展，他们已经采用了在计算机视觉任务中非常突出的 CNN。但是为了简单起见，我们将使用 Pytorch 用线性函数来定义它。

```py
class Generator(nn.Module):
   def __init__(self, z_dim, img_dim):
       super().__init__()
       self.gen = nn.Sequential(
           nn.Linear(z_dim, 256),
           nn.LeakyReLU(0.01),
           nn.Linear(256, img_dim),
           nn.Tanh(),  
       )

   def forward(self, x):
       return self.gen(x)
```

#### 鉴别器

鉴别器只是一个分类器，它对生成器生成的数据是真是假进行分类。它通过从真实数据中学习原始分布，然后在两者之间进行评估来实现这一点。我们将保持简单，使用线性函数定义鉴别器。

```py
class Discriminator(nn.Module):
   def __init__(self, in_features):
       super().__init__()
       self.disc = nn.Sequential(
           nn.Linear(in_features, 128),
           nn.LeakyReLU(0.01),
           nn.Linear(128, 1),
           nn.Sigmoid(),
       )

   def forward(self, x):
       return self.disc(x)
```

生成器和鉴别器的关键区别是最后一层。前者产生与图像相同的形状，而后者只产生一个输出，0 或 1。

#### 损失函数和训练

损失函数是任何深度学习算法中最重要的组件之一。例如，如果我们设计一个 CNN 来最小化真实情况和预测结果之间的欧几里德距离，它将倾向于产生模糊的结果。**这是因为欧几里德距离通过平均所有可能的输出而最小化，这导致了模糊**。

以上这一点很重要，我们必须记住。也就是说，我们将用于普通 GAN 的损失函数将是二进制交叉熵损失或 BCELoss，因为我们正在执行二进制分类。

```py
criterion = nn.BCELoss()
```

现在我们来定义一下**优化**方法以及其他相关参数。

```py
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  
batch_size = 32
num_epochs = 100

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0
```

我们来理解一下训练循环。甘的**训练循环开始于:**

1.  使用高斯分布从生成器生成样本
2.  使用生成器产生的真实数据和虚假数据训练鉴别器
3.  更新鉴别器
4.  更新生成器

下面是训练循环的样子:

```py
for epoch in range(num_epochs):
   for batch_idx, (real, _) in enumerate(loader):
       real = real.view(-1, 784).to(device)
       batch_size = real.shape[0]

       noise = torch.randn(batch_size, z_dim).to(device)
       fake = gen(noise)
       disc_real = disc(real).view(-1)
       lossD_real = criterion(disc_real, torch.ones_like(disc_real))
       disc_fake = disc(fake).view(-1)
       lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
       lossD = (lossD_real + lossD_fake) / 2
       disc.zero_grad()
       lossD.backward(retain_graph=True)
       opt_disc.step()

       output = disc(fake).view(-1)
       lossG = criterion(output, torch.ones_like(output))
       gen.zero_grad()
       lossG.backward()
       opt_gen.step()
       if batch_idx == 0:
           print(
               f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)}
                     Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
           )
           with torch.no_grad():
               fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
               data = real.reshape(-1, 1, 28, 28)
               img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
               img_grid_real = torchvision.utils.make_grid(data, normalize=True)
               writer_fake.add_image(
                   "Mnist Fake Images", img_grid_fake, global_step=step
               )
               writer_real.add_image(
                   "Mnist Real Images", img_grid_real, global_step=step
               )
               step += 1

```

以上循环的要点:

1.  鉴别器的**损失函数计算两次:一次用于真实图像，另一次用于虚假图像。**
    *   对于**实像**，地面真实被转换成使用 torch.ones_like 函数的真实，该函数返回定义形状的一个矩阵。
    *   对于**假图像**，使用 torch.zeros_like 函数将地面真实转换为一，该函数返回定义形状的零矩阵。
2.  发电机的**损失函数只计算一次。如果你仔细观察，鉴别器使用相同的损失函数来计算假图像的损失。唯一的区别是不使用 torch.zeros_like 函数，而是使用 torch.ones_like 函数。标签从 0 到 1 的互换使得生成器能够学习将产生真实图像的表示，因此欺骗了鉴别器。**

数学上，我们可以将整个过程定义为:

其中 Z 是噪声，x 是真实数据，G 是发生器，D 是鉴频器。

### GANs 的应用

gan 广泛用于:

*   **生成训练样本:** GANs 通常用于生成特定任务的样本，如恶性和良性癌细胞的分类，特别是在数据较少的情况下训练分类器。
*   人工智能艺术或生成艺术:人工智能或生成艺术是 GANs 被广泛使用的另一个新领域。自从引入不可替代的代币以来，全世界的艺术家都在以非正统的方式创作艺术，即数字化和生成性。像 DeepDaze，BigSleep，BigGAN，CLIP，VQGAN 等 GAN 是创作者最常用的。

![AI Art or Generative Art](img/d7a5d91c9577ab401b88bc9be59e753e.png)

*AI Art or Generative Art | Source: Author*

*   **图像到图像的翻译**:图像到图像的翻译再次被数字创作者使用。这里的想法是将某种类型的图像转换成目标域中的图像。例如，将日光图像转换为夜间图像，或将冬季图像转换为夏季图像(见下图)。像 pix2pix、cycleGAN、styleGAN 这样的 GAN 是少数几个最受欢迎的 GAN。

![Image-to-image translation](img/110111292a85f48705c9fb848cec3330.png)

*Image-to-image translation | [Source](https://web.archive.org/web/20221117203630/https://research.nvidia.com/publication/2017-12_Unsupervised-Image-to-Image-Translation)*

*   文本到图像的翻译:文本到图像的翻译就是将文本或给定的字符串转换成图像。这是一个非常热门的领域，而且是一个不断发展的社区。如前所述，来自 OpenAI 的 DeepDaze、BigSleep 和 DALL E 等 GANs 在这方面非常受欢迎。

![Text-to-image translation](img/2f993b3819f733cf6fad298cb8fddf9e.png)

*Text-to-image translation | [Source](https://web.archive.org/web/20221117203630/https://openai.com/blog/dall-e/) *

### 甘的问题

虽然 GANs 可以从随机高斯分布中产生与真实图像相似的图像，但这个过程在大多数时候并不完美。原因如下:

*   **模式崩溃:**模式崩溃是指生成器能够通过从整体数据中学习较少的数据样本来欺骗鉴别器的问题。由于模式崩溃，GAN 不能学习多种分布，并且仍然局限于少数几种。
*   **递减梯度**:递减或**消失梯度下降发生在网络的导数非常小时，以至于对原始权重的更新几乎可以忽略不计。为了克服这个问题，建议使用 WGANs。**
*   **不收敛:**它发生在网络无法收敛到全局最小值的时候。这是不稳定训练的结果。这个问题可以通过光谱归一化来解决。你可以在这里阅读光谱归一化。

### GAN 的变体

自从第一个 GAN 发布以来，已经出现了许多 GAN 的变体。以下是一些最受欢迎的 GANs:

*   CycleGAN
*   StyleGAN
*   像素网络
*   文本 2 图像
*   迪斯科根
*   伊斯甘

**本文只关注 Pix2Pix GAN** 。在下一节中，我们将了解一些关键组件，如架构、损失函数等。

## Pix2Pix GAN 是什么？

Pix2Pix GAN 是由 [Phillip Isola](https://web.archive.org/web/20221117203630/http://web.mit.edu/phillipi/) 等人开发的有条件 GAN ( [cGAN](https://web.archive.org/web/20221117203630/https://golden.com/wiki/Conditional_generative_adversarial_network_(cGAN)) )，与只使用真实数据和噪声来学习和生成图像的 vanilla GAN 不同，cGAN 使用真实数据、噪声以及标签来生成图像。

本质上，生成器从真实数据以及噪声中学习映射。

类似地，鉴别器也从标签和真实数据中学习表示。

此设置使 cGAN 适用于图像到图像的转换任务，其中生成器根据输入图像生成相应的输出图像。换句话说，生成器使用条件分布(或数据)如指导或蓝图来生成目标图像(见下图)。

![Pix2Pix is a conditional GAN](img/0eb83a50967a9252ef645a45aa6dd469.png)

*Pix2Pix is a conditional GAN | Source: Author*

![Application of Pix2Pix](img/437cd39edad97b86674b85c309244fad.png)

*Application of Pix2Pix | [Source](https://web.archive.org/web/20221117203630/https://phillipi.github.io/pix2pix/)*

Pix2Pix 的想法依赖于为训练提供的数据集。将图像翻译与训练样本{x，y}配对是一对，它们之间具有对应关系。

## Pix2Pix 网络架构

pix2pix 有两个重要的架构，一个用于生成器，另一个用于鉴别器，即 U-net 和 patchGAN。让我们更详细地探讨一下这两个问题。

### u 网生成器

如前所述，pix2pix 使用的架构称为 U-net。U-net 最初是由 Ronneberger 等人为生物医学图像分割而开发的。艾尔。2015 年。

UNet 由两个主要部分组成:

1.  由卷积层(左侧)组成的**收缩路径**，在提取信息的同时对数据进行下采样。
2.  由上转置卷积层(右侧)组成的**扩展路径**对信息进行上采样。

假设我们的下采样有三个卷积层 C_l(1，2，3)，那么我们必须确保我们的上采样有三个转置卷积层 C_u(1，2，3)。这是因为我们想要使用**跳过连接**来连接相同大小的相应块。

向下采样

![Skip connection](img/af0051f7203cf15d747d250cf49d1c3b.png)

*Skip connection | Source: Author*

#### 在下采样期间，每个卷积块提取空间信息，并将该信息传递给下一个卷积块以提取更多信息，直到它到达被称为**瓶颈**的中间部分。上采样从瓶颈开始。

上采样

#### 在上采样期间，每个转置卷积块扩展来自前一块的信息，同时连接来自相应下采样块的信息。通过连接信息，网络可以学习根据这些信息组合更精确的输出。

这种架构能够定位，即，它能够逐个像素地找到感兴趣的对象。此外，UNet 还允许网络将上下文信息从较低分辨率层传播到较高分辨率层。这允许网络产生高分辨率样本。

马尔可夫鉴别器

### 鉴频器采用贴片 GAN 架构。该架构包含多个转置卷积模块。它取图像的一个 NxN 部分，并试图发现它是真的还是假的。n 可以是任意大小。它可以比原始图像小，但仍然能够产生高质量的结果。鉴别器在整个图像上卷积应用。此外，因为鉴别器更小，即与发生器相比它具有更少的参数，所以它实际上更快。

PatchGAN 可以有效地将图像建模为马尔可夫随机场，其中 NxN 被视为独立的面片。所以 PatchGAN 可以理解为一种质感/风格的丧失。

损失函数

### 损失函数是:

上面的等式有两个部分:一个用于鉴别器，另一个用于发生器。让我们一个一个的了解他们两个。

在任何 GAN 中，在每次迭代中首先训练鉴别器，以便它可以识别真实和虚假数据，从而可以在它们之间进行鉴别或分类。本质上，

**D(x，y) = 1 即实数和，**

**D(x，G(z)) = 0 即伪。**

值得注意的是，G(z)也将产生假样本，因此其值将更接近于零。理论上，鉴别器应该总是只将 G(z)分类为零。因此鉴别器应该在每次迭代中保持真实和虚假之间的最大距离，即 1 和 0。换句话说，鉴别器应该最大化损失函数。

在鉴别器之后，发生器被训练。生成器，即 G(z)应该学习产生更接近真实样本的样本。为了学习原始分布，它从鉴别器获得帮助，即，我们将 D(x，G(z)) = 1 而不是 D(x，G(z)) = 0。

随着标记的改变，发生器现在根据属于具有基本事实标记的鉴别器的参数来优化其参数。该步骤确保发生器现在可以产生接近真实数据的样本，即 1。

损失函数也与 L1 损失混合，使得发生器不仅欺骗鉴别器，而且产生接近地面真实的图像。本质上，损耗函数对发电机有一个额外的 L1 损耗。

因此，最终损失函数为:

值得注意的是，L1 损失能够保留图像中的低频细节，但它将无法捕捉高频细节。因此，它仍然会产生模糊的图像。为了解决这个问题，使用了 PatchGAN。

最佳化

### 优化和训练过程类似于香草甘。但是训练本身是一个困难的过程，因为 GAN 的目标函数更多地是凹-凹的而不是凸-凹的。正因为如此，很难找到一个鞍点，这就是为什么训练和优化 GANs 很困难。

正如我们之前看到的，生成器不是直接训练的，而是通过鉴别器训练的。这实质上限制了发电机的优化。如果鉴别器未能捕获高维空间，那么可以肯定的是，生成器将不能产生好的样本。另一方面，如果我们能够以一种更加优化的方式训练鉴别器，那么我们就可以保证生成器也将得到优化的训练。

在训练的早期阶段，G 未经训练，产生好样本的能力较弱。这使得鉴别器非常强大。因此，不是将 log(1D(G(z))最小化，而是将发生器训练为将 log D(G(z))最大化。这在训练的早期阶段创造了某种稳定性。

解决不稳定性的其他方法有:

在模型的每一层使用**光谱归一化**

1.  使用 **Wasserstein 损失**计算真实或虚假图像的平均分数。
2.  Pix2Pix 动手示例

## 让我们用 PyTorch 对 Pix2Pix 进行编码，直观地了解它的工作原理及其背后的各种组件。本节将让您清楚地了解 Pix2Pix 是如何工作的。

让我们从下载数据开始。以下代码可用于下载数据。

数据可视化

```py
!wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
!tar -xvf facades.tar.gz
```

### 一旦下载了数据，我们就可以将它们可视化，以了解根据需求格式化数据所需的必要步骤。

我们将导入以下库进行数据可视化。

从上面的图像中，我们可以看到数据有两个图像连接在一起。如果我们看到上面的图像的形状，我们发现宽度是 512，这意味着图像可以很容易地分成两部分。

```py
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

path = '/content/facades/train/'
plt.imshow(cv2.imread(f'{path}91.jpg'))
```

![Data visualization](img/44a072ab90851b332a00b6797c389192.png)

*Source: Author*

*> >图像的形状:(256，512，3)*

```py
print('Shape of the image: ',cv2.imread(f'{path}91.jpg').shape)
```

为了分离图像，我们将使用以下命令:

左边的图像将是我们的基础真理，而右边的图像将是我们的条件图像。我们将它们分别称为 y 和 x。

```py
image = cv2.imread(f'{path}91.jpg')
w = image.shape[1]//2
image_real = image[:, :w, :]
image_cond = image[:, w:, :]
fig, axes = plt.subplots(1,2, figsize=(18,6))
axes[0].imshow(image_real, label='Real')
axes[1].imshow(image_cond, label='Condition')
plt.show()
```

![Data visualization](img/cc020425c470b18ef2ec4574719ca462.png)

*Source: Author*

创建数据加载器

### Dataloader 是一个允许我们按照 PyTorch 要求格式化数据的功能。这将包括两个步骤:

1.格式化数据，即从源中读取数据，裁剪数据，然后将其转换为 Pytorch 张量。

2.在将数据输入神经网络之前，使用 Pytorch 的 DataLoader 函数加载数据以创建批处理。

```py
class data(Dataset):
   def __init__(self, path='/content/facades/train/'):
       self.filenames = glob(path+'*.jpg')

   def __len__(self):
       return len(self.filenames)

   def __getitem__(self, idx):
       filename = self.filenames[idx]

       image = cv2.imread(filename)
       image_width = image.shape[1]
       image_width = image_width // 2
       real = image[:, :image_width, :]
       condition = image[:, image_width:, :]

       real = transforms.functional.to_tensor(real)
       condition = transforms.functional.to_tensor(condition)

       return real, condition
```

请记住，我们将为培训和验证创建一个数据加载器。

```py
train_dataset = data()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = data(path='/content/facades/val/')
val_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

Utils

### 在本节中，我们将创建用于构建生成器和鉴别器的组件。我们将创建的组件是用于下采样的卷积函数和用于上采样的转置卷积函数，分别称为 cnn_block 和 tcnn_block。

发电机

```py
def cnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0, first_layer = False):

   if first_layer:
       return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

def tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0, first_layer = False):
   if first_layer:
       return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )
```

### 现在，让我们定义生成器。我们将使用这两个组件来定义相同的。

鉴别器

```py
class Generator(nn.Module):
 def __init__(self,instance_norm=False):
   super(Generator,self).__init__()
   self.e1 = cnn_block(c_dim,gf_dim,4,2,1, first_layer = True)
   self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1,)
   self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1,)
   self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1,)
   self.e5 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e6 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e7 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
   self.e8 = cnn_block(gf_dim*8,gf_dim*8,4,2,1, first_layer=True)

   self.d1 = tcnn_block(gf_dim*8,gf_dim*8,4,2,1)
   self.d2 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d3 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d4 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
   self.d5 = tcnn_block(gf_dim*8*2,gf_dim*4,4,2,1)
   self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1)
   self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1)
   self.d8 = tcnn_block(gf_dim*1*2,c_dim,4,2,1, first_layer = True)
   self.tanh = nn.Tanh()

 def forward(self,x):
   e1 = self.e1(x)
   e2 = self.e2(F.leaky_relu(e1,0.2))
   e3 = self.e3(F.leaky_relu(e2,0.2))
   e4 = self.e4(F.leaky_relu(e3,0.2))
   e5 = self.e5(F.leaky_relu(e4,0.2))
   e6 = self.e6(F.leaky_relu(e5,0.2))
   e7 = self.e7(F.leaky_relu(e6,0.2))
   e8 = self.e8(F.leaky_relu(e7,0.2))
   d1 = torch.cat([F.dropout(self.d1(F.relu(e8)),0.5,training=True),e7],1)
   d2 = torch.cat([F.dropout(self.d2(F.relu(d1)),0.5,training=True),e6],1)
   d3 = torch.cat([F.dropout(self.d3(F.relu(d2)),0.5,training=True),e5],1)
   d4 = torch.cat([self.d4(F.relu(d3)),e4],1)
   d5 = torch.cat([self.d5(F.relu(d4)),e3],1)
   d6 = torch.cat([self.d6(F.relu(d5)),e2],1)
   d7 = torch.cat([self.d7(F.relu(d6)),e1],1)
   d8 = self.d8(F.relu(d7))

   return self.tanh(d8)
```

### 让我们使用下采样函数来定义鉴别器。

定义参数

```py
class Discriminator(nn.Module):
 def __init__(self,instance_norm=False):
   super(Discriminator,self).__init__()
   self.conv1 = cnn_block(c_dim*2,df_dim,4,2,1, first_layer=True) 
   self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)
   self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)
   self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)
   self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)

   self.sigmoid = nn.Sigmoid()
 def forward(self, x, y):
   O = torch.cat([x,y],dim=1)
   O = F.leaky_relu(self.conv1(O),0.2)
   O = F.leaky_relu(self.conv2(O),0.2)
   O = F.leaky_relu(self.conv3(O),0.2)
   O = F.leaky_relu(self.conv4(O),0.2)
   O = self.conv5(O)

   return self.sigmoid(O)
```

### 在本节中，我们将定义参数。这些参数将帮助我们训练神经网络。

初始化模型

```py
batch_size = 4
workers = 2

epochs = 30

gf_dim = 64
df_dim = 64

L1_lambda = 100.0

in_w = in_h = 256
c_dim = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 让我们初始化这两个模型，并启用 CUDA 进行更快的训练。

我们还将定义优化器和损失函数。

```py
G = Generator().to(device)
D = Discriminator().to(device)
```

培养

```py
G_optimizer = optim.Adam(G.parameters(), lr=2e-4,betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(), lr=2e-4,betas=(0.5,0.999))

bce_criterion = nn.BCELoss()
L1_criterion = nn.L1Loss()
```

### 一旦定义了所有重要的函数，我们将初始化训练循环。

监控我们的模型

```py
for ep in range(epochs):
 for i, data in enumerate(train_loader):

   y, x = data
   x = x.to(device)
   y = y.to(device)

   b_size = x.shape[0]

   real_class = torch.ones(b_size,1,30,30).to(device)
   fake_class = torch.zeros(b_size,1,30,30).to(device)

   D.zero_grad()

   real_patch = D(y,x)
   real_gan_loss=bce_criterion(real_patch,real_class)

   fake=G(x)

   fake_patch = D(fake.detach(),x)
   fake_gan_loss=bce_criterion(fake_patch,fake_class)

   D_loss = real_gan_loss + fake_gan_loss
   D_loss.backward()
   D_optimizer.step()

   G.zero_grad()
   fake_patch = D(fake,x)
   fake_gan_loss=bce_criterion(fake_patch,real_class)

   L1_loss = L1_criterion(fake,y)
   G_loss = fake_gan_loss + L1_lambda*L1_loss
   G_loss.backward()

   G_optimizer.step()

   run["Gen Loss"].log(G_loss.item())
   run["Dis Loss"].log(D_loss.item())
   run['L1 Loss'].log(L1_loss.item())
   run['Gen GAN Loss'].log(fake_gan_loss.item())

   torch.save(G.state_dict(), 'PIX2PIX.ckpt')
   run['model_checkpoints'].upload('PIX2PIX.ckpt')

   if (i+1)%5 == 0 :
     print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f},D(real): {:.2f}, D(fake):{:.2f},g_loss_gan:{:.4f},g_loss_L1:{:.4f}'
           .format(ep, epochs, i+1, len(train_loader), D_loss.item(), G_loss.item(),real_patch.mean(), fake_patch.mean(),fake_gan_loss.item(),L1_loss.item()))
     G_losses.append(G_loss.item())
     D_losses.append(D_loss.item())
     G_GAN_losses.append(fake_gan_loss.item())
     G_L1_losses.append(L1_loss.item())

     with torch.no_grad():
       G.eval()
       fake = G(fixed_x).detach().cpu()
       G.train()
     figs=plt.figure(figsize=(10,10))
     plt.subplot(1,3,1)
     plt.axis("off")
     plt.title("conditional image (x)")
     plt.imshow(np.transpose(vutils.make_grid(fixed_x, nrow=1,padding=5, normalize=True).cpu(),(1,2,0)))

     plt.subplot(1,3,2)
     plt.axis("off")
     plt.title("fake image")
     plt.imshow(np.transpose(vutils.make_grid(fake, nrow=1,padding=5, normalize=True).cpu(),(1,2,0)))

     plt.subplot(1,3,3)
     plt.axis("off")
     plt.title("ground truth (y)")
     plt.imshow(np.transpose(vutils.make_grid(fixed_y, nrow=1,padding=5, normalize=True).cpu(),(1,2,0)))

     plt.savefig(os.path.join('./','pix2pix'+"-"+str(ep) +".png"))

     run['Results'].log(File(f'pix2pix-{str(ep)}.png'))
     plt.close()
     img_list.append(figs)
```

### 训练模型不是最后一步。您需要监控和跟踪培训，以分析绩效并在必要时实施更改。考虑到监控有太多损失、图和指标要处理的 GAN 的性能是多么费力，我们将在这一步使用 Neptune。

Neptune 允许用户:

1 监控模特的现场表演

## 2 监控硬件的性能

*   3 存储和比较不同运行的不同元数据(如指标、参数、性能、数据等。)
*   与他人分享工作
*   要开始，只需遵循以下步骤:
*   1.在本地系统上使用`pip install neptune-client`或`conda install -c conda-forge neptune-client`安装 neptune-client。

2.创建账号，登录 [Neptune.ai](https://web.archive.org/web/20221117203630/https://docs.neptune.ai/getting-started/installation) 。

3.登录后，[创建一个**新项目**](https://web.archive.org/web/20221117203630/https://docs.neptune.ai/administration/projects#create-project) 。

```py
!pip install neptune-client
```

4.现在，您可以将不同的元数据记录到 Neptune。点击了解更多信息[。](https://web.archive.org/web/20221117203630/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

对于这个项目，我们将把我们的参数记录到 Neptune 仪表板中。要将参数或任何信息记录到仪表板中，请创建字典。

一旦创建了字典，我们将使用以下命令记录它们:

请记住，损耗、生成的图像和模型的权重都是使用“run”命令记录到 Neptune 仪表盘中的。

```py
PARAMS = {'Epoch': epochs,
         'Batch Size': batch_size,
         'Input Channels': c_dim,

         'Workers': workers,
         'Optimizer': 'Adam',
         'Learning Rate': 2e-4,
         'Metrics': ['Binary Cross Entropy', 'L1 Loss'],
         'Activation': ['Leaky Relu', 'Tanh', 'Sigmoid' ],
         'Device': device}

```

例如，在上面的培训中，您会发现以下命令:

```py
run['parameters'] = PARAMS
```

这些基本上是用来记录数据到海王星仪表板。

培训初始化后，所有记录的信息将自动记录到仪表板中。Neptune 从训练中获取实时信息，允许[实时监控整个过程](https://web.archive.org/web/20221117203630/https://docs.neptune.ai/how-to-guides/model-monitoring)。

```py
   run["Gen Loss"].log(G_loss.item())
   run["Dis Loss"].log(D_loss.item())
   run['L1 Loss'].log(L1_loss.item())
   run['Gen GAN Loss'].log(fake_gan_loss.item())
```

以下是监控过程截图。

您还可以访问所有元数据并查看生成的样本。

最后，您可以比较不同运行的元数据。这是很有用的，例如，当您想要查看在调整一些参数后，您的模型是否比前一个模型执行得更好。

[![Monitoring the performance of the model](img/16ab1b5ccb0bc657094019ebb5f7cb9d.png)](https://web.archive.org/web/20221117203630/https://neptune.ai/pix2pix-key-model-architecture-decisions_22)

*Monitoring the performance of the model | *[Source](https://web.archive.org/web/20221117203630/https://app.neptune.ai/nielspace/Pix2Pix/e/PIX-13/charts)**

[![Monitoring the performance of the hardware](img/d7f18002266b56538ae654123e1415bb.png)](https://web.archive.org/web/20221117203630/https://neptune.ai/pix2pix-key-model-architecture-decisions_12)

*Monitoring the performance of the hardware | *[Source](https://web.archive.org/web/20221117203630/https://app.neptune.ai/nielspace/Pix2Pix/e/PIX-13/monitoring)**

关键要点

[![Access to all metadata](img/fbd321745f226642c132f2edf91061d9.png)](https://web.archive.org/web/20221117203630/https://neptune.ai/pix2pix-key-model-architecture-decisions_7)

*Access to all metadata | *[Source](https://web.archive.org/web/20221117203630/https://app.neptune.ai/nielspace/Pix2Pix/e/PIX-13/all)**

[![Access to the generated samples](img/3a697f762aca3e4d5b62387f167fd41d.png)](https://web.archive.org/web/20221117203630/https://neptune.ai/pix2pix-key-model-architecture-decisions_19)

*Access to the generated samples | *[Source](https://web.archive.org/web/20221117203630/https://app.neptune.ai/nielspace/Pix2Pix/e/PIX-13/all?path=&attribute=Results)**

Pix2Pix 是一个有条件的 GAN，它使用图像和标签来生成图像。

[![Pix2Pix compare runs](img/e6f6486ec431d5f38efa0586f5e48a9e.png)](https://web.archive.org/web/20221117203630/https://neptune.ai/blog/pix2pix-key-model-architecture-decisions/attachment/pix2pix-compare-runs)

*Comparing metadata from different runs | [Source](https://web.archive.org/web/20221117203630/https://app.neptune.ai/nielspace/Pix2Pix/experiments?compare=IwJgNMQ&split=cmp&dash=charts&viewId=standard-view)*

## 它使用两种架构:

*   发电机的 u 形网
*   鉴别器的 PatchGAN
    *   PatchGAN 在生成的图像中使用 NxN 大小的较小补丁来区分真假，而不是一次性区分整个图像。
    *   Pix2Pix 有一个专门针对生成器的额外损耗，以便它可以生成更接近地面真实情况的图像。
*   Pix2Pix 是一种成对图像翻译算法。
*   您可以探索的其他 gan 包括:
*   CycleGAN:它类似于 Pix2Pix，因为除了数据部分，大部分方法都是相同的。它不是成对图像翻译，而是不成对图像翻译。学习和探索 CycleGAN 会容易得多，因为它是由相同的作者开发的。

#### 如果您对文本到图像的翻译感兴趣，那么您应该探索:

1.  您可能想尝试的其他有趣的 GANs 项目:
2.  StyleGAN
3.  阿尼梅根
    *   比根
    *   年龄-cGAN
    *   starman 参考资料
    *   [使用条件对抗网络的图像到图像翻译](https://web.archive.org/web/20221117203630/https://arxiv.org/abs/1611.07004)
    *   [深度学习书籍:Ian Goodfellow](https://web.archive.org/web/20221117203630/https://www.deeplearningbook.org/)

### [生成性对抗网络:古德菲勒等人](https://web.archive.org/web/20221117203630/https://arxiv.org/abs/1406.2661)

1.  [生成性对抗网络和一些 GAN 应用——你需要知道的一切](/web/20221117203630/https://neptune.ai/blog/generative-adversarial-networks-gan-applications)
2.  [生成性对抗性网络损失函数的温和介绍](https://web.archive.org/web/20221117203630/https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)。
3.  [Generative Adversarial Networks: Goodfellow et al.](https://web.archive.org/web/20221117203630/https://arxiv.org/abs/1406.2661)
4.  [Generative Adversarial Networks and Some of GAN Applications – Everything You Need to Know](/web/20221117203630/https://neptune.ai/blog/generative-adversarial-networks-gan-applications)
5.  [A Gentle Introduction to Generative Adversarial Network Loss Functions](https://web.archive.org/web/20221117203630/https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/).