# 通过迁移学习、数据扩充、LR Finder 等提高模型准确性

> 原文：<https://blog.paperspace.com/improving-model-accuracy/>

在处理由专家预先清理、测试、分割和处理的流行数据集时，很容易获得 90%以上的准确率。您只需要将数据集导入并提供给互联网上最流行的模型架构。

在影像分类中，当一个新数据集的某个类别中只有很少的影像，或者影像与您将在生产中处理的影像不相似时，事情会变得有点困难。流行的模型架构似乎没有帮助，迫使你陷入一个只有 50%准确率的角落，然后变成一个概率游戏，而不是机器学习本身。

本文重点探讨所有这些方法、工具以及更多内容，以帮助您构建健壮的模型，这些模型可以在生产中轻松部署。尽管其中一些方法也适用于其他目标，但我们将重点放在图像分类上来探讨这个主题。

### 为什么自定义数据集达不到高精度？

重要的是要说明为什么定制数据集在实现良好的性能指标方面最失败。当您尝试使用自己创建的数据集或从团队获得的数据集来创建模型时，可能会遇到这种情况。缺乏多样性可能是基于它的模型性能不佳的主要原因之一。图像的光照、颜色、形状等参数可能会有显著变化，在构建数据集时可能不会考虑这一点。数据增强可能会帮助您解决这个问题，我们将进一步讨论这个问题。

另一个原因可能是缺乏对每个类别的关注:一个数据集有一种咖啡的 1000 多张图像，而另一种只有 100 多张图像，这在可以学习的特征方面产生了很大的不平衡。另一个失败可能是数据收集的来源与生产中收集数据的来源不匹配。这种情况的一个很好的例子可以是从具有差的视频质量的安全摄像机中检测到鸟，该差的视频质量作为在高清晰度图像上训练的模型的输入。有各种方法可以处理这种情况。

### 为什么生产水平的准确性很重要？

既然我们已经讨论了为什么定制数据集在第一次运行时无法达到“生产级别的准确性”，那么理解为什么生产级别的准确性很重要。简而言之，我们的模型应该能够给出在现实场景中可以接受的结果，但不一定要达到 100%的准确性。使用测试数据集图像或文本很容易看到正确的预测，这些图像或文本用于将模型超调至最佳状态。尽管我们不能确定一个阈值精度，超过这个精度，我们的模型就有资格进行部署，但根据经验，如果训练和验证数据是随机分割的，那么至少有 85-90%的验证精度是很好的。始终确保验证数据是多样化的，并且其大部分数据与模型在生产中使用的数据相似。数据预处理可以通过在输入前调整大小或过滤文本来确保图像大小，从而在一定程度上帮助您实现这一点。在开发过程中处理此类错误有助于改进您的生产模式并获得更好的结果。

## 数据扩充:改善数据集的完美方式

只要你能通过**数据扩充**等方法充分利用数据集，拥有一个小数据集是没问题的。这个概念侧重于预处理现有数据，以便在我们没有足够数据的时候生成更多样化的数据用于训练。让我们用一个小例子来讨论一下图像数据增强。这里我们有一个来自 TensorFlow 的[石头剪子布数据集，我们希望不重复生成更多。](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) [Tensorflow 数据集对象](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)提供了许多有助于数据扩充的操作，等等。这里我们首先缓存数据集，这有助于我们进行内存管理，因为数据集第一次迭代时，它的元素将被缓存在指定的文件或内存中。然后缓存的数据可以在以后使用。

之后我们重复数据集两次，这增加了它的基数。仅仅重复的数据对我们没有帮助，但是我们在加倍的数据集上添加了一个映射层，这在某种程度上帮助我们随着基数的增加生成新的数据。在这个例子中，我们将随机图像左右翻转，这避免了重复并确保了多样性。

```py
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = 'rock_paper_scissors'

(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)

def preprocess_img(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Make sure that image has a right size
    image = tf.image.resize(image, [256,256])
    return image, label

dataset_train = dataset_train_raw.map(preprocess_img)
dataset_test = dataset_test_raw.map(preprocess_img)

print("Dataset Cardinality Before Augmentation: ",dataset_train.cardinality().numpy())

dataset_train = dataset_train.cache().repeat(2).map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
)

print("Dataset Cardinality After Augmentation: ",dataset_train.cardinality().numpy())
```

输出

```py
Dataset Cardinality Before Augmentation:  2520
Dataset Cardinality After Augmentation:  5040
```

图像上有更多的映射可以探索，可以在对比度、旋转等方面进一步创造更多的变化。阅读这篇文章了解更多细节。您可以对图像执行更多操作，如旋转、剪切、改变对比度等等。在图像数据在照明、背景和其他方面不代表真实世界输入的情况下，数据扩充至关重要。在这里，我们讨论了通过 Tensorflow 等框架进行数据扩充，但是除了旋转和剪切之外，您还可以进行手动数据扩充。

[Mapping](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) 是一个强大的工具，因为您可以对单个数据执行任何操作，而无需经历迭代。调整图像大小、格式化文本等等都可以用它来灵活处理。

## 迁移学习:使用小数据集

有些情况下，您只有少量图像，并且希望构建一个图像分类模型。如果图像很少，模型可能无法学习模式和更多内容，这将导致模型过拟合或欠拟合，在现实世界输入的生产中表现不佳。在这种情况下，建立一个好模型最简单的方法就是通过迁移学习。

有像 [VGG16](https://keras.io/api/applications/vgg/) 这样著名的预训练模型，确实很擅长图像分类。由于它在构建时所接触到的各种各样的数据，以及其体系结构的复杂性质(包括许多卷积神经网络)，它在图像分类目标方面比我们可以用小数据集构建的小模型更有深度。我们可以使用这样的预训练模型，通过替换最后几层(在大多数情况下)来处理我们问题的相同目标。我们之所以替换最后一层，是为了重构适合我们用例的模型输出，在图像分类的情况下，选择合适的类别数进行分类。如果我们遵循相应的预训练模型架构的文档和围绕它的框架文档，我们不仅可以替换最后一层，还可以替换任意多的层。让我们建立一个样本迁移学习机器学习模型。

首先，我们正在加载和预处理我们之前使用的相同的石头剪刀布数据集。

```py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import BatchNormalization, Dropout
from keras.models import Model

DATASET_NAME = 'rock_paper_scissors'

(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)

def preprocess_img(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Resize images to ensure same input size
    image = tf.image.resize(image, [256,256])
    return image, label

dataset_train = dataset_train_raw.map(preprocess_img)
dataset_test = dataset_test_raw.map(preprocess_img)

dataset_train = dataset_train.batch(64)
dataset_test = dataset_test.batch(32)
```

现在我们将使用 [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) 作为迁移学习模型。我们将设置**training able = false**来冻结 ResNet50 架构，不将其暴露于训练。这将为我们节省大量时间，因为模型将只训练最后几层。当我们按小时付费进行培训时，这是有益的。

```py
# ResNet50 with Input shape of our Images
# Include Top is set to false to allow us to add more layers

res = ResNet50(weights ='imagenet', include_top = False, 
               input_shape = (256, 256, 3)) 

# Setting the trainable to false
res.trainable = False

x= res.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(3, activation ='softmax')(x)
model = Model(res.input, x)

model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"])

model.summary()
```

### 简而言之

```py
( Only the Bottom part of Model Summary Included here as the ResNet Summary is long)
_____________________________________________________________________________
conv5_block3_out (Activation)   (None, 8, 8, 2048)   0           conv5_block3_add[0][0]           
_____________________________________________________________________________
global_average_pooling2d_5 (Glo (None, 2048)         0           conv5_block3_out[0][0]           
_____________________________________________________________________________
batch_normalization_11 (BatchNo (None, 2048)         8192        global_average_pooling2d_5[0][0] 
_____________________________________________________________________________
dropout_11 (Dropout)            (None, 2048)         0           batch_normalization_11[0][0]     
_____________________________________________________________________________
dense_11 (Dense)                (None, 512)          1049088     dropout_11[0][0]                 
_____________________________________________________________________________
batch_normalization_12 (BatchNo (None, 512)          2048        dense_11[0][0]                   
_____________________________________________________________________________
dropout_12 (Dropout)            (None, 512)          0           batch_normalization_12[0][0]     
_____________________________________________________________________________
dense_12 (Dense)                (None, 3)            1539        dropout_12[0][0]                 
=============================================================================
Total params: 24,648,579
Trainable params: 1,055,747
Non-trainable params: 23,592,832
```

模特培训

```py
model.fit(dataset_train, epochs=6, validation_data=dataset_test)
```

```py
Epoch 1/10
40/40 [==============================] - 577s 14s/step - loss: 0.2584 - sparse_categorical_accuracy: 0.9147 - val_loss: 1.1330 - val_sparse_categorical_accuracy: 0.4220

Epoch 2/10
40/40 [==============================] - 571s 14s/step - loss: 0.0646 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.8574 - val_sparse_categorical_accuracy: 0.4247

Epoch 3/10
40/40 [==============================] - 571s 14s/step - loss: 0.0524 - sparse_categorical_accuracy: 0.9813 - val_loss: 0.7408 - val_sparse_categorical_accuracy: 0.6425

Epoch 4/10
40/40 [==============================] - 570s 14s/step - loss: 0.0376 - sparse_categorical_accuracy: 0.9881 - val_loss: 0.6260 - val_sparse_categorical_accuracy: 0.7016

Epoch 5/10
40/40 [==============================] - 570s 14s/step - loss: 0.0358 - sparse_categorical_accuracy: 0.9881 - val_loss: 0.5864 - val_sparse_categorical_accuracy: 0.6532

Epoch 6/10
40/40 [==============================] - 570s 14s/step - loss: 0.0366 - sparse_categorical_accuracy: 0.9873 - val_loss: 0.4445 - val_sparse_categorical_accuracy: 0.8602
```

我们可以看到，在相对较小的数据集上训练的模型表现非常好，验证准确率为 86%。如果你关注每个时期所花费的时间，它不到 10 分钟，因为我们保持 ResNet 层不可训练。ResNet50 帮助我们把它的学习转移到我们的问题上。您可以尝试各种预先训练好的模型，看看它们如何适合您的问题，以及哪种模型性能最好。

## LR Finder:寻找完美的学习速度

[Learning Rate Finder](https://github.com/surmenok/keras_lr_finder) 是一款功能强大的工具，顾名思义，可以帮助你轻松找到 LR。尝试所有的学习率来找到完美的学习率是一种低效且耗时的方法。LR Finder 是实现这一点的最高效、最省时的方法。我们来看看如何实现。我们继续使用相同的数据集、预处理和模型架构，因此从这里开始不再重复。

```py
!pip install tensorflow-hub
!git clone https://github.com/beringresearch/lrfinder/
!cd lrfinder && python3 -m pip install .

import numpy as np
from lrfinder import LRFinder
K = tf.keras.backend

BATCH = 64

# STEPS_PER_EPOCH = np.ceil(len(train_data) / BATCH)
# here Cardinality or Length of Train dataset is 2520

STEPS_PER_EPOCH = np.ceil(2520 / BATCH)
lr_finder = LRFinder(model)
lr_finder.find(dataset_train, start_lr=1e-6, end_lr=1, epochs=10,
               steps_per_epoch=STEPS_PER_EPOCH)

learning_rates = lr_finder.get_learning_rates()
losses = lr_finder.get_losses()

best_lr = lr_finder.get_best_lr(sma=20)

# Setting it as our model's LR through Keras Backend
K.set_value(model.optimizer.lr, best_lr)
print(best_lr)
```

```py
Epoch 1/10
40/40 [==============================] - 506s 13s/step - loss: 1.7503 - sparse_categorical_accuracy: 0.3639
Epoch 2/10
40/40 [==============================] - 499s 12s/step - loss: 1.5044 - sparse_categorical_accuracy: 0.4302
Epoch 3/10
40/40 [==============================] - 498s 12s/step - loss: 0.9737 - sparse_categorical_accuracy: 0.6163
Epoch 4/10
40/40 [==============================] - 495s 12s/step - loss: 0.4744 - sparse_categorical_accuracy: 0.8218
Epoch 5/10
40/40 [==============================] - 495s 12s/step - loss: 0.1946 - sparse_categorical_accuracy: 0.9313
Epoch 6/10
40/40 [==============================] - 495s 12s/step - loss: 0.1051 - sparse_categorical_accuracy: 0.9663
Epoch 7/10
40/40 [==============================] - 89s 2s/step - loss: 0.1114 - sparse_categorical_accuracy: 0.9576
```

我们得到的最佳学习率是 **6.31 e-05，**，我们使用 Keras 后端将它设置为我们的模型 LR。从输出来看，很明显，这个过程只花了几个时期，它分析了所有可能的学习率，并找到了最好的一个。我们可以使用 Matplotlib 可视化学习率及其性能。**红线**代表最佳学习率。

```py
import matplotlib.pyplot as plt

def plot_loss(learning_rates, losses, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
    f, ax = plt.subplots()
    ax.set_ylabel("loss")
    ax.set_xlabel("learning rate (log scale)")
    ax.plot(learning_rates[:-1],
            losses[:-1])
    ax.set_xscale(x_scale)
    return(ax)

axs = plot_loss(learning_rates, losses)
axs.axvline(x=lr_finder.get_best_lr(sma=20), c='r', linestyle='-.')
```

![](img/e2b8b868d63a8027192d62ff915d0334.png)

learning rate finder graph

## 提前停止:在你的模型忘记之前拯救它

你可能记得训练一个模型超过 20 个历元，模型的损失在一个点之后开始增加。你被卡住了，你什么也做不了，因为打断会扼杀这个过程，等待会给你一个表现更差的模型。在这种情况下，当损失等参数开始增加时，你可以轻松地获得最佳模型并逃离这个过程，尽早停止正是你想要的。这也将节省您的时间，如果模型开始显示早期的正损失，该过程将通过向您提供最后的最佳损失模型而停止，并且不计算进一步的时期。您也可以根据任何可以监控的参数(如精确度)设置提前停止。提前停车的主要参数之一是**耐心**。这是您希望看到模型是否停止显示增加的损失并回到学习轨道的次数，否则它将保存增加前的最后最佳损失并停止训练。现在你可能已经有了一个小想法，让我们来看一个例子。

```py
from tensorflow.keras.callbacks import EarlyStopping

earlystop_callback = EarlyStopping(
  monitor='val_loss', min_delta=0.0001, patience=2)

model.fit(dataset_train, epochs=20, validation_data=dataset_test, callbacks=[earlystop_callback])
```

在本例中，提前停止被设置为监控**验证损失。**参数**最小 delta** ，即我们希望损失的最小差值，被设置为 0.0001，**耐心**被设置为 2。耐心为 2 意味着模型可以在验证损失增加的情况下再运行 2 个时期，但是如果它没有显示减少的损失，那么(低于从其开始增加的损失)，该过程将通过返回最后的最佳损失版本而被终止。

```py
( Only the last part of training shown )

Epoch 10/20
40/40 [==============================]  loss: 0.0881 - sparse_categorical_accuracy: 0.9710 - val_loss: 0.4059 
Epoch 11/20
40/40 [==============================]  loss: 0.0825 - sparse_categorical_accuracy: 0.9706 - val_loss: 0.4107 
Epoch 12/20
40/40 [==============================]  loss: 0.0758 - sparse_categorical_accuracy: 0.9770 - val_loss: 0.3681 
Epoch 13/20
40/40 [==============================]  loss: 0.0788 - sparse_categorical_accuracy: 0.9754 - val_loss: 0.3904 
Epoch 14/20
40/40 [==============================]  loss: 0.0726 - sparse_categorical_accuracy: 0.9770 - val_loss: 0.3169 
Epoch 15/20
40/40 [==============================]  loss: 0.0658 - sparse_categorical_accuracy: 0.9786 - val_loss: 0.3422 
Epoch 16/20
40/40 [==============================]  loss: 0.0619 - sparse_categorical_accuracy: 0.9817 - val_loss: 0.3233 
```

即使设置了 20 个时期来训练，模型也在第 16 个时期后停止训练，从而避免了模型因验证损失增加而遗忘。我们的训练结果有一些非常好的观察，可以帮助我们更深入地了解早期停止。在第 14 个纪元模型是在其最佳损失，0.3168。下一个时期显示了 0.3422 的增加的损失，即使下一个时期显示了 0.3233 的减少的损失，其小于前一个时期，其仍然大于从增加开始的点(0.3168)，因此在保存第 14 个时期的模型版本时训练停止。它等待 2 个时期，以查看训练是否会因为耐心参数被设置为 2 而自我纠正。

另一个有趣的观察结果是从第 10 个时期到第 12 个时期，尽管在第 11 个时期损失增加了(0.4107)，但是与第 10 个时期的损失(0.4059)相比，第 12 个时期的损失减少了(0.3681)。因此，随着模型回到正轨，训练仍在继续。这可以被看作是对耐心的一个很好的利用，因为让它默认会在第 11 个纪元后终止训练，而不是尝试下一个纪元。

使用早期停止的一些技巧是，如果你正在 CPU 上训练，使用小的耐心设置。如果在 GPU 上训练，使用更大的耐心值。对于甘这样的模型，不如用小耐心，省模型检查点。如果您的数据集不包含大的变化，那么使用更大的耐心。设置 min_delta 参数始终基于运行几个历元并检查验证损失，因为这将使您了解验证损失如何随着历元而变化。

## 分析您的模型架构

这是一个通用的方法，而不是一个确定的方法。在大多数情况下，例如涉及卷积神经网络的图像分类，非常重要的一点是，即使框架处理流程，您也要很好地了解您的卷积、它们的核大小、输出形状等等。像 ResNet 这样的非常深入的体系结构是为了在 256x256 大小的图像上进行训练而设计的，调整它的大小以适应数据集图像为 64x64 的情况可能会执行得很差，导致某些预训练模型的精度为 10%。这是因为预训练模型中的层数和您的图像大小。很明显，随着通道平行增加，图像张量通过卷积时在尺寸上变得更小。在 256x256 上训练的预训练模型最终将具有至少 8×8 的张量大小，而如果将它重构为 64×64，则最后几个卷积将获得 1×1 的张量，与 8×8 的输入相比，它学习得很少。这是在处理预训练模型时要小心处理的事情。

另一方面是当你建立自己的回旋。确保它有超过 3 层的深度，同时，考虑到你的图像尺寸，它也不会影响输出尺寸。分析模型摘要非常重要，因为您可以根据卷积层的输出形状等因素来决定密集层的设置。在处理[多特性和多输出模型](https://blog.paperspace.com/combining-multiple-features-outputs-keras/)时，架构非常重要。在这种情况下，模型可视化会有所帮助。

## 结论

到目前为止，我们已经讨论了一些最有影响力和最受欢迎的方法，这些方法可以提高您的模型准确性、改善您的数据集以及优化您的模型架构。还有很多其他的方法等着你去探索。除此之外，还有更多次要的方法或指南可以帮助您实现上述所有方面，例如加载数据时的混排、使用 TensorFlow dataset 对象处理您自定义创建的数据集、使用我们之前讨论的映射来处理操作。我建议你在训练时注重验证准确性，而不是训练准确性。验证数据必须得到很好的处理，它的多样性和对模型在生产中将要接触到的真实世界输入的代表性是非常重要的。

尽管我们对一个图像分类问题使用了所有的方法，其中一些像映射、学习率查找器等等。适用于涉及文本和更多的其他问题。构建低于平均精度的模型在现实生活中没有价值，因为精度很重要，在这种情况下，这些方法可以帮助我们构建一个接近完美的模型，并考虑到所有方面。超参数调优是一种流行的方法，本文没有详细讨论。简而言之，就是尝试超参数的各种值，如时期、批量大小等。超参数调整的目的是获得最佳参数，最终得到更好的模型。LR 查找器是超参数调整学习速率的有效方法。在处理 SVR 等其他机器学习算法时，超参数调整起着至关重要的作用。

我希望您已经很好地了解了使用各种想法和方法来处理您的模型以实现更好的性能是多么重要，并为您未来的机器学习之旅尽善尽美。我希望这些方法对你有用。感谢阅读！