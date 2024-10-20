# 如何使用 Google Colab 进行深度学习-完整教程

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-use-google-colab-for-deep-learning-complete-tutorial>

如果你是一名程序员，你想探索深度学习，需要一个平台来帮助你做到这一点——这篇教程正是为你准备的。

Google Colab 是深度学习爱好者的一个很好的平台，它也可以用来测试基本的机器学习模型，获得经验，并开发一种关于深度学习方面的直觉，如超参数调整，预处理数据，模型复杂性，过拟合等等。

让我们一起探索吧！

## 介绍

Google 的 Colaboratory(简称 Google Colab)是一个基于 Jupyter 笔记本的运行时环境，它允许你完全在云上运行代码。

这是必要的，因为这意味着你可以训练大规模的 ML 和 DL 模型，即使你没有强大的机器或高速互联网接入。

Google Colab 支持 GPU 和 TPU 实例，这使得它成为深度学习和数据分析爱好者的完美工具，因为本地机器上的计算限制。

因为 Colab 笔记本可以通过浏览器从任何机器远程访问，所以它也非常适合商业用途。

在本教程中，您将学习:

*   在 Google Colab 中四处逛逛
*   在 Colab 中安装 python 库
*   在 Colab 中下载大型数据集
*   在 Colab 中训练深度学习模型
*   在 Colab 中使用 TensorBoard

## 创造你的第一个。colab 中的 ipynb 笔记本

打开你选择的浏览器，进入[colab.research.google.com](https://web.archive.org/web/20230308085238/http://colab.research.google.com/)，使用你的谷歌账户登录。单击一个新的笔记本来创建一个新的运行时实例。

在左上角，你可以通过点击将笔记本的名称从“Untitled.ipynb”更改为你选择的名称。

单元执行块是您键入代码的地方。若要执行单元格，请按 shift + enter。

在一个单元格中声明的变量可以作为全局变量在其他单元格中使用。如果明确声明，环境会自动在代码块的最后一行打印变量值。

## 训练样本张量流模型

在 Colab 中训练一个机器学习模型是非常容易的。它最大的好处是不必建立一个定制的运行时环境，这一切都是为您处理的。

例如，让我们看看训练一个基本的深度学习模型来识别在 MNIST 数据集上训练的手写数字。

数据从标准的 Keras 数据集档案中加载。该模型非常基本，它将图像分类为数字并识别它们。

### **设置:**

```py
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

```

该代码片段的输出如下所示:

```py
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step

```

**接下来，我们使用 Python 定义 Google Colab 模型:**

```py
model = tf.keras.models.Sequential([
                               tf.keras.layers.Flatten(input_shape=(28,28)),
                                   tf.keras.layers.Dense(128,activation='relu'),
                                   tf.keras.layers.Dropout(0.2),
                                   tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
             loss=loss_fn,
             metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

```

执行上述代码片段的预期输出是:

```py
Epoch 1/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3006 - accuracy: 0.9125
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1461 - accuracy: 0.9570
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1098 - accuracy: 0.9673
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0887 - accuracy: 0.9729
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0763 - accuracy: 0.9754
<tensorflow.python.keras.callbacks.History at 0x7f2abd968fd0>

```

```py
model.evaluate(x_test,y_test,verbose=2)

```

预期产出:

```py
313/313 - 0s - loss: 0.0786 - accuracy: 0.9761
[0.07860152423381805, 0.9761000275611877]

```

```py
probability_model = tf.keras.Sequential([
                                        model,
                                        tf.keras.layers.Softmax()])

```

## 在 Google Colab 中安装软件包

您不仅可以使用 Colab 中的 code 单元运行 Python 代码，还可以运行 shell 命令。随便加个**！**前一个命令。感叹号告诉笔记本单元将下面的命令作为 shell 命令运行。

深度学习所需的大多数通用包都是预装的。在某些情况下，您可能需要不太流行的库，或者您可能需要在不同版本的库上运行代码。为此，您需要手动安装软件包。

用于安装包的包管理器是 pip。

![import tensorflow](img/dfbe364c31f002c4a0e0a594e3b1ee75.png)

要安装 TensorFlow 的特定版本，请使用以下命令:

```py
!pip3 install tensorflow==1.5.0

```

运行以上命令后，预期会有以下输出:

![tensorflow install output](img/0dd0feaca91cfdbf0992d860a4910c1c.png)

点击**重启运行时间**，使用新安装的版本。

![tf version](img/0c525694b21f4da5ab2c8e4db17221a4.png)

正如你在上面看到的，我们将 Tensorflow 版本从“2.3.0”更改为“1.5.0”。

对于我们上面训练的模型，测试精度大约为 97%。一点也不差，但这是一个简单的问题。

训练模型通常不是那么容易，我们经常不得不从像 Kaggle 这样的第三方来源下载数据集。

因此，让我们看看当我们没有直接链接时，如何下载数据集。

下载数据集

## 当您在本地机器上训练机器学习模型时，您可能会遇到下载和存储训练模型所需的数据集所带来的存储和带宽成本问题。

深度学习数据集的规模可能非常大，从 20 到 50 Gb 不等。如果你生活在发展中国家，下载它们是最具挑战性的，因为那里不可能有高速互联网。

使用数据集最有效的方式是使用云接口下载它们，而不是从本地机器手动上传数据集。

令人欣慰的是，Colab 为我们提供了多种从通用数据托管平台下载数据集的方法。

**从 Kaggle 下载数据集**

### 要从 Kaggle 下载现有数据集，我们可以遵循以下步骤:

进入你的 Kaggle 账户，点击“创建新的 API 令牌”。这将下载一个 kaggle.json 文件到你的机器上。

1.  转到您的 Google Colab 项目文件，并运行以下命令:

```py
! pip install -q kaggle
from google.colab import files

files.upload()
! mkdir ~/.kaggle

cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle competitions download -c 'name-of-competition'

```

**从任何通用网站下载数据集**

### 根据您使用的浏览器，有一些扩展可以将数据集下载链接转换为“curl”或“wget”格式。您可以使用它来有效地下载数据集。

对于 Firefox，有一个 cliget 浏览器扩展:[CLI Get–为 Firefox (en-US)获取这个扩展](https://web.archive.org/web/20230308085238/https://addons.mozilla.org/en-US/firefox/addon/cliget/)

1.  对于 Chrome，有一个 curlget 扩展: [Ad 增加了 curlget 52](https://web.archive.org/web/20230308085238/https://chrome.google.com/webstore/detail/curlwget/dgcfkhmmpcmkikfmonjcalnjcmjcjjdn?hl=en)
2.  只要你点击浏览器中的任何下载按钮，这些扩展就会生成一个 curl/wget 命令。

然后，您可以复制该命令，并在您的 Colab 笔记本中执行它，以下载数据集。

***注*** *:默认情况下，Colab 笔记本使用 Python shell。要在 Colab 中运行终端命令，您必须使用"* ***！*** *”命令的开始。*

例如，要从 some.url 下载文件并将其保存为 some.file，可以在 Colab 中使用以下命令:

***注****:curl 命令会在 Colab 工作区下载数据集，每次运行时断开连接都会丢失。因此，一个安全的做法是，一旦数据集下载完成，就将数据集移动到您的云驱动器中。*

```py
!curl http://some.url --output some.file

```

**从 GCP 或 Google Drive 下载数据集**

### 谷歌云平台是一个云计算和存储平台。您可以使用它来存储大型数据集，并且可以将该数据集直接从云中导入到 Colab 中。

要在 GCP 上传和下载文件，首先你需要验证你的谷歌账户。

它会要求你使用你的谷歌账户访问一个链接，并给你一个认证密钥。将密钥粘贴到提供的空白处，以验证您的帐户。

```py
from google.colab import auth
auth.authenticate_user()

```

之后安装 gsutil 上传下载文件，然后初始化 gcloud。

![google colab auth](img/b64f11a8e2c6745614c2e17d27083ce0.png)

这样做将要求您从基本设置的某些选项中进行选择:

```py
!curl https://sdk.cloud.google.com | bash
!gcloud init

```

一旦您配置了这些选项，您就可以使用以下命令从 Google 云存储中下载/上传文件。

![gcloud init](img/3262ebf8e17435023d06656d03d69ead.png)

要将文件从云存储下载到 Google Colab，请使用:

要将文件从 Google Colab 上传到云，请使用:

```py
!gsutil cp gs://maskaravivek-data/data_file.csv

```

在启用 GPU/TPU 的情况下启动运行时

```py
gsutil cp test.csv gs://maskaravivek-data/

```

## 深度学习是一个计算量很大的过程，需要同时执行大量计算来训练一个模型。为了缓解这个问题，Google Colab 不仅为我们提供了经典的 CPU 运行时，还提供了 GPU 和 TPU 运行时的选项。

CPU 运行时最适合训练大型模型，因为它提供了高内存。

GPU 运行时对于不规则计算，如小批量和非大型计算，表现出更好的灵活性和可编程性。

TPU 运行时针对大批量和 CNN 进行了高度优化，具有最高的训练吞吐量。

如果你有一个较小的模型要训练，我建议在 GPU/TPU 运行时上训练模型，以充分发挥 Colab 的潜力。

要创建支持 GPU/TPU 的运行时，您可以在文件名下方的工具栏菜单中单击运行时。从那里，点击“**更改运行时类型**”，然后在硬件加速器下拉菜单下选择 GPU 或 TPU。

请注意，Google Colab 的免费版本并不保证 GPU/TPU 支持的运行时的持续可用性。如果使用时间过长，您的会话可能会被终止！

你可以购买 Colab Pro(如果你在美国或加拿大，目前只能在这些国家购买)。每月 10 美元，不仅提供更快的 GPU，还提供更长的会话。前往[此链接](https://web.archive.org/web/20230308085238/https://colab.research.google.com/signup) [。](https://web.archive.org/web/20230308085238/https://colab.research.google.com/signup)

训练更复杂和更大的模型

## 为了训练复杂的模型，通常需要加载大型数据集。建议使用 mount drive 方法直接从 Google Drive 加载数据。

这将把所有数据从您的驱动器导入运行时实例。首先，您需要安装存储数据集的 Google Drive。

您还可以使用 Colab 中的默认存储，并将数据集从 GCS 或 Kaggle 直接下载到 Colab。

**安装驱动器**

### Google Colab 允许您从 Google Drive 帐户导入数据，以便您可以从 Google Drive 访问训练数据，并使用大型数据集进行训练。

有两种方法可以在 Colab 中挂载驱动器:

使用 GUI

*   使用代码片段
*   **1。使用 GUI**

点击屏幕左侧的文件图标，然后点击“安装驱动器”图标来安装您的谷歌驱动器。

**2。使用代码片段**

执行以下代码块在 Colab 上安装您的 Google Drive:

单击链接，复制代码，并将其粘贴到提供的框中。按 enter 键安装驱动器。

```py
from google.colab import drive
drive.mount('/content/drive')

```

接下来，我们将训练一个卷积神经网络(CNN)来识别手写数字。这是在一个基本数据集和一个原始模型上训练的，但现在我们将使用一个更复杂的模型。

![Google colab code](img/54d0dfce834d6362fa2679edc1a318fa.png)

**用 Keras 训练模型**

### Keras 是用 Python 写的 API，它运行在 Tensorflow 之上。它用于快速原型实验模型和评估性能。

与 Tensorflow 相比，在 Keras 中部署模型非常容易。这里有一个例子:

一旦这个单元被执行，您将会看到类似如下的输出:

```py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

cd drive/My Drive/

data = pd.read_csv('train.csv')

train = data.iloc[0:40000,:] 
train_X = train.drop('label',axis=1)
train_Y = train.iloc[:,0]

val = data.iloc[40000:,:]    
val_X = val.drop('label',axis=1)
val_Y = val.iloc[:,0]

train_X = train_X.to_numpy() 
train_Y = train_Y.to_numpy()
val_X = val_X.to_numpy()

val_Y = val_Y.to_numpy()
train_X = train_X/255.    
val_X = val_X/255.

train_X = np.reshape(train_X,(40000,28,28,1)) 
val_X = np.reshape(val_X,(2000,28,28,1))

model = keras.Sequential([
   keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
   keras.layers.MaxPooling2D((2,2)),
   keras.layers.Conv2D(64,(3,3),activation='relu'),
   keras.layers.MaxPooling2D((2,2)),
   keras.layers.Conv2D(64,(3,3),activation='relu'),
   keras.layers.Flatten(),

   keras.layers.Dense(64,activation='relu'),

   keras.layers.Dense(10)
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(train_X,train_Y,epochs=10,validation_data=(val_X, val_Y))

```

预期产出:

```py
Epoch 1/10
1250/1250 [==============================] - 37s 30ms/step - loss: 0.1817 - accuracy: 0.9433 - val_loss: 0.0627 - val_accuracy: 0.9770
Epoch 2/10
1250/1250 [==============================] - 36s 29ms/step - loss: 0.0537 - accuracy: 0.9838 - val_loss: 0.0471 - val_accuracy: 0.9850
Epoch 3/10
1250/1250 [==============================] - 36s 29ms/step - loss: 0.0384 - accuracy: 0.9883 - val_loss: 0.0390 - val_accuracy: 0.9875
...
1250/1250 [==============================] - 36s 29ms/step - loss: 0.0114 - accuracy: 0.9963 - val_loss: 0.0475 - val_accuracy: 0.9880
Epoch 10/10
1250/1250 [==============================] - 36s 29ms/step - loss: 0.0101 - accuracy: 0.9967 - val_loss: 0.0982 - val_accuracy: 0.9735

```

```py
test_loss, test_acc = model.evaluate(val_X,val_Y,verbose=2)
Expected output:
63/63 - 1s - loss: 0.0982 - accuracy: 0.9735

predict_model = tf.keras.Sequential([
   model,tf.keras.layers.Softmax()
])

test_image = val_X[140]
test_image = np.reshape(test_image,(1,28,28,1))
result = predict_model.predict(test_image)
print(np.argmax(result))
plt.imshow(val_X[140].reshape(28,28))
plt.show()

```

**使用 FastAI**

### FastAI 是一个工作在 PyTorch 之上的高级库。它让你用很少几行代码来定义。它用于通过遵循自上而下的方法来学习深度学习的基础知识，即首先我们编码并查看结果，然后研究其背后的理论。

这里可以看到 FastAI 在 Colab 上的一个示例实现[。](https://web.archive.org/web/20230308085238/https://course.fast.ai/start_colab)

如何使用 [Neptune-Keras integration](https://web.archive.org/web/20230308085238/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras) 或[Neptune-fastai integration](https://web.archive.org/web/20230308085238/https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai)跟踪模型训练元数据。

Google Colab 中的 TensorBoard

## TensorBoard 是 Tensorflow 提供的用于可视化机器学习相关数据的工具包。

它通常用于绘制度量标准，例如迭代次数的损失和准确性。它还可以用于可视化和总结模型，并显示图像、文本和音频数据。

**使用张量板的监测数据**

### 要使用 TensorBoard，需要导入一些必要的库。运行以下代码片段来导入这些库:

在我们开始可视化数据之前，我们需要在 model.fit()中做一些更改:

```py
%load_ext tensorboard
import datetime, os

```

训练结束后，您可以启动 TensorBoard 工具包来查看模型的表现:

```py
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.fit(x=x_train,y=y_train,epochs=5,validation_data=(x_test, y_test),callbacks=[tensorboard_callback])

```

它给出了精度和损失如何随运行的时期数而变化的信息。

```py
%tensorboard --logdir logs
```

保存和加载模型

## 训练模型需要花费大量时间，因此能够保存训练好的模型以便反复使用将是明智的。每次都训练它会非常令人沮丧和耗时。Google Colab 允许您保存模型并加载它们。

**保存和加载模型重量**

### 训练 DL 模型的基本目的是以正确预测输出的方式调整权重。仅保存模型的权重，并在需要时加载它们是有意义的。

要手动保存重量，请使用:

要在模型中加载权重，请使用:

```py
 model.save_weights('./checkpoints/my_checkpoint')

```

**保存并加载整个模型**

```py
model.load_weights('./checkpoints/my_checkpoint'
```

### 有时候，最好保存整个模型，这样可以省去定义模型、处理输入维度和其他复杂性的麻烦。您可以保存整个模型并将其导出到其他机器。

要保存整个模型，请使用:

要加载已保存的模型，请使用:

```py
model.save('saved_model/my_model'
```

结论

```py
new_model = tf.keras.models.load_model('saved_model/my_model')

```

## 现在你看到 Google Colab 是一个很好的工具，可以原型化和测试深度学习模型。

凭借其免费的 GPU 和从 Google Drive 导入数据的能力，Colab 脱颖而出，成为在计算和存储有限的低端机器上训练模型的非常有效的平台。

它还管理 Google Drive 中的笔记本，为希望一起从事同一项目的程序员提供一个稳定而有组织的数据管理系统。

感谢阅读！

Thanks for reading!