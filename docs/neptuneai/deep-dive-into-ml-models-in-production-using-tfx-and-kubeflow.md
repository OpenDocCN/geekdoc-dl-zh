# 使用 TensorFlow Extended (TFX)和 Kubeflow 深入研究生产中的 ML 模型

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/deep-dive-into-ml-models-in-production-using-tfx-and-kubeflow>

> 如果整个行业的 Jupyter 笔记本电脑中每浪费一个机器学习模型，我就会成为百万富翁。—我

构建模型是一回事，将模型投入生产又是另一回事。机器学习最难的部分之一是有效地将模型投入生产。

这里强调的是**有效地**，因为虽然有很多方法可以将模型投入生产，但是很少有工具可以有效地部署、监控、跟踪和自动化这个过程。

在本教程中，我将向您介绍 **TensorFlow Extended，俗称 TFX。**你将使用 TFX、谷歌人工智能平台管道和 Kubeflow，将一个示例机器学习项目投入生产。

如果您是第一次学习这些工具，请不要担心，我将尽力正确地解释它们，同时也实际地实现它们。

要完成本教程或继续学习，你需要一个 [GCP 账户](https://web.archive.org/web/20221201153827/https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=emea-emea-all-en-dr-bkws-all-all-trial-b-gcp-1009139&utm_content=text-ad-none-any-DEV_c-CRE_380533847713-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+BMM_M:1_EMEA_EN_General_Cloud_gcp-KWID_43700053286073021-kwd-20903505266-userloc_1029654&utm_term=KW_%2Bgcp-NET_g-PLAC_&ds_rl=1242853&ds_rl=1245734&ds_rl=1242853&ds_rl=1245734&gclid=Cj0KCQjw7sz6BRDYARIsAPHzrNLXt8SrFgZ1ZFiZXIwbxO1p3mcq5dXgDv2YcvgFmaSNQTAY8kBF9sYaAmrqEALw_wcB)。如果你还没有，请到[这里](https://web.archive.org/web/20221201153827/https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=emea-emea-all-en-dr-bkws-all-all-trial-b-gcp-1009139&utm_content=text-ad-none-any-DEV_c-CRE_380533847713-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+BMM_M:1_EMEA_EN_General_Cloud_gcp-KWID_43700053286073021-kwd-20903505266-userloc_1029654&utm_term=KW_%2Bgcp-NET_g-PLAC_&ds_rl=1242853&ds_rl=1245734&ds_rl=1242853&ds_rl=1245734&gclid=Cj0KCQjw7sz6BRDYARIsAPHzrNLXt8SrFgZ1ZFiZXIwbxO1p3mcq5dXgDv2YcvgFmaSNQTAY8kBF9sYaAmrqEALw_wcB)注册，你会得到 300 美元的注册积分，可以用来进行实验。除此之外，您还需要:

*   对机器学习的基本理解
*   对[张量流](/web/20221201153827/https://neptune.ai/integrations/tensorflow)的基本理解
*   对云平台有一点熟悉
*   显然，你也需要 Python

## TFX 和库伯弗洛简介

TFX 是一个基于 Tensorflow 的生产规模的机器学习平台。它由谷歌所有并积极维护，在谷歌内部使用。

TFX 提供了一系列框架、库和组件，用于定义、启动和监控生产中的机器学习模型。

TFX 提供的组件让您可以从一开始就构建专为扩展而设计的高效 ML 管道。这些组件包括:

*   建模，
*   培养
*   上菜(推断)，
*   以及管理不同目标的部署，如 web、移动或物联网设备。

在下图中，您可以看到可用 TFX 库和可用管道组件之间的关系，您可以使用它们来实现这一点。

您会注意到，TFX 库和组件涵盖了一个典型的端到端机器学习管道，从数据摄取开始，到模型服务结束。

Python 中提供了用于实现上述不同任务的 TFX 库，这些库可以单独安装。但是建议只安装 TFX，它带有所有的组件。

随着本教程的进行，你将会使用不同的 TFX 组件，在向你展示如何使用它之前，我将首先解释它的功能。

### **幕后的 TFX 和指挥者**

默认情况下，TFX 会为您的 ML 管道创建一个有向无环图(DAG)。它使用 [Apache-Beam](https://web.archive.org/web/20221201153827/https://beam.apache.org/) 来管理和实现管道，这可以在分布式处理后端上轻松执行，如 [Apache Spark](https://web.archive.org/web/20221201153827/https://spark.apache.org/) 、 [Google Cloud Dataflow、](https://web.archive.org/web/20221201153827/https://cloud.google.com/dataflow) [Apache Flink](https://web.archive.org/web/20221201153827/https://flink.apache.org/) 等等。

值得一提的是，Beam 附带了一个直接通道，因此它可以用于测试或小型部署等场景。

虽然运行 Apache Beam 的 TFX 很酷，但是很难配置、监控和维护定义的管道和工作流。这就产生了我们称之为管弦乐队的工具。

像 Kubeflow 或 Apache Airflow 这样的编排器使得配置、操作、监控和维护 ML 管道变得很容易。它们大多带有你容易理解的图形用户界面。

**以下是显示已定义任务的示例气流 GUI:**

**这里有一个给 Kubeflow:**

在本教程中，您将使用 Kubeflow 作为 ML 管道的编排器，所以让我们简单地谈谈 Kubeflow。

[Kubeflow](https://web.archive.org/web/20221201153827/https://www.kubeflow.org/) 是一款开源[kubernetes](https://web.archive.org/web/20221201153827/https://github.com/kubeflow/kubeflow)-专门为开发、编排、部署和运行可扩展 ML 工作负载而设计的原生平台。

它可以帮助您轻松管理端到端的 ML 管道流程编排，在内部或云等众多环境中运行工作流，并为工作流的可视化提供支持。

现在我们已经有了基本的介绍，让我们进入本教程的实践方面。在下一部分中，您将在 GCP 上设置您的项目。

## 建立一个新的谷歌云项目

如果您已经创建了一个 Google Cloud 帐户，那么您已经拥有了免费的 GC 积分。接下来，您将为本教程创建一个新项目。

![](img/6ee4db9241456412a31ee02e477cd458.png)

如果你已经用完了你的免费积分，那么你的 GC 账户将会收费，所以记得在本教程结束时清理项目。

要设置新项目，请按照以下步骤操作:

*   为您的项目命名。我会把我的叫做 tfx-project。

*   点击**创建**新项目
*   创建完成后，再次单击项目下拉菜单，并选择您刚刚创建的新项目。

如果操作正确，您应该会看到您的项目仪表板。

## 配置 AI 平台管道并设置 Kubeflow

现在您已经创建了项目，您将设置 Kubeflow。按照以下步骤实现这一点:

*   点击汉堡菜单，显示 GCP 上可用的服务，滚动到你有 **AI 平台**的地方，选择**管道。**

*   在 pipelines 页面中，单击**新实例**按钮。这将打开 Kubeflow 管道页面。您可以在这里配置 Kubeflow 引擎。
*   您可以为您的群集选择一个区域，或者保留默认值。
*   **确保选中允许访问框**，因为这是您的集群访问其他云 API 所必需的。

*   点击**创建集群**按钮，等待几分钟直到集群创建完毕。
*   您可以选择一个名称空间，或者保留默认值。此外，如果您将为您的工件使用一个托管存储，那么您可以添加您的存储细节，或者留空。
*   最后点击**部署**，等待管道部署完成。

## 设置云人工智能平台笔记本

接下来，您将在 AI 平台中设置一个笔记本。这让我们可以在熟悉的 Jupyter 笔记本环境中进行实验。要实例化这个新笔记本，请执行以下步骤:

*   从 GC 汉堡菜单中选择 **AI 平台**选项，点击**笔记本**选项。
*   接下来，选择**启用笔记本 API** ，然后点击**新建实例**，在 AI 平台上创建一个新的笔记本实例。
*   根据您想要做的事情和您的预算，您可以选择一个自定义的启动库来添加到您的笔记本实例。对于本文，我将使用没有安装 GPU 的 Tensorflow 2.1 版本。

*   在弹出菜单中，点击底部的**自定义**。这将打开一个配置页面，如果向下滚动，您将看到一个减少计算实例的 CPU 和 RAM 数量的选项。为了降低成本，您将使用更小的产品。

如果你有足够的预算，并且需要非常快或大的东西，你可以跳过上面的步骤。

*   最后，滚动到最后，点击**创建**创建笔记本。

## 在云端体验笔记本电脑

现在您已经设置了笔记本，您将打开它。按照以下步骤实现这一点:

*   在 AI 平台仪表板中，再次单击管道。这一次您将看到刚刚创建的 Kubeflow 管道，在它旁边有一个 **OPEN PIPELINES DASHBOARD** 命令。点击它。

这将在 Kubeflow 中打开您的管道，从这里您可以执行许多特定于 Kubeflow 的功能。

*   接下来点击**打开 TF 2.1 笔记本**。这将打开**笔记本**页面，您可以在其中选择一个笔记本实例。单击您之前创建的笔记本实例，如下所示。

*   最后，点击 **Create** 来启动笔记本实例。这将打开一个熟悉的 Jupyter 笔记本，您将在其中进行实验。

在打开的 Jupyter 笔记本中，**导入的**文件夹中提供了一个模板文件，帮助您执行一些重要的配置和设置，以及一个用于处理 TFX 组件的模板。

稍后我们将利用这种配置，但是现在让我们开始构建我们的项目。

在笔记本中，为您的项目创建一个新文件夹( **advert-pred** )。在那个文件夹中，创建一个新的 Jupyter 笔记本(**广告-实验**)并打开它。

在这个 Jupyter 笔记本中，您将单独并交互地浏览每个 TFX 组件。**这将帮助你理解每个组件在做什么**，然后在最后，你将把你的实验变成一个完整的管道并部署它。

你可以在这里得到完整的笔记本[，在这里](https://web.archive.org/web/20221201153827/https://github.com/risenW/tfx-adClickPrediction/blob/master/advert-experiment.ipynb)得到完整的项目代码[。](https://web.archive.org/web/20221201153827/https://github.com/risenW/tfx-adClickPrediction)

### **设置**

在笔记本的第一个单元中，你将安装 TFX、库比弗洛(kfp)和一个名为 skaffold 的软件包:

```py
# Install tfx and kfp Python packages.
import sys
!{sys.executable} -m pip install --user --upgrade -q tfx==0.22.0
!{sys.executable} -m pip install --user --upgrade -q kfp==1.0.0
# Download skaffold and set it 
!curl ///skaffold//latest/ /home//.local//Import packages

```

[**Skaffold**](https://web.archive.org/web/20221201153827/https://skaffold.dev/) **是一个命令行工具，方便 Kubernetes 应用程序的持续开发**。它帮助我们轻松管理和处理构建、推送和部署应用程序的工作流。以后你就会明白斯卡福德的用途了。

运行第一个单元后，您会得到一些警告——现在忽略它们。其中一个通知您您的安装不在您的 env 路径中。通过将它们添加到 PATH 中，您可以在下一个单元格中解决这个问题。

```py
# Set `PATH` to  ``
PATH=%env 
%env /home//.local/

```

接下来，设置一些重要的环境变量，Kubeflow 稍后将使用这些变量来编排管道。将下面的代码复制到新的单元格中:

```py
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GOOGLE_CLOUD_PROJECT=shell_output[0]
%env GOOGLE_CLOUD_PROJECT={GOOGLE_CLOUD_PROJECT}
print("GCP project ID:" + GOOGLE_CLOUD_PROJECT)

```

第一个变量是您的 **GCP 项目 ID** 。这可以从你的环境中访问，因为你在人工智能平台上。接下来是 Kubeflow 管道集群端点。

在部署管道时，**端点 URL** 用于访问 KFP 集群。要获得您的 KFP 端点，请从 Kubeflow“入门”页面复制 URL。

将复制的 URL 分配给变量**端点**:

```py
ENDPOINT='https://2adfdb83b477n893-dot-us-central2.pipelines.googleusercontent.com'

```

**复制端点 URL 时，确保删除 URL 的最后一个路径，并在. com 结尾停止复制。**

接下来，您将创建一个 Docker 名称，Skaffold 将使用它来捆绑您的管道。

```py
CUSTOM_TFX_IMAGE='gcr.io/' + GOOGLE_CLOUD_PROJECT + '/advert-pred-pipeline'

```

最后，您将设置基本路径，并将当前工作目录设置为项目文件夹。

```py
#set base 
BASE_PATH  
%cd 

```

**您完成了设置**。

接下来，您将开始对每个 TFX 组件进行交互式探索。

## 交互式运行 TFX 组件

TFX 附带了一个内置的 orchestrator，允许您在 Jupyter 笔记本中交互式地运行每个组件。

这有助于您轻松探索每个组件，并可视化输出。在探索结束时，您可以将代码作为管道导出。

现在，让我们来看看实际情况。在新的代码单元格中，导入以下包:

```py
import os
import pprint
import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()
import tfx
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
%load_ext tfx.orchestration.experimental.interactive.notebook_extensions.skip

```

在导入的包中，您会注意到您从 tfx 组件包中导入了不同的模块。这些组件将在 ML 流程的不同阶段按顺序用于定义管道。

最后一行代码(% **load_ext** )加载一个 tfx notebook 扩展，该扩展可用于标记从 notebook 自动生成管道时应该跳过的代码单元。

是啊！您可以自动将笔记本导出为管道，在 Apache Beam 或 Airflow 上运行。

### **上传您的数据**

在你的广告预测目录中，创建一个名为 data 的新文件夹。您将在此上传数据，供实验阶段使用。

在您继续之前，请从[这里](https://web.archive.org/web/20221201153827/https://drive.google.com/file/d/1dvT89N1f6ecmDEsIqvY667PssAbk_juI/view?usp=sharing)下载广告数据。

要将数据上传到您的笔记本实例，请打开您创建的数据文件夹，单击上传图标并选择您刚刚下载的数据文件。

上传完毕，让我们先来看一下最上面的几排。

```py
data_root = 'data'
data_filepath = os.path.join(data_root, "advertising.csv")
!head {data_filepath}

```

//输出

![TFX output 1](img/54b2634e9fc5b71d99848270e796e04d.png)

*Sample data output*

**数据集来自于本次挑战上的**[](https://web.archive.org/web/20221201153827/https://www.kaggle.com/fayomi/advertising#advertising.csv)****。任务是预测客户是否会点击广告。所以这是一个二元分类任务。****

**![](img/6ee4db9241456412a31ee02e477cd458.png)

这是一个关于构建 ML 管道的教程，因此，我不会涉及广泛的特征工程或分析。

让我们从 TFX 组件开始。

在您继续进行之前，首先您将创建一个叫做 InteractiveContext 的东西。InteractiveContext 将允许您在笔记本中以交互方式运行 TFX 组件，以便您可以可视化其输出。这与生产环境不同，在生产环境中，您将使用 orchestrator 来运行您的组件。

在新单元格中，创建并运行 InteractiveContext，如下所示:

```py
context = InteractiveContext()

```

### **范例生成**

您将使用的第一个组件是 ExampleGen。ExampleGen 通常位于 ML 管道的开头，因为它用于接收数据，分成训练集和评估集，将数据转换为高效的 [**tf。示例**](https://web.archive.org/web/20221201153827/https://www.tensorflow.org/tutorials/load_data/tfrecord) 格式，并且还将数据复制到一个托管目录中，以便于管道中的其他组件访问。

在下面的代码单元格中，您将把数据源传递给 ExampleGen 输入参数，并使用上下文运行它:

```py
example_gen = CsvExampleGen(input=external_input(_data_root))
context.run(example_gen)

```

上面，您可以查看 ExampleGen 输出的一个交互式小部件。这些输出被称为工件，ExampleGen 通常产生两个工件——**训练**示例和**评估**示例。

默认情况下，ExampleGen 将数据分成 2/3 的定型集，1/3 用于评估集。

您还可以查看存储工件和 URI 的位置:

```py
artifact = example_gen.outputs['examples'].get()[0]
print(artifact.split_names, artifact.uri)

```

//输出

![TFX output 2](img/5151515a94aebf99ba712aebc561a60d.png)

既然 **ExampleGen** 已经接收完数据，下一步就是数据分析。

### **StatisticsGen**

**StatisticsGen** 组件用于计算数据集的统计数据。这些统计数据提供了数据的快速概览，包括形状、存在的要素以及值分布等详细信息。

为了计算数据的统计信息，您将把 ExampleGen 的输出作为输入传入。

```py
statistics_gen = StatisticsGen(
   examples=example_gen.outputs['examples'])
context.run(statistics_gen)

```

这些统计数据可以使用上下文的 show 方法可视化，如下所示:

```py
context.show(statistics_gen.outputs['statistics'])

```

下面您将看到为数字特征生成的统计数据:

![numerical features statistics](img/0029028115d15d2bd70d4a47e22a6098.png)

*Statistics for numerical features*

如果您向下滚动，在 statistics 小部件中，您还会找到分类变量的描述。

![categorical variables](img/1a10c3dbf84606088e41e4e9a4c31d44.png)

*Statistics for categorical features*

使用 StatisticGen，您可以快速了解您的数据概况。您可以检查缺失、零的存在以及分布情况。

这些统计数据将被下游组件使用，如 **SchemaGen** 和 **ExampleValidator** ，用于在生产中接收新数据时检测异常、偏差和漂移。

### **SchemaGen**

[SchemaGen](https://web.archive.org/web/20221201153827/https://www.tensorflow.org/tfx/guide/schemagen) 组件将从统计数据中为您的数据生成一个模式。模式只是数据的一个定义。它从数据特征中定义类型、期望的属性、界限等等。

在下面的代码单元格中，您将把 StatisticsGen 输出传递给 SchemaGen 输入，然后可视化输出。

```py
schema_gen = SchemaGen(
   statistics=statistics_gen.outputs['statistics'],
   infer_feature_shape=False)
context.run(schema_gen)
context.show(schema_gen.outputs['schema'])

```

数据集中的每个要素都被表示，以及预期的类型、存在、化合价和域。

### **示例验证器**

ML 管道中的下一个组件是 ExampleValidator。该组件根据定义的模式验证您的数据并检测异常。

这可用于在生产中验证进入管道的任何新数据。在新数据输入到您的模型之前，它对于检测新数据中的漂移、变化和偏差非常有用。

在下面的代码单元格中，我们将 StatisticsGen 和 SchemaGen 输出传递给 ExampleValidator:

```py
example_validator = ExampleValidator(
   statistics=statistics_gen.outputs['statistics'],
   schema=schema_gen.outputs['schema'])
context.run(example_validator)
context.show(example_validator.outputs['anomalies'])

```

我们的数据目前没有异常。我们可以进入下一个部分。

### **变换**

管道中的下一个组件是转换组件。该组件对训练和服务数据执行特征工程。

要对接收到的数据执行转换，需要从 ExampleGen 传入数据，从 SchemaGen 传入模式，最后传入包含转换代码的 Python 模块。

在您的项目目录(advert-pred)中，创建一个名为 **model** 的新文件夹。在这个文件夹中，您将定义您的转换代码以及模型代码。

在模型文件夹中，创建三个脚本— **constants.py、advert-transform.py** 和 **__init__.py.**

要在 Jupyter Lab 中创建一个 Python 脚本，打开相应的文件夹，点击 **+** 图标，创建一个文本文件，然后将扩展名改为 **.py.**

在 **constants.py** 中，您将定义一些变量，如分类特征、数字特征以及需要编码的特征的名称。在我们的例子中，您将使用如下所示的一些选定功能:

```py
DENSE_FLOAT_FEATURE_KEYS = ['DailyTimeSpentOnSite', 'Age',                                     'AreaIncome', 'DailyInternetUsage' ]
VOCAB_FEATURE_KEYS = ['City', 'Male', 'Country' ]

VOCAB_SIZE = 1000

OOV_SIZE = 10

LABEL_KEY = 'ClickedOnAd'

def transformed_name(key):
   return key + '_xf'

```

**DENSE_FLOAT_FEATURE_KEYS** 代表所有数字特征，而 **VOCAB_FEATURE_KEYS** 特征包含所有你想要编码的字符串特征。

此外，您还添加了一个小的助手函数，它将把 **_xf** 附加到每个特性名称上。这在变换模块中用于区分变换后的要素和原始要素。

在 advert-transform.py 中，您将导入您的联系人，然后定义转换步骤。这是所有处理、清理、填充缺失值的代码所在的位置。

```py
import tensorflow as tf
import tensorflow_transform as tft
from model import constants

_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_LABEL_KEY = constants.LABEL_KEY
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_transformed_name = constants.transformed_name

def preprocessing_fn(inputs):
 """tf.transform's callback function for preprocessing inputs.
 Args:
   inputs: map from feature keys to raw not-yet-transformed features.
 Returns:
   Map from string feature key to transformed feature operations.
 """
 outputs = {}
 for key in _DENSE_FLOAT_FEATURE_KEYS:

   outputs[_transformed_name(key)] = tft.scale_to_z_score(
       inputs[key])

 for key in _VOCAB_FEATURE_KEYS:

   outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
       inputs[key],
       top_k=_VOCAB_SIZE,
       num_oov_buckets=_OOV_SIZE)
outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
return outputs

```

在 **advert-constants.py** 脚本的顶部，您初始化了在 **constants.py.** 中定义的常数。接下来，您定义了**预处理 _fn** 函数。该函数由**转换**组件调用，名字(**预处理 _fn** ) 要保留。

在**预处理 _fn** 函数中，您将根据每个特征的类型对其进行处理，对其进行重命名，然后将其追加到输出字典中。变换组件期望**预处理 _fn** 返回一个变换特征的字典。

**另外，注意我们的预处理代码是用纯 Tensorflow** 编写的。建议这样做，以便您的操作能够以最佳方式分布并在集群中运行。如果您想使用纯 Python 代码，尽管不推荐，您可以使用“[@ TF _ function](https://web.archive.org/web/20221201153827/https://www.tensorflow.org/api_docs/python/tf/function)”包装器将它们转换为 Tensorflow 的函数。

现在回到您的笔记本，在新的单元格中添加以下代码:

```py
advert_transform = 'model/advert-transform.py'
transform = Transform(
   examples=example_gen.outputs['examples'],
   schema=schema_gen.outputs['schema'],
   module_file=advert_transform)
context.run(transform)

```

首先，定义转换代码的路径，然后传递示例和模式。当您运行这个程序时，您应该会看到一个非常长的输出，显示了一些转换后的特性，最后，您可以看到如下所示的工件小部件:

转换组件将生成两个工件:一个 **transform_graph** 和 **transformed_examples** 。 **transform_graph** 将所有预处理步骤定义为一个有向无环图(DAG ),它可以用于任何摄取的新数据，而 **transformed_examples** 包含实际的预处理训练和评估数据。

您可以通过调用如下所示的转换输出来轻松查看这一点:

```py
transform.outputs

```

//输出

![TXF output 3](img/3ab458f5c4bd8757ce46996fb9998be7.png)

*Transform component output*

现在您已经获取、分析和转换了您的数据，您将定义下一个组件，称为**训练器**。

**教练**

### **训练器**组件用于训练 Tensorflow/Keras 中定义的模型。培训师将接受模式、转换后的数据和转换图、转换参数以及您的模型定义代码。

在您的 **model** 文件夹中，创建一个名为 **advert-trainer.py、**的新 Python 脚本，并添加以下代码。

上面的代码很长，所以我们将逐一介绍:

```py
import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from model import constants

_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_LABEL_KEY = constants.LABEL_KEY
_transformed_name = constants.transformed_name

def _transformed_names(keys):
 return [_transformed_name(key) for key in keys]

def _gzip_reader_fn(filenames):
 """Small utility returning a record reader that can read gzip'ed files."""
 return tf.data.TFRecordDataset(
     filenames,
     compression_type='GZIP')

def _get_serve_tf_examples_fn(model, tf_transform_output):
 """Returns a function that parses a serialized tf.Example and applies TFT."""
model.tft_layer = tf_transform_output.transform_features_layer()
@tf.function

def serve_tf_examples_fn(serialized_tf_examples):
   """Returns the output to be used in the serving signature."""
   feature_spec = tf_transform_output.raw_feature_spec()
   feature_spec.pop(_LABEL_KEY)
   parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
   transformed_features = model.tft_layer(parsed_features)
   return model(transformed_features)
return serve_tf_examples_fn

def _input_fn(file_pattern, tf_transform_output,
             batch_size=100):
 """Generates features and label for tuning/training.
Args:
   file_pattern: List of paths or patterns of input tfrecord files.
   tf_transform_output: A TFTransformOutput.
   batch_size: representing the number of consecutive elements of returned
     dataset to combine in a single batch
Returns:
   A dataset that contains (features, indices) tuple where features is a
     dictionary of Tensors, and indices is a single Tensor of label indices.
 """
 transformed_feature_spec = (
     tf_transform_output.transformed_feature_spec().copy())
dataset = tf.data.experimental.make_batched_features_dataset(
     file_pattern=file_pattern,
     batch_size=batch_size,
     features=transformed_feature_spec,
     reader=_gzip_reader_fn,
     label_key=_transformed_name(_LABEL_KEY))
return dataset

def _build_keras_model(hidden_units):
 """Creates a DNN Keras model for classifying taxi data.
 """
 real_valued_columns = [
     tf.feature_column.numeric_column(key, shape=())
     for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
 ]
 categorical_columns = [
     tf.feature_column.categorical_column_with_identity(
         key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
     for key in _transformed_names(_VOCAB_FEATURE_KEYS)
 ]
 indicator_column = [
     tf.feature_column.indicator_column(categorical_column)
     for categorical_column in categorical_columns
 ]
model = _wide_and_deep_classifier(
     wide_columns=indicator_column,
     deep_columns=real_valued_columns,
     dnn_hidden_units=hidden_units or [100, 70, 60, 50])
 return model

def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
 """returns a simple keras wide and deep model.
 """
 input_layers = {
     colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
     for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
 }

 input_layers.update({
     colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
     for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
 })
deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
 for numnodes in dnn_hidden_units:
   deep = tf.keras.layers.Dense(numnodes)(deep)

 wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
output = tf.keras.layers.Dense(
     1, activation='sigmoid')(
         tf.keras.layers.concatenate([deep, wide]))
model = tf.keras.Model(input_layers, output)
 model.compile(
     loss='binary_crossentropy',
     optimizer=tf.keras.optimizers.Adam(lr=0.01),
     metrics=[tf.keras.metrics.BinaryAccuracy()])
 model.summary(print_fn=absl.logging.info)
 return model

def run_fn(fn_args: TrainerFnArgs):
 """Train the model based on given args.
 Args:
   fn_args: Holds args used to train the model as name/value pairs.
 """

 first_dnn_layer_size = 150
 num_dnn_layers = 4
 dnn_decay_factor = 0.7
tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
 eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)
model = _build_keras_model(

     hidden_units=[
         max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
         for i in range(num_dnn_layers)
     ])
log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
 tensorboard_callback = tf.keras.callbacks.TensorBoard(
     log_dir=log_dir, update_freq='batch')
 model.fit(
     train_dataset,
     steps_per_epoch=fn_args.train_steps,
     validation_data=eval_dataset,
     validation_steps=fn_args.eval_steps,
     callbacks=[tensorboard_callback])
signatures = {
     'serving_default':
         _get_serve_tf_examples_fn(model,
                                   tf_transform_output).get_concrete_function(
                                       tf.TensorSpec(
                                           shape=[None],
                                           dtype=tf.string,
                                           name='examples')),
 }
model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

```

在导入部分，我们从常量模块导入一些您将使用的常量。

*   接下来，定义三个实用函数。第一个 **_transformed_names** 只是为每个特性返回一个修改后的名称。 **_gzip_reader_fn** 用于读取 TFRecordDataset 格式的文件。这就是我们的数据由 **ExampleGen** 表示的格式。最后是**_ get _ serve _ TF _ examples _ fn**，它解析每个 *tf。示例*并应用转换函数。
*   接下来，在 **_input_fn** 函数中，您只需生成一个 tf。来自变换要素的数据集文件。这是用于训练我们的模型的有效数据格式。
*   接下来的两个函数 **_build_keras_model** 和**_ wide _ and _ deep _ classifier**使用函数式 API 构建 keras 模型。这个函数式 API 在这里很有用，因为我们正在定义一个可以编排的静态图，因此，在编译模型之前，必须正确定义每个特性。
*   接下来，也是最重要的，您将定义 **run_fn** 。这个函数由训练器组件调用，因此名称不应该更改。在此函数中，您将从转换后的输出中初始化训练和评估数据集，初始化模型，为模型输出和张量板定义记录目录，最后拟合模型。

*   接下来，您定义服务签名，它被下一个组件**推送器**用于服务您的模型。
*   最后，您使用定义的签名将模型保存到服务目录中。
*   当您完成教练脚本的定义后，返回到您的笔记本并添加教练组件，如下所示:

培训师接受培训师模块、来自转换输出的转换示例、转换图、模式以及用于培训和评估步骤的培训师参数。

```py
advert_trainer = 'model/advert-trainer.py'
trainer = Trainer(
   module_file=advert_trainer,
   custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
   examples=transform.outputs['transformed_examples'],
   transform_graph=transform.outputs['transform_graph'],
   schema=schema_gen.outputs['schema'],
   train_args=trainer_pb2.TrainArgs(num_steps=1000),
   eval_args=trainer_pb2.EvalArgs(num_steps=500))
context.run(trainer)

```

现在您已经完成了培训，您将添加下一个组件——评估员。

求值程序

### 评估器组件计算评估集中的模型性能度量。它还可以用于验证任何新训练的模型。

当您在生产中改进和测试新型号时，这很有用。要设置评估器，您需要定义一个配置。

该配置只是指示评估者报告什么指标，在评估新模型时使用什么阈值，等等。更多关于这个[的内容请看这里](https://web.archive.org/web/20221201153827/https://www.tensorflow.org/tfx/model_analysis/get_started)。

将以下配置代码添加到您的笔记本中:

接下来，您将把此配置以及示例和训练模型输出传递给评估者，如下所示:

```py
eval_config = tfma.EvalConfig(
   model_specs=[tfma.ModelSpec(label_key='ClickedOnAd')],
   metrics_specs=[
       tfma.MetricsSpec(
           metrics=[
               tfma.MetricConfig(class_name='ExampleCount'),
               tfma.MetricConfig(class_name='BinaryAccuracy',
                 threshold=tfma.MetricThreshold(
                     value_threshold=tfma.GenericValueThreshold(
                         lower_bound={'value': 0.5}),
                     change_threshold=tfma.GenericChangeThreshold(
                         direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                         absolute={'value': -1e-10})))
           ]
       )
   ])
model_resolver = ResolverNode(
     instance_name='latest_blessed_model_resolver',
     resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
     model=Channel(type=Model),
     model_blessing=Channel(type=ModelBlessing))
context.run(model_resolver)

```

要可视化评估器的输出，请使用如下所示的 show 方法:

```py
evaluator = Evaluator(
   examples=example_gen.outputs['examples'],
   model=trainer.outputs['model'],
   baseline_model=model_resolver.outputs['model'],
   eval_config=eval_config)
context.run(evaluator)

```

上面，您可以看到评估者报告的指标。如果你训练一个新的模型，性能将与基线模型进行比较，在我们的例子中，基线模型不存在，因为这是我们的第一个模型。

```py
context.show(evaluator.outputs['evaluation'])

```

此外，评价者可以告诉我们一个模特是**被祝福的**还是**没有被祝福的**。一款**加持的**车型已经顺利通过所有评测标准，比现在的车型更好。然后这可以由**推动器**组件推动和服务，否则它抛出一个错误。这意味着您可以轻松地自动化模型部署。

要检查我们的当前模型是否得到了评估者的认可，请获取评估者的输出，如下所示:

//输出

```py
blessing_uri = evaluator.outputs.blessing.get()[0].uri
!ls -l {blessing_uri}

```

我们的模型被自动地祝福，因为它是我们管道中的第一个模型。如果您训练另一个模型，并重新运行评估器管道，那么它将与当前模型进行比较，并变得有福气或没有福气。

![TFX output 4](img/a2823604f59a21425a2667d3bc24ef6f.png)![](img/6ee4db9241456412a31ee02e477cd458.png)

既然您的模型已经被训练和认可，您可以使用 Pusher 组件轻松地将它导出到服务模型目录中。

**推动器**

### 在管道中总是排在最后的 Pusher 组件用于将一个受祝福的模型导出到服务目录。

要添加这个组件，您需要传入训练器输出、评估器输出以及服务目录。如下所示:

首先，您可以指定服务模型目录，您的训练模型将被推送到该目录。这可以是云存储或本地文件系统。

```py
serving_model_dir = 'serving_model/advert-pred'
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
    base_directory=serving_model_dir)))
    context.run(pusher)

```

**就这样！您已经成功实现了所有的 TFX 组件**，从获取数据到生成统计数据到生成模式，再到转换、模型训练、评估以及最终的模型保存。您可以轻松地导出笔记本，以便在 Apache Beam、Kubeflow 或 Apache Airflow 等流程编排中运行。

在下一节中，我将向您展示如何使用 Kubeflow 来编排您的管道。

设置您的 Kubeflow 管道

## 要使用 Kubeflow 运行或编排您的管道，您需要编写一些配置代码。这有助于设置 Kubeflow 以及定义要添加的 TFX 管道组件。按照以下步骤实现这一点:

首先，在项目目录中创建一个名为 **pipeline.py.** 的新 Python 脚本

*   在您的 **pipeline.py** 文件中，您将像我们在交互探索阶段所做的那样添加所有组件。

该文件定义了 TFX 管道和管道中的各种组件。

如果您注意到，上面的代码类似于您在交互式探索阶段编写的代码，这里您只需删除 InteractiveContext，并将每个组件添加到组件列表中。

```py
from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
import tensorflow_model_analysis as tfma

def create_pipeline(pipeline_name,
                   pipeline_root,
                   data_path,
                   preprocessing_fn,
                   run_fn,
                   train_args,
                   eval_args,
                   eval_accuracy_threshold,
                   serving_model_dir,
                   metadata_connection_config=None,
                   beam_pipeline_args=None,
                   ai_platform_training_args=None,
                   ai_platform_serving_args=None):
components = []
example_gen = CsvExampleGen(input=external_input(data_path))
 components.append(example_gen)
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
 components.append(statistics_gen)
schema_gen = SchemaGen(
     statistics=statistics_gen.outputs['statistics'],
     infer_feature_shape=True)
 components.append(schema_gen)
example_validator = ExampleValidator(
     statistics=statistics_gen.outputs['statistics'],
     schema=schema_gen.outputs['schema'])
 components.append(example_validator)
transform = Transform(
     examples=example_gen.outputs['examples'],
     schema=schema_gen.outputs['schema'],
     preprocessing_fn=preprocessing_fn)
 components.append(transform)
trainer_args = {
     'run_fn': run_fn,
     'transformed_examples': transform.outputs['transformed_examples'],
     'schema': schema_gen.outputs['schema'],
     'transform_graph': transform.outputs['transform_graph'],
     'train_args': train_args,
     'eval_args': eval_args,
     'custom_executor_spec':
         executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
 }

 if ai_platform_training_args is not None:
   trainer_args.update({
       'custom_executor_spec':
           executor_spec.ExecutorClassSpec(
               ai_platform_trainer_executor.GenericExecutor
           ),
       'custom_config': {
           ai_platform_trainer_executor.TRAINING_ARGS_KEY:
               ai_platform_training_args,
       }
   })

 trainer = Trainer(**trainer_args)
 components.append(trainer)
model_resolver = ResolverNode(
     instance_name='latest_blessed_model_resolver',
     resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
     model=Channel(type=Model),
     model_blessing=Channel(type=ModelBlessing))
 components.append(model_resolver)
eval_config = tfma.EvalConfig(
       model_specs=[tfma.ModelSpec(label_key='ClickedOnAd')],
       metrics_specs=[
           tfma.MetricsSpec(
               metrics=[
                   tfma.MetricConfig(class_name='ExampleCount'),
                   tfma.MetricConfig(class_name='BinaryAccuracy',
                   threshold=tfma.MetricThreshold(
                       value_threshold=tfma.GenericValueThreshold(
                           lower_bound={'value': 0.5}),
                       change_threshold=tfma.GenericChangeThreshold(
                           direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                           absolute={'value': -1e-10})))
               ]
           )
       ])
 evaluator = Evaluator(
     examples=example_gen.outputs['examples'],
     model=trainer.outputs['model'],
     baseline_model=model_resolver.outputs['model'],
     eval_config=eval_config)
 components.append(evaluator)

 pusher_args = {
     'model':
         trainer.outputs['model'],
     'model_blessing':
         evaluator.outputs['blessing'],
     'push_destination':
         pusher_pb2.PushDestination(
             filesystem=pusher_pb2.PushDestination.Filesystem(
                 base_directory=serving_model_dir)),
 }
 if ai_platform_serving_args is not None:
   pusher_args.update({
       'custom_executor_spec':
           executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                          ),
       'custom_config': {
           ai_platform_pusher_executor.SERVING_ARGS_KEY:
               ai_platform_serving_args
       },
   })
 pusher = Pusher(**pusher_args)  
 components.append(pusher)
return pipeline.Pipeline(
     pipeline_name=pipeline_name,
     pipeline_root=pipeline_root,
     components=components,
     metadata_connection_config=metadata_connection_config,
     beam_pipeline_args=beam_pipeline_args,
 )

```

为了创建管道，您从 **tfx.orchestrator** 导入管道，并传递所需的参数，这些参数只是管道名称、将存储所有输出的根目录、组件列表和元数据配置。

接下来，创建一个 Kubeflow runner 脚本( **kubeflow_dag_runner.py** )并粘贴下面的代码:

*定义 KubeflowDagRunner 以使用 Kubeflow 运行管道。*

这个脚本是特定于 Kubeflow 的，是通用的，所以你可以在你的项目中使用它。在这里，您可以定义所有变量，例如数据路径、输出的存储位置、管道名称以及训练和评估参数。

```py
import os
from absl import logging
import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils

try:
 import google.auth
 try:
   _, GOOGLE_CLOUD_PROJECT = google.auth.default()
 except google.auth.exceptions.DefaultCredentialsError:
   GOOGLE_CLOUD_PROJECT = ''
except ImportError:
 GOOGLE_CLOUD_PROJECT = ''
PIPELINE_NAME = 'advert_pred_pipeline'

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
PREPROCESSING_FN = 'model.advert-transform.preprocessing_fn'
RUN_FN = 'model.advert-trainer.run_fn'
TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 500
EVAL_ACCURACY_THRESHOLD = 0.5

OUTPUT_DIR = os.path.join('gs://', GCS_BUCKET_NAME)

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'advert_pred_pipeline_output',
                            PIPELINE_NAME)

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = 'gs://{}/advert-pred/data/'.format(GCS_BUCKET_NAME)

def run():
 """Define a kubeflow pipeline."""
metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
 tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
     kubeflow_metadata_config=metadata_config, tfx_image=tfx_image)

 pod_labels = kubeflow_dag_runner.get_default_pod_labels()
 pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'advert-pred'})
 kubeflow_dag_runner.KubeflowDagRunner(
     config=runner_config, pod_labels_to_attach=pod_labels
 ).run(
     pipeline.create_pipeline(
         pipeline_name=PIPELINE_NAME,
         pipeline_root=PIPELINE_ROOT,
         data_path=DATA_PATH,
         preprocessing_fn=PREPROCESSING_FN,
         run_fn=RUN_FN,
         train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_NUM_STEPS),
         eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_NUM_STEPS),
         eval_accuracy_threshold=EVAL_ACCURACY_THRESHOLD,
         serving_model_dir=SERVING_MODEL_DIR,
     ))
if __name__ == '__main__':
 logging.set_verbosity(logging.INFO)
 run()

```

这个脚本还包含一个叫做 **run 的重要函数。****运行**函数定义了成功执行 Kubeflow 管道的配置。它根据指定的参数实例化一个管道对象，然后执行它。

现在，再次回到您的笔记本实例。在你的互动探索后，在一个新的单元格中，将广告数据复制到你的谷歌云存储中。

如果您的数据已经在 GCS 中，请忽略这一点，只需在 pipeline.py 文件中指定路径。

![](img/6ee4db9241456412a31ee02e477cd458.png)

默认情况下，当您实例化 Kubeflow 时，已经为您的管道创建了一个云桶。您将在复制数据集时指定此路径。

可以导航到云存储[浏览器](https://web.archive.org/web/20221201153827/https://console.cloud.google.com/storage/browser)确认文件已经上传。

```py
## copy data to cloud storage for easy access from Kubeflow
!gsutil cp data/advertising.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/advert-pred/data/data.csv

```

接下来，设置**管道**变量名。这与 **Kubeflow_dag_runner.py** 脚本中的相同:

管道名称= '广告预测管道'

接下来，在新单元中，您将使用 tfx pipeline create 命令创建管道，如下所示:

注意，您指定了 Kubeflow runner 脚本、Kubeflow 实例的端点以及 Docker 图像名称。

运行上面的命令需要几分钟才能完成。这是因为 TFX 使用我们之前安装的 **skaffold** 包来构建管道的 docker 映像。

```py
!tfx pipeline create
--pipeline-path=kubeflow_dag_runner.py --endpoint={ENDPOINT} --build-target-image={CUSTOM_TFX_IMAGE} 
```

成功构建后，您会发现两个新文件(Dockerfile 和 build.yaml)。接下来，您将使用 tfx run 命令向 Kubeflow 提交这个管道作业。如下所示:

您的 Kubeflow 管道已成功推出！要对此进行监控，请转到您的 Kubeflow 实例页面(从这里复制端点 URL)，单击 **Experiments，**，然后选择您的管道名称。

```py
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

在执行时，您应该会看到一个管道图。如果有任何错误，它显示一个红色的失败图标，管道停止，否则它显示绿色，所有组件都被执行。

**您还可以在 Kubeflow 中查看每个输出的可视化效果。**这是从每个组件的输出中自动生成的。

例如，下面我们检查 **statisticsgen** 的输出。点击 **statisticsgen** 节点，选择可视化。

要查看生成的输出和保存的模型，您可以导航到您的 [GCS](https://web.archive.org/web/20221201153827/https://console.cloud.google.com/storage/) bucket。

如果您对您的管道进行更新，您可以使用下面的代码轻松地更新和运行它:

**就这样！您已经使用 TFX 和 Kubeflow** 成功地编排了一个端到端的 ML 管道。结合本教程中使用的工具，您可以轻松有效地构建整个 ML 工作流。

```py
!tfx pipeline update

!tfx run create 

```

摘要

在本教程中，您学习了如何使用 TFX 构建 ML 组件，如何在 AI 云平台上创建笔记本实例，如何交互式运行 TFX 组件，以及如何使用 Kubeflow 协调您的管道。

## 这是非常重要的知识，你可以开始在你的下一个公司或个人项目中使用。

希望对你有帮助。我等不及要看你造的东西了！

**本教程完整课程代码的链接可在[此处](https://web.archive.org/web/20221201153827/https://github.com/risenW/tfx-adClickPrediction)找到**。

参考

本教程旨在介绍 TensorFlow Extended (TFX)和 Cloud AI 平台管道，帮助你学会|[www.tensorflow.org](https://web.archive.org/web/20221201153827/http://www.tensorflow.org/)

TFX 是一个基于 TensorFlow 的谷歌生产规模的机器学习(ML)平台。它提供了一个配置|[www.tensorflow.org](https://web.archive.org/web/20221201153827/http://www.tensorflow.org/)

## 从命令行。对于这个特定的管道，查找带有前缀 workflow1(它的前缀)的服务，并注意|[cloud.google.com](https://web.archive.org/web/20221201153827/http://cloud.google.com/)

Apache Beam 为运行批处理和流数据处理作业提供了一个框架，这些作业可以在各种|[www.tensorflow.org](https://web.archive.org/web/20221201153827/http://www.tensorflow.org/)上运行

TFX is a Google-production-scale machine learning (ML) platform based on TensorFlow. It provides a configuration | [www.tensorflow.org](https://web.archive.org/web/20221201153827/http://www.tensorflow.org/)

From the command line. For this particular pipeline, look for the services with prefix workflow1 (its prefix), and note | [cloud.google.com](https://web.archive.org/web/20221201153827/http://cloud.google.com/)

Apache Beam provides a framework for running batch and streaming data processing jobs that run on a variety of | [www.tensorflow.org](https://web.archive.org/web/20221201153827/http://www.tensorflow.org/)**