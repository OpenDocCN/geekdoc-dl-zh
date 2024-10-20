# 机器学习实验管理:如何组织你的模型开发过程

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/experiment-management>

机器学习或深度学习[实验跟踪](/web/20220928194919/https://neptune.ai/experiment-tracking)是交付成功结果的关键因素。没有它，你不可能成功。

我来分享一个听了太多次的故事。

> *“所以我和我的团队开发了一个机器学习模型，经过几周的大量实验，我们**得到了有希望的结果**……*
> 
> *…不幸的是，我们无法确切地说出什么表现最好，因为**我们没有跟踪**功能版本，没有记录参数，并且使用不同的环境来运行我们的模型……*
> 
> *…几周后，**我们甚至不确定我们实际尝试了什么**，所以我们需要重新运行几乎所有的东西"*

听起来很熟悉？

在这篇文章中，我将向您展示如何跟踪您的机器学习实验，并组织您的模型开发工作，以便这样的故事永远不会发生在您身上。

**您将了解到:**

## 什么是机器学习实验管理？

机器学习环境中的实验管理是一个**跟踪实验元数据**的过程，例如:

*   代码版本，
*   数据版本，
*   超参数，
*   环境，
*   度量标准，

**以有意义的方式组织它们**，并使它们**可用于在您的组织内访问和协作**。

在接下来的部分中，您将通过示例和实现看到这到底意味着什么。

## 如何跟踪机器学习实验

我所说的**跟踪是指收集所有关于你的机器学习实验的元信息**,这是为了:

*   与团队(以及未来的你)分享你的成果和见解，
*   重现机器学习实验的结果，
*   保持你的结果，这需要很长时间才能产生，安全。

让我们一个接一个地检查我认为应该被记录下来的所有实验片段。

### 数据科学的代码版本控制

好的，在 2022 年，我认为几乎每个从事代码工作的人都知道版本控制。未能跟踪您的代码是一个很大的(但明显且容易修复的)疏忽。

我们应该继续下一部分吗？没那么快。

#### 问题 1: Jupyter 笔记本版本控制

很大一部分**数据科学发展发生在 Jupyter 笔记本**中，它不仅仅是代码。幸运的是，有一些工具可以帮助笔记本版本控制和区分。我知道的一些工具:

一旦你有了你的笔记本版本，我会建议你再多做一点，确保它从上到下运行。为此，您可以使用 jupytext 或 nbconvert:

```py
jupyter nbconvert --to script train_model.ipynb;
python train_model.py

```

#### 问题 2:脏提交实验

数据科学人员倾向于不遵循软件开发的最佳实践。你总能发现有人(包括我)会问:

但是如何跟踪提交之间的代码呢？如果有人在没有提交代码的情况下运行一个实验会怎么样？”

一种选择是明确禁止在脏提交(包含修改或未跟踪文件的提交)上运行代码。另一个选择是每当用户进行实验时，给他们一个额外的安全网和快照代码。

### 注意:

Neptune [记录你的 git 信息](https://web.archive.org/web/20220928194919/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#git-information) 同时跟踪一个 *运行* ，并提醒你 *运行* 是否在脏回购中启动。

### 跟踪超参数

大多数像样的机器学习模型和管道都调整了非默认超参数。这些可能是学习率、树的数量或缺失值插补方法。未能跟踪超参数会导致浪费数周时间寻找它们或重新训练模型。

好的一面是，**跟踪超参数非常简单**。让我们从人们倾向于定义它们的方式开始，然后我们将继续进行超参数跟踪:

#### 配置文件

通常是一个*。yaml* 文件，包含脚本运行所需的所有信息。例如:

```py
data:
    train_path: '/path/to/my/train.csv'
    valid_path: '/path/to/my/valid.csv'

model:
    objective: 'binary' 
    metric: 'auc'
    learning_rate: 0.1
    num_boost_round: 200
    num_leaves: 60
    feature_fraction: 0.2
```

#### 命令行+ argparse

您只需将参数作为参数传递给脚本:

```py
python train_evaluate.py \
    --train_path '/path/to/my/train.csv' \
    --valid_path '/path/to/my/valid.csv' \
    -- objective 'binary' \
    -- metric 'auc' \
    -- learning_rate 0.1 \
    -- num_boost_round 200 \
    -- num_leaves 60 \
    -- feature_fraction 0.2
```

#### main.py 中的参数字典

您将所有参数放在脚本中的字典中:

```py
TRAIN_PATH = '/path/to/my/train.csv' 
VALID_PATH = '/path/to/my/valid.csv'

PARAMS = {'objective': 'binary',
          'metric': 'auc',
          'learning_rate': 0.1,
          'num_boost_round': 200,
          'num_leaves': 60,
          'feature_fraction': 0.2}
```

#### 水螅

Hydra 是脸书开源开发的一个配置管理框架。

其背后的关键理念是:

*   动态地**创建一个**一个**层次化的** **配置** **由** **组成**，
*   需要时通过命令行覆盖它，
*   通过 CLI 传递新参数(配置中没有)——它们将被自动处理

Hydra 使您能够准备和覆盖复杂的配置设置(包括配置组和层次结构)，同时跟踪任何被覆盖的值。

为了理解它是如何工作的，让我们举一个 config.yaml 文件的简单例子:

```py
project: ORGANIZATION/home-credit
name: home-credit-default-risk
parameters:

	n_cv_splits: 5
	validation_size: 0.2
	stratified_cv: True
	shuffle: 1

	rf__n_estimators: 2000
	rf__criterion: gini
	rf__max_depth: 40
	rf__class_weight: balanced
```

只需调用 hydra decorator，就可以在应用程序中使用这种配置:

```py
import hydra
from omegaconf import DictConfig
@hydra.main(config_path='config.yaml')
def train(cfg):
	print(cfg.pretty())  
	print(cfg.parameters.rf__n_estimators)  
if __name__ == "__main__":
	train()
```

运行上述脚本将产生以下输出:

```py
name: home-credit-default-risk
parameters:
	n_cv_splits: 5
	rf__class_weight: balanced
	rf__criterion: gini
	rf__max_depth: 40
	rf__n_estimators: 2000
	shuffle: 1
	stratified_cv: true
	validation_size: 0.2
project: ORGANIZATION/home-credit
2000

```

要覆盖现有参数或添加新参数，只需将它们作为 CLI 参数传递即可:

```py
python hydra-main.py parameters.rf__n_estimators=1500 parameters.rf__max_features=0.2

```

***注意:**添加新参数必须关闭严格模式:*

```py
@hydra.main(config_path='config.yaml', strict=False)

```

Hydra 的一个缺点是，要共享配置或跨实验跟踪它，您必须手动保存 config.yaml 文件。

Hydra 正在积极开发中，请务必查看他们的最新文档。

#### 到处都是神奇的数字

每当你需要传递一个参数时，你只需传递该参数的一个值。

```py
...
train = pd.read_csv('/path/to/my/train.csv')

model = Model(objective='binary',
              metric='auc',
              learning_rate=0.1,
              num_boost_round=200,
              num_leaves=60,
              feature_fraction=0.2)
model.fit(train)

valid = pd.read_csv('/path/to/my/valid.csv')
model.evaluate(valid)
```

我们有时都会这样做，但这不是一个好主意，尤其是当有人需要接管你的工作时。

好吧，所以我确实喜欢*。yaml* 从命令行配置和传递参数(选项 1 和 2)，但是除了幻数之外的任何东西都可以。重要的是你**记录每个实验的参数**。

如果您决定将所有参数作为脚本参数**传递，请确保将它们记录在某个地方**。这很容易忘记，所以使用一个实验管理工具可以自动做到这一点，可以节省你的时间。

```py
parser = argparse.ArgumentParser()
parser.add_argument('--number_trees')
parser.add_argument('--learning_rate')
args = parser.parse_args()

experiment_manager.create_experiment(params=vars(args))
...

...
```

没有什么比**在一个完美的数据版本上拥有一个完美的脚本来产生完美的指标更痛苦的了，只是**发现你不记得作为参数传递的超参数**是什么了。**

 **#### **海王星**

Neptune 通过提供各种选项，使得在运行中跟踪超参数变得非常容易:

*   单独记录超参数:

```py
run["parameters/epoch_nr"] = 5
run["parameters/batch_size"] = 32
run["parameters/dense"] = 512
run["parameters/optimizer"] = "sgd"
run["parameters/metrics"] = ["accuracy", "mae"]
run["parameters/activation"] = "relu"  
```

*   将它们作为字典记录在一起:

```py
params = {
	"epoch_nr": 5,
	"batch_size": 32,
	"dense": 512,
	"optimizer": "sgd",
	"metrics": ["accuracy", "binary_accuracy"],
	"activation": "relu",
}

run["parameters"] = params

```

在上述两种情况下，参数都记录在*运行* UI 的*所有元数据*部分下:

```py
run["config_file"].upload("config.yaml")

```

该文件将被记录在*运行*界面的*所有元数据*部分下:

### 数据版本化

在实际项目中，数据会随着时间而变化。一些典型的情况包括:

*   添加新的图像，
*   标签得到了改进，
*   标签错误/错误的数据被移除，
*   发现了新的数据表，
*   新的特征被设计和处理，
*   验证和测试数据集会发生变化，以反映生产环境。

每当你的**数据改变**，你的分析、报告或者**实验结果的输出将可能改变**，即使代码和环境没有改变。这就是为什么要确保你在比较苹果和苹果，你需要**跟踪你的数据版本**。

拥有几乎所有的版本并得到不同的结果是非常令人沮丧的，**可能意味着浪费大量的时间(和金钱)**。可悲的是，事后你对此无能为力。所以，再一次，保持你的实验数据版本化。

对于绝大多数用例，每当有新数据进来时，您可以**将它保存在一个新位置，并记录这个位置和数据的散列**。即使数据非常大，例如在处理图像时，您也可以创建一个包含图像路径和标签的较小的元数据文件，并跟踪该文件的更改。

一位智者曾经告诉我:

> “存储很便宜，但在一个 8 GPU 的节点上训练一个模型两周就不便宜了。”

如果你仔细想想，记录这些信息并不一定是火箭科学。

```py
exp.set_property('data_path', 'DATASET_PATH')
exp.set_property('data_version', md5_hash('DATASET_PATH'))
```

你可以自己计算散列，使用一个简单的数据版本扩展(T1)或者将散列外包给一个成熟的数据版本工具，比如 T2 DVC T3。

您可以自己计算和记录散列，或者使用成熟的数据版本化工具，该工具为您提供了更强大的版本化功能。阅读以下市场上一些最佳工具的更多信息。

无论您决定哪个选项最适合您的项目**，请将您的数据**版本化。

### 跟踪模型性能指标

我从来没有发现自己在这种情况下认为我为我的实验记录了太多的指标，你呢？

**在现实世界的项目中，由于新的发现或不断变化的规范，您关心的指标可能会发生变化**,因此记录更多的指标实际上可以在将来为您节省一些时间和麻烦。

不管怎样，我的建议是:

> *“记录指标，全部记录”*

通常，指标就像一个简单的数字

```py
exp.send_metric('train_auc', train_auc)
exp.send_metric('valid_auc', valid_auc)
```

但我喜欢把它想得更宽泛一些。为了了解你的模型是否有所改进，你可能想看看图表、混淆矩阵或预测分布。在我看来，这些仍然是度量标准，因为它们帮助你衡量实验的表现。

```py
exp.send_image('diagnostics', 'confusion_matrix.png')
exp.send_image('diagnostics', 'roc_auc.png')
exp.send_image('diagnostics', 'prediction_dist.png')
```

### 注意:

在训练和验证数据集上跟踪指标**可以帮助您评估模型在生产中表现不佳的风险。差距越小，风险越低。Jean-Fran ois Puget 的 kaggle days 演讲是一个很好的资源。**

[https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/VC8Jc9_lNoY?feature=oembed](https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/VC8Jc9_lNoY?feature=oembed)

视频

此外，如果您正在处理在不同时间戳收集的数据，您可以评估模型性能衰减并**建议一个合适的模型再训练方案**。只需跟踪验证数据不同时间段的指标，并查看性能如何下降。

### 版本化实验环境

环境版本控制的大部分问题可以用一句臭名昭著的话来概括:

> “我不明白，它在我的机器上工作。”

有助于解决这个问题的一种方法可以称为 ***“环境作为代码”*** ，其中环境可以通过逐步执行指令( *bash/yaml/docker* )来创建。通过采用这种方法，您可以**从版本化环境切换到版本化环境设置代码**，我们知道如何做。

据我所知，在实践中有几个选项可以使用(这绝不是一个完整的方法列表)。

#### Docker 图像

这是首选方案，关于这个主题有很多资源。我特别喜欢的一个是杰夫·黑尔的“学足够多的 Docker 有用”系列。简而言之，您用一些指令定义 docker 文件。

```py
FROM continuumio/miniconda3

RUN pip install jupyterlab==0.35.6 && \
pip install jupyterlab-server==0.2.0 && \
conda install -c conda-forge nodejs

RUN pip install neptune-client && \
pip install neptune-notebooks && \
jupyter labextension install neptune-notebooks

ARG NEPTUNE_API_TOKEN
ENV NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN

ADD . /mnt/workdir
WORKDIR /mnt/workdir
```

您可以根据这些说明构建您的环境:

```py
docker build -t jupyterlab \
    --build-arg NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN .
```

您可以通过以下方式在环境中运行脚本:

```py
docker run \
    -p 8888:8888 \
    jupyterlab:latest \
    /opt/conda/bin/jupyter lab \
    --allow-root \
    --ip=0.0.0.0 \
    --port=8888
```

#### 康达环境

这是一个更简单的选择，在许多情况下，它足以管理您的环境，不会出现任何问题。它不像 docker 那样给你很多选择或保证，但对你的用例来说已经足够了。环境可以定义为一个*。yaml* 配置文件如下:

```py
name: salt

dependencies:
   - pip=19.1.1
   - python=3.6.8
   - psutil
   - matplotlib
   - scikit-image

- pip:
   - neptune-client==0.3.0
   - neptune-contrib==0.9.2
   - imgaug==0.2.5
   - opencv_python==3.4.0.12
   - torch==0.3.1
   - torchvision==0.2.0
   - pretrainedmodels==0.7.0
   - pandas==0.24.2
   - numpy==1.16.4
   - cython==0.28.2
   - pycocotools==2.0.0
```

您可以通过运行以下命令来创建 conda 环境:

```py
conda env create -f environment.yaml

```

非常酷的是，您总是可以通过运行以下命令将环境状态转储到这样的配置中:

```py
conda env export > environment.yaml

```

简单，完成工作。

#### 生成文件

您总是可以在 Makefile 中显式定义所有 bash 指令。例如:

```py
git clone git@github.com:neptune-ml/open-solution-mapping-challenge.git
cd open-solution-mapping-challenge

pip install -r requirements.txt

mkdir data
cd data
curl -0 https://www.kaggle.com/c/imagenet-object-localization-challenge/data/LOC_synset_mapping.txt
```

并通过运行以下命令进行设置:

```py
source Makefile

```

阅读这些文件通常很困难，你放弃了 conda 和/或 docker 的大量附加功能，但没有比这更简单的了。

现在，您已经将您的环境定义为代码，确保**为每个实验**记录环境文件。

同样，如果您使用的是实验管理器，您可以在创建新实验时对代码进行快照，即使您忘记了 git commit:

```py
experiment_manager.create_experiment(upload_source_files=['environment.yml')
...

...
```

并将它安全地存储在应用程序中:

### 版本化机器学习模型

现在，您已经使用模型的最佳超参数对模型进行了训练，并对数据、超参数和环境进行了记录和版本化。但是模型本身呢？在大多数情况下，训练和推理发生在不同的地方(脚本/笔记本)，您需要能够将您训练的模型用于其他地方的推理。

有两种基本方法可以做到这一点:

#### 1.将模型保存为二进制文件

您可以将模型导出为二进制文件，并在需要进行推理的地方从二进制文件中加载它。

有多种方法可以做到这一点——像 [PyTorch](https://web.archive.org/web/20220928194919/https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) 和 [Keras](https://web.archive.org/web/20220928194919/https://www.tensorflow.org/guide/keras/save_and_serialize) 这样的库有自己的保存和加载方法，而深度学习之外的 [Pickle](https://web.archive.org/web/20220928194919/https://docs.python.org/3/library/pickle.html) 仍然是从文件中保存和加载模型的最流行的方法:

```py
import pickle

with open(“saved_model.pkl”, “wb”) as f:
	pickle.dumps(trained_model, f)

with open(“saved_model.pkl”, “rb”) as f:
	model = pickle.load(f)

```

由于模型被保存为文件，您可以使用文件版本控制工具，如 git，或者将文件上传到实验跟踪器，如 Neptune:

```py
run[“trained_model”].upload(“saved_model.pkl”)

```

#### 2.使用模型注册表

模型注册中心是发布和访问模型的中央存储库。在这里，ML 开发人员可以将他们的模型推给其他利益相关者或他们自己在以后使用。

目前可用的一些流行的模型注册中心有:

**a)海王星:**

Neptune 是 MLOps 的元数据存储库，为运行大量实验的研究和生产团队而构建。

它为您提供了一个中心位置来记录、存储、显示、组织、比较和查询机器学习生命周期中生成的所有元数据。

个人和组织使用 Neptune 进行实验跟踪和模型注册，以控制他们的实验和模型开发。

海王星提供:

**b) MLflow:**

MLflow 模型注册中心是当今市场上为数不多的开源模型注册中心之一。你可以决定在你的基础设施上管理这个 T1，或者在像 T4 数据块 T5 这样的平台上使用 T2 完全管理的实现 T3。

MLflow 提供:

*   **注释和描述工具**用于标记模型，提供文档和模型信息，例如模型的注册日期、注册模型的修改历史、模型所有者、阶段、版本等；
*   **，odel versioning** 更新时自动跟踪注册模型的版本；
*   一个 **API 集成**，将机器学习模型作为 RESTful APIs，用于在线测试、仪表板更新等；
*   **CI/CD 工作流程集成**记录阶段转换、请求、审查和批准变更，作为 CI/CD 管道的一部分，以实现更好的控制和治理；
*   一个**模型阶段特性**,为每个模型版本分配预设或定制的阶段，如“阶段”和“生产”来代表模型的生命周期；
*   **促销方案配置**方便在不同阶段之间移动模型。

**c)亚马逊 Sagemaker 模型注册中心**

亚马逊 SageMaker 是一个完全托管的服务，开发者可以在 ML 开发的每一步使用它，包括模型注册。[模型注册中心](https://web.archive.org/web/20220928194919/https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html%5C)是 SageMaker 中[MLOps 套件](https://web.archive.org/web/20220928194919/https://aws.amazon.com/sagemaker/mlops/)的一部分，该套件通过在整个组织中**自动化**和**标准化** MLOps 实践来帮助用户构建和操作机器学习解决方案。

使用 SageMaker 模型注册表，您可以执行以下操作:

*   **生产用目录型号**；
*   管理**型号版本**；
*   **将元数据**，例如训练度量，与模型相关联；
*   管理模型的**审批状态；**
*   将模型部署到生产中；
*   使用 CI/CD 自动进行模型部署。

## 如何组织你的模型开发过程？

尽管我认为跟踪实验和确保工作的可重复性很重要，但这只是难题的一部分。一旦你跟踪了数百次实验，你将很快面临新的问题:

*   如何搜索和可视化所有这些实验，
*   如何将它们组织成你和你的同事可以消化的东西，
*   如何在您的团队/组织内部共享和访问这些数据？

这就是实验管理工具真正派上用场的地方。他们让你:

*   过滤/分类/标记/分组实验，
*   可视化/比较实验运行，
*   共享(应用程序和编程查询 API)实验结果和元数据。

例如，通过发送链接，我可以分享机器学习实验的[比较以及所有可用的附加信息。](https://web.archive.org/web/20220928194919/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/experiments?compare=GwGgjOkMwgTEA&split=bth&dash=charts&viewId=eccd5adf-42b3-497e-9cc2-9fa2655429b3&query=((%60sys%2Ftags%60%3AstringSet%20CONTAINS%20%22keras%22))%20AND%20(last(%60metrics%2Fepoch%2Faccuracy%60%3AfloatSeries)%20%3E%200.87)&sortBy=%5B%22metrics%2Fepoch%2Fval_accuracy%22%5D&sortDirection=%5B%22descending%22%5D&sortFieldType=%5B%22floatSeries%22%5D&sortFieldAggregationMode=%5B%22last%22%5D&suggestionsEnabled=true&lbViewUnpacked=true)

有了这些，您和您团队中的所有人就能确切地知道在模型开发中会发生什么。它使跟踪进度、讨论问题和发现新的改进想法变得容易。

### 在创造性迭代中工作

像这样的工具非常有用，是对电子表格和笔记的巨大改进。然而，我认为可以让你的机器学习项目更上一层楼的是一种专注的实验方法，我称之为创造性迭代。

我想从一些伪代码开始，稍后再解释:

```py
time, budget, business_goal = business_specification()

creative_idea = initial_research(business_goal)

while time and budget and not business_goal:
   solution = develop(creative_idea)
   metrics = evaluate(solution, validation_data)
   if metrics > best_metrics:
      best_metrics = metrics
      best_solution = solution
   creative_idea = explore_results(best_solution)

   time.update()
   budget.update()
```

在每个项目中，都有一个创建**业务规范**的阶段，通常需要机器学习项目的**时间框架、预算和目标**。当我说目标时，我指的是一组 KPI，业务指标，或者如果你超级幸运的话，机器学习指标。在这个阶段，管理业务预期非常重要，但这是以后的事了。如果你对这些东西感兴趣，我建议你看看凯西·科济尔科夫的一些文章，比如，[这篇](https://web.archive.org/web/20220928194919/https://medium.com/hackernoon/ai-reality-checklist-be34e2fdab9)。

假设你和你的团队知道商业目标是什么，你就可以做**初始研究**并制定一个基线方法，一个第一**创意想法**。然后你**开发**它并提出**解决方案**，你需要**评估**并得到你的第一套**指标**。如前所述，这些数据不一定是简单的数字(通常不是)，也可以是图表、报告或用户研究结果。现在，您应该研究您的**解决方案、指标和 explore_results** 。

您的项目可能会在这里结束，因为:

*   您的第一个解决方案**足够好**来满足业务需求，
*   你可以合理地预期**没有办法在先前假定的时间和预算内达到业务目标**，
*   你发现在附近的某个地方有一个**低挂水果的问题，你的团队应该把精力集中在那里。**

如果以上都不适用，你列出你的**解决方案**中所有表现不佳的部分，找出哪些可以改进，哪些**创意**可以帮你实现。一旦你有了这个清单，你需要根据预期的**目标**改进和**预算**对它们进行优先排序。如果您想知道如何评估这些改进，答案很简单:**结果探索**。

你可能已经注意到结果探索出现了很多。这是因为它非常重要，值得拥有自己的一部分。

### 模型结果探索

这是这个过程中极其重要的一部分。您需要**彻底了解当前方法的失败之处**，您距离目标的时间/预算还有多远，在生产中使用您的方法会有什么风险。实际上，这一部分并不容易，但掌握它非常有价值，因为:

*   它导致对业务问题的理解，
*   它导致关注重要的问题，并为团队和组织节省大量时间和精力，
*   它导致发现新的商业见解和项目想法。

**目前使用的一些流行的模型解释工具有:**

**SHAP(SHapley Additive explaints)**是一种解释任何机器学习模型输出的博弈论方法。它将最优信用分配与使用博弈论及其相关扩展的经典 Shapley 值的本地解释联系起来。

阅读如何在他们的[文档](https://web.archive.org/web/20220928194919/https://shap.readthedocs.io/en/latest/index.html)中使用 SHAP。

局部可解释模型不可知解释(LIME)是一篇论文，作者在其中提出了局部代理模型的具体实现。代理模型被训练来近似底层黑盒模型的预测。LIME 不是训练一个全局代理模型，而是专注于训练局部代理模型来解释个体预测。当前的 [Python 实现](https://web.archive.org/web/20220928194919/https://github.com/marcotcr/lime)支持表格、文本和图像分类器。

这是一个用于解释 scikit-learn 的决策树和随机森林预测的包。允许将每个预测分解成偏差和特征贡献分量。在这里学习用法[。](https://web.archive.org/web/20220928194919/https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)

**我找到的一些关于这个主题的好资源有:**

*   Gael Varoquaux 的 PyData 演讲“理解和诊断你的机器学习模型”

 [https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/kbj3llSbaVA?version=3&rel=1&showsearch=0&showinfo=1&iv_load_policy=1&fs=1&hl=en-US&autohide=2&wmode=transparent](https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/kbj3llSbaVA?version=3&rel=1&showsearch=0&showinfo=1&iv_load_policy=1&fs=1&hl=en-US&autohide=2&wmode=transparent)

视频

*   伊恩·奥斯瓦尔德的《创造正确而有能力的分类器》

 [https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/DkLPYccEJ8Y?version=3&rel=1&showsearch=0&showinfo=1&iv_load_policy=1&fs=1&hl=en-US&autohide=2&wmode=transparent](https://web.archive.org/web/20220928194919if_/https://www.youtube.com/embed/DkLPYccEJ8Y?version=3&rel=1&showsearch=0&showinfo=1&iv_load_policy=1&fs=1&hl=en-US&autohide=2&wmode=transparent)

视频

深入探索结果是另一个故事，也是另一篇博文，但关键的一点是，投入时间**了解您当前的解决方案对您的业务极其有益**。

## 最后的想法

在这篇文章中，我解释道:

*   什么是实验管理，
*   组织您的模型开发过程如何改进您的工作流程。

对我来说，将**实验管理工具**添加到我的“标准”软件开发最佳实践中是一个**顿悟时刻**，这使得我的机器学习项目更有可能成功。我想，如果你试一试，你会有同样的感觉。

### 雅各布·查肯

大部分是 ML 的人。构建 MLOps 工具，编写技术资料，在 Neptune 进行想法实验。

### 西达丹·萨达特

我目前是 Neptune.ai 的一名开发人员，我坚信最好的学习方式是边做边教。

* * *

**阅读下一篇**

## 真实世界的 MLOps 示例:超因子中的模型开发

6 分钟阅读|作者斯蒂芬·奥拉德勒| 2022 年 6 月 28 日更新

在“真实世界的 MLOps 示例”系列的第一部分中，[MLOps 工程师 Jules Belveze](https://web.archive.org/web/20220928194919/https://www.linkedin.com/in/jules-belveze) 将带您了解 [Hypefactors](https://web.archive.org/web/20220928194919/https://hypefactors.com/) 的模型开发流程，包括他们构建的模型类型、他们如何设计培训渠道，以及您可能会发现的其他有价值的细节。享受聊天！

### 公司简介

[Hypefactors](https://web.archive.org/web/20220928194919/https://hypefactors.com/) 提供一体化媒体智能解决方案，用于管理公关和沟通、跟踪信任度、产品发布以及市场和金融情报。他们运营着大型数据管道，实时传输世界各地的媒体数据。人工智能用于许多以前手动执行的自动化操作。

### 嘉宾介绍

#### 你能向我们的读者介绍一下你自己吗？

嘿，斯蒂芬，谢谢你邀请我！我叫朱尔斯。我 26 岁。我在巴黎出生和长大，目前住在哥本哈根。

#### 嘿朱尔斯！谢谢你的介绍。告诉我你的背景以及你是如何成为催眠师的。

我拥有法国大学的统计学和概率学士学位以及普通工程学硕士学位。除此之外，我还毕业于丹麦的丹麦技术大学，主修深度学习的数据科学。我对多语言自然语言处理非常着迷(并因此专攻它)。在微软的研究生学习期间，我还研究了高维时间序列的异常检测。

今天，我在一家名为 Hypefactors 的媒体智能技术公司工作，在那里我开发 NLP 模型，帮助我们的用户从媒体领域获得洞察力。对我来说，目前的工作是有机会从原型一直到产品进行建模。我想你可以叫我书呆子，至少我的朋友是这么形容我的，因为我大部分空闲时间不是编码就是听迪斯科黑胶。

### 超因子模型开发

#### 你能详细说明你在 Hypefactors 建立的模型类型吗？

尽管我们也有计算机视觉模型在生产中运行，但我们主要为各种用例构建 [NLP(自然语言处理)](https://web.archive.org/web/20220928194919/https://neptune.ai/blog/category/natural-language-processing)模型。我们需要覆盖多个国家和处理多种语言。多语言方面使得用“经典机器学习”方法开发变得困难。我们在[变形金刚库](https://web.archive.org/web/20220928194919/https://github.com/huggingface/transformers)的基础上打造深度学习模型。

我们在生产中运行各种模型，从跨度提取或序列分类到文本生成。这些模型旨在服务于不同的用例，如主题分类、情感分析或总结。

[Continue reading ->](/web/20220928194919/https://neptune.ai/blog/mlops-examples-model-development-in-hypefactors)

* * ***