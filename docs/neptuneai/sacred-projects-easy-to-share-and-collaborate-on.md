# 如何让你神圣的项目易于分享和协作

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/sacred-projects-easy-to-share-and-collaborate-on>

冗长的电子表格详述混乱的人工授精实验的日子已经一去不复返了。有了[神圣](https://web.archive.org/web/20221206133152/https://github.com/IDSIA/sacred)这样的平台，你就可以安心的记录实验了。神圣为您做了有益的工作:它跟踪您的参数、模型架构、数据集变更、培训工作、工件、度量、元、调试等等。

本质上，Sacred 是一个 Python 模块，允许您配置、组织、记录和再现实验。这样，您可以:

*   轻松管理您的实验参数
*   为你的实验做些设置
*   在 [MongoDB 数据库](https://web.archive.org/web/20221206133152/https://www.mongodb.com/)中保存单次运行的配置
*   复制结果

神圣的带有一个强大的命令行界面，在那里你可以改变参数，进行不同的实验和变种。通过它的“观察者”，它可以记录实验的各种细节——如依赖关系、配置、使用的机器或结果。这些信息可以进入数据库或其他实验跟踪工具。

有陷阱吗？好吧，尽管它很有价值，但神圣并没有一个完整的用户界面或关键的实验来追踪功能。

这就是 Omniboard 的用武之地。Omniboard 是神圣的孪生兄弟——你不能缺了一个使用另一个。Omniboard 是一个 NodeJS 服务器，连接到神圣的数据库，并可视化实验+指标/日志。您可以:

*   通过表格或列表视图访问实验管理功能
*   对比实验
*   审查关于实验的详细数据，例如度量图、源文件、工件或 git 散列/版本控制信息。

神圣和全能的结合可能是强大的，但是它们缺乏有价值的功能，而这些功能对于大规模团队来说是至关重要的。例如:

*   详细团队协作的特性
*   显示笔记本版本或笔记本自动快照的用户界面
*   可以保存实验视图或分组实验的用户界面
*   可以扩展到数百万次运行的用户界面
*   专门的用户支持

然而，不要失去希望。你可以很容易地获得所有这些功能，以及你所知道和喜爱的神圣功能。

## 介绍海王星

Neptune 是一个轻量级的 ML 实验管理工具。它灵活且易于与不同的工作流程集成。您的队友可以使用任何 ML 库和平台，共享结果，并在单个 Neptune 仪表板上进行协作。如果不想在自己的硬件上部署，可以使用他们的 web 平台。

海王星的主要特征是:

*   实验管理:跟踪你团队的所有实验，并对它们进行标记、过滤、分组、排序和比较
*   **笔记本版本和区分:**比较两个笔记本或同一笔记本中的检查点；与源代码类似，您可以进行并排比较
*   **团队协作:**添加评论，提及队友，比较实验结果

*更多详情请看- >* [*海王 vs 圣物+全能*](/web/20221206133152/https://neptune.ai/vs/sacred-omniboard)

## 海王星+神圣整合

Omniboard 是神圣的流行前端——然而，正如我们上面讨论的，它缺乏许多关键特性，特别是对于协作。另一方面，Neptune 让您继续使用神圣的日志 API，同时提供其圆滑、直观的 UI。这是 Omniboard 更实用的替代品。

在与 Sacred 集成时，Neptune 用自己的 Observer 替换了 MongoDB 后端。这样，您就不必建立数据库。你所有的数据都可以记录到云端或者本地，随你怎么想。

现在我们来看一下如何同步它们:

### Neptune 设置(如果您已经有一个 Neptune 帐户，请跳过)

1.  先注册一个[海王 AI](/web/20221206133152/https://neptune.ai/register) 账号。它对个人和非组织都是免费的，你可以获得 100 GB 的存储空间。
2.  通过点击右上角的菜单获取您的 API 令牌。

3.  创建一个 NEPTUNE_API_TOKEN 环境变量，并在控制台中运行它。export NEPTUNE _ API _ TOKEN = ' your _ API _ TOKEN '
4.  创建一个项目。在您的项目仪表板中，单击“新建项目”并填写以下信息。注意隐私设置！

### 综合

首先，您需要安装您的 [neptune 客户端](https://web.archive.org/web/20221206133152/https://github.com/neptune-ai/neptune-client):

```py
pip install neptune-client

```

确保创建一个实验:

```py
ex = Experiment('iris_rbf_svm')
```

然后，传递实验对象作为第一个参数:

```py
from neptunecontrib.monitoring.sacred import NeptuneObserver
ex.observers.append(NeptuneObserver(api_token='ANONYMOUS',
                                   project_name='shared/sacred-integration'))

```

确保用您自己的 API 令牌替换“ANONYMOUS ”(参见上文！)和项目名称。

在那之后，简单地像平常一样运行你的实验。现在，他们将在海王星训练！

## 与海王星合作

您的实验元数据现在应该存储在 Neptune 中，您可以在您的实验仪表板中查看它:

您可以通过添加标签和使用自定义过滤器对实验进行分组来自定义仪表板。

Neptune 让你只需发送一个链接就可以分享 ML 实验。它可以是:

Neptune 还带有[工作区](https://web.archive.org/web/20221206133152/https://docs.neptune.ai/administration/workspaces)，一个你可以管理项目、用户和订阅的中心枢纽；有个人和团队工作空间。

在团队工作区中，团队成员可以浏览与其分配的角色相关的内容。您可以在项目和工作区中分配各种角色。在团队工作区中，您可以邀请管理员或成员，每个人都有不同的权限。

可以在顶栏上的工作区名称中更改工作区设置:

在*概述、项目、人员和订阅*选项卡下，您可以看到工作场所设置:

项目中有三个角色:所有者、贡献者和查看者。根据角色的不同，用户可以运行实验、创建笔记本、修改以前存储的数据等。

*更多详情参见- >* [*用户管理*](https://web.archive.org/web/20221206133152/https://docs.neptune.ai/administration/user-management)

## 了解更多关于海王星的信息

如你所见，海王星很好地补充了神圣。您可以轻松替换 Omniboard 并添加 Neptune Observer 来解锁更多功能。海王星可以作为一个中心枢纽来管理你的实验，并与你的团队合作。

如果你想了解更多关于 Neptune 的知识，请查阅官方[文档](https://web.archive.org/web/20221206133152/https://docs.neptune.ai/)。如果你想尝试一下，[创建你的账户](https://web.archive.org/web/20221206133152/https://ui.neptune.ai/register/)，开始用 Neptune 跟踪你的机器学习实验。