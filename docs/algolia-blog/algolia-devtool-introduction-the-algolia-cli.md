# DevTool 简介:Algolia CLI！

> 原文：<https://www.algolia.com/blog/engineering/algolia-devtool-introduction-the-algolia-cli/>

```
Wake up Neo...
```

我们需要您为我们的药品索引重置配置！我们对高亮设置进行了修改，现在查询“红色药丸”的结果被高亮显示为蓝色，这真的非常令人困惑…

此外，我们有点担心所有经历过的现实都可能是机器运行的模拟，以奴役人类！

不过请先修复索引！

—

哦，伙计，你们不都喜欢在周一日出前被叫醒做这些事情吗？在理想情况下，这将是一个快速解决方案。随着我黑色风衣的挥动，我会从床上滚下来，蹲在房间中央，手里拿着笔记本电脑。我会按回车键跳过白兔，忽略敲我的门。我会用一只手调整太暗的夹鼻眼镜，而另一只手在几秒钟内发出命令。一些简短的，甜蜜的，中肯的。大概是这样…

```
algolia settings import prod_pharmaceuticals_index -F pharm_settings_snapshot.ndjson
```

看了一眼终端，显示出愉快的反应:

```
✓ Imported settings on prod_pharmaceuticals_index
```

很好……那么关于机器奴役人类的另一件事是什么？

——

### [](#meet-the-algolia-cli)迎接阿洛利亚 CLI！

如果你和我一样，命令行是你的朋友和信任的盟友。无论我是在测试一个快速的 API 调用还是在我最喜欢的 IDE 中编码，使用命令行都能让我流畅地迭代。此外，我喜欢通过命令行界面(CLI)做的任何事情都可以轻松地编写脚本并安排在我喜欢的任何时间运行。在我的 CLI 工具箱中有一套强大的工具，解决问题和自动化解决方案就像在公园里散步一样容易。

这就是为什么我们如此激动地宣布全新 Algolia CLI 工具的公开测试版发布，完整版即将推出！T31


Algolia CLI 使上传索引、自动化常见仪表板操作或保存和重新加载您的配置快照成为可能，只需通过命令行即可实现！不需要 API 客户端！

那么我该如何开始呢？别担心，您不必为此访问 oracle 开始使用 Algolia CLI 轻而易举！

如果你在 MacOS 上，只需在终端中运行以下命令，使用 homebrew 安装该工具:

```
brew install algolia/algolia-cli/algolia
```

想戳代码自己造？Algolia CLI 是完全开源的，在麻省理工学院的许可下，存在于一个公共的 github 存储库中(你可以在这里 [找到](https://github.com/algolia/cli) )！

Linux 和 Windows 版本即将发布！

让我们来看看一些具体的使用案例，看看 Algolia CLI 如何让您的生活变得更加轻松，这样您就可以专注于拯救世界了，明星！

——

### [](#example-0-set-your-active-algolia-application-with-profiles)示例#0:使用情景模式设置您的活动 Algolia 应用程序

任何 Algolia CLI 命令均可通过 `--admin-api-key [string]` 和 `--application-id [string]` 标志调用，以指定与哪个应用程序交互。为了使您能够发出命令，而不必总是在剪贴板中拖动 api-keys 和 app-id，您可以使用下面的命令来注册一个默认的概要文件。

```
algolia profile add --name [string] --app-id [string] --admin-api-key [string] --default
```

```
✓ Profile 'test_pharm_app' (##APP#ID##) successfully added and set as default.
```

当没有提供配置文件或 API 密钥/应用程序 ID 对时，`--default`标志告诉 CLI 在将来的命令中使用哪个应用程序。一次只能将一个应用程序设置为默认。花时间注册个人资料可以让未来的互动更加流畅，尤其是在使用 2 个或更多 Algolia 应用程序时。

您也可以通过运行来交互提供该命令的字段

```
algolia profile add
```

然后，该工具将提示您输入每个字段，一次一个，并根据指定添加应用程序。

要查看所有添加的应用程序列表，只需运行:

```
algolia profile list
```

```
NAME                  APP ID      NUMBER OF INDICES  DEFAULT
test_pharm_app        #APP#ID#1#  2                  ✓
playground_pharm_app  #APP#ID#2#  1
prod_pharm_app        #APP#ID#3#  15
```

牛逼！我们在这里看到我们添加了多个 Algolia 应用程序，应用程序 `test_pharm_app`是我们的默认应用程序！

注:应用简介保存在`~/.config/algolia/config.toml`

让我们上传一些数据吧！

——

### [](#example-1-create-an-index-and-upload-records-from-a-file)示例#1:创建索引并从文件上传记录

我想创建一个新的索引，给它一个相关的名称，并从本地系统上的一个文件中上传记录。CLI 能处理这种情况吗？当然可以！

```
algolia objects import new_index_name -F ./path_to_file.ndjson
```

您是否注意到我们既没有指定应用 id、api 密钥，也没有指定应用配置文件？在这种情况下，CLI 将根据默认应用程序(即上一步中指出的 `test_pharm_app`)进行操作。

我们还可以将`-F` 标志替换为`-` ，通过 stdin 将内容传递给 CLI 进行上传。让我们一起用管道发送一些命令吧！以下命令产生与上一个命令相同的结果(读取文件并将内容上传到指定的索引)，但是使用 cat 将文件的内容传递到 CLI 工具的 stdin ，而不是将文件路径作为参数传递。整洁！

```
cat ./path_to_file.ndjson  objects import new_index_name -F -
```

[什么是 ndjson？](https://ndjson.org/) 换行符分隔的 JSON 是 Algolia CLI 读取和写入文件的格式。这意味着，任何将 ndjson 格式的数据作为输出传递或作为输入接受的命令都可以与 Algolia CLI 命令连接在一起！我们将在下一个例子中看到更多

–

### [](#example-2-managing-index-settings-snapshots)例 2:管理索引设置快照

到目前为止，我们已经将 CLI 连接到我们的 Algolia 应用程序，并将一些数据上传到一个索引中！让我们更进一步，拍摄索引设置的快照，以便将来可以将它们恢复到健康的检查点。

哪些指标存在于我们默认的 app 中， `test_pharm_app?`

```
algolia index list
```

```
NAME               ENTRIES  SIZE    UPDATED AT  CREATED AT  LAST BUILD DURATION  PRIMARY  REPLICAS
pills_treatments   1,000    100 kB  1 day ago   1 day ago   2s                            []
pills_cures        0        0 B     2 days ago  2 days ago  3s                            []
```

让我们将索引 `pills_treatments` 的设置保存到系统上的一个文件中。

```
algolia settings get pills_treatments > ./pt_settings_snapshot.ndjson
```

快照已创建！现在，恢复到快照非常简单…

```
algolia settings import pills_treatments -F ./pt_settings_snapshot.ndjson
```

```
✓ Imported settings on pills_treatments
```

我们还可以在一行中将设置从一个索引转移到另一个索引！让我们将 `pills_treatments` 指标的设置复制到 `pills_cures` 指标中！在导入命令中，我们将使用 `-F` 标志后的 `-` 值来告诉 CLI 从 `stdin` 而不是指定的文件中读取输入。

```
algolia settings get pills_treatments  settings import pills_cures -F -
```

```
✓ Imported settings on pills_cures
```

简单！还有一个场景…我们已经在测试应用中的所有指数上测试过了，现在我们准备将这些设置迁移到我们的生产应用中！我们将 `prod_pharm_app` 配置文件中的 `pills_treatments_prod` 索引作为目标。

```
algolia settings get pills_treatments  settings import pills_treatments_prod -F - -p prod_pharm_app
```

```
✓ Imported settings on pills_treatments_prod
```

你开始相信了，是吗？

——

### [](#example-3-cicd-pipelines-that-execute-algolia-tasks)示例#3:执行 Algolia 任务的 CI/CD 管道

这是事情开始变得真正令人兴奋的地方。考虑以下工作流程:

1.  一个数据库包含你的数据。
2.  自动脚本每晚运行，根据数据库中的数据生成索引。
3.  一个自动脚本将该索引上传到 Algolia。

整齐！酷！方便！让我们改变这一点。我们会把事情搞砸一点。我们把一只虫子推到刺戳处怎么样？

*   工程师 Anderson 负责执行数据库迁移。他创造了改变！
*   工程师贝基负责审阅变更，她说 LGTM！
*   变更已提交！
*   CI/CD 管道将更改应用于测试数据库。没有错误！数据库迁移测试通过！
*   更改将应用于生产数据库。没有错误！数据库迁移测试通过！
*   工程师安德森面带微笑地签了字，回家，并帮他的女房东倒垃圾。

…

那天晚上，当工程师安德森睡得正香，梦见在飞行中的直升机下面晃来晃去时，灾难降临了！

…

*   自动索引脚本在夜间运行，根据数据库中的数据生成索引。
*   然而，数据库迁移转换了该脚本使用的字段。
*   输出索引缺少重要的搜索属性，但上传成功。
*   搜索格式错误的索引会产生不准确和不相关的结果。哦不！
*   直到 3 天后，随着搜索指标下降，问题才被发现。工程师 Anderson 又花了一天时间回滚更改并修复索引。

对此能做些什么？你如何防止这种混乱再次发生？

如果 DB changes 在 CI/CD 管道中运行了一些简单的查询测试用例会怎么样？每一个变化都会立即生成一个索引，并针对它启动一系列测试查询。工程师 Anderson 的更改会使 CI/CD 管道失败，因为它会产生与预期不同的查询结果。只需创建一些测试用例以及一个简单的脚本，针对这些测试用例运行 Algolia CLI 的 `algolia search` 命令，就可以提前检测到这个问题！

让我们配置一个这样的测试用例！当索引正常工作时，我们可以执行测试查询并保存结果。保存的数据是我们的“预期结果”，我们将根据它来比较后续查询测试的结果。我们将在前一个例子的索引中搜索“蓝色药丸”:

```
algolia search pills_treatments_prod -p prod_pharm_app --query "blue pills" > blue_pills_expected_results.ndjson
```

爽！我们有一个查询“蓝色药丸”的测试用例！让我们在测试 app 中的 `pills_treatments` 索引上运行这个， `test_pharm_app` ，看看是否匹配。

```
algolia search pills_treatments -p test_pharm_app --query "blue pills" > blue_pills_actual_results.ndjson && diff blue_pills_expected_results.ndjson blue_pills_actual_results.ndjson | wc -m
```

```
0
```

我们查询测试索引并保存结果。然后，我们通过调用 `diff` 命令将预期结果与实际结果进行比较，该命令只输出两个文件之间的差异。最后，我们将 `diff` 的输出传送到 `wc -m` 命令，该命令计算 `diff` 输出中的字符数。我们观察到输出为 0，这意味着两个文件之间没有差异，这表明我们的查询在两个索引中产生了相同的结果，并且我们的查询测试通过了！

按照这些思路，非零输出意味着这两个文件中存在一些差异，我们的查询测试未能在两个索引中产生相同的输出。

这只是如何将 Algolia CLI 集成到 CI/CD 管道中的一个例子。想象一下，使用索引设置管理工作流将营销团队设置的仪表板配置设置提升到生产应用程序。然后，促销可以调用一系列查询测试来验证更改，瞧！CLI 也成为您的业务用户的力量倍增器！    使用 Algolia CLI 自动执行任务，改善您的工作流程，并保持您的搜索体验无缝运行，即使您正在更改和开发内容！

——

### [](#get-involved)参与进来！

我们已经看到，Algolia CLI 几乎可以处理任何事情，从索引上传等简单操作到快照恢复等更复杂的工作流。请继续关注我们推出的更新，并进一步扩展 CLI 的功能，让您作为 Algolia 开发人员的生活更加轻松！

请记住，Algolia CLI 处于公开测试阶段！在 MacOS 上，用自制软件安装:

```
brew install algolia/algolia-cli/algolia
```

Windows 和 Linux 安装程序即将推出！

或者， [克隆 GitHub repo](https://github.com/algolia/cli) 自己构建！

想了解更多？查看[官方 CLI 文档](https://www.algolia.com/doc/tools/cli/get-started/overview/)获取指导教程！

别忘了在你的日历上标记 9 月 14 日至 15 日！Algolia DevCon 快速接近。我们将重点介绍 CLI 工具并展示其功能！开发者体验团队将在那里主持现场 Q & A，回答你所有的问题。你不会想错过的！

——

**弃用建议**–在创建这个 CLI 之前，我们有一个 [npm Algolia CLI 包](https://www.npmjs.com/package/@algolia/cli)提供类似的功能。传统 CLI**已弃用**——今后我们将不再维护/更新/支持它。但是，我们不会从 npm 中删除该包(因为客户仍在使用它)。相反，我们已经宣布弃用[传统 CLI GitHub 回购](https://github.com/algolia/algolia-cli-old)(并将回购重命名为“algolia-cli- **old** ”)，并将客户重新定向到[新 CLI GitHub 回购](https://github.com/algolia/cli)。