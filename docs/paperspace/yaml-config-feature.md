# 新功能:梯度 YAML 配置文件

> 原文：<https://blog.paperspace.com/yaml-config-feature/>

[2021 年 12 月 2 日更新:本文包含关于梯度实验的信息。实验现已被弃用，渐变工作流已经取代了它的功能。[请参见工作流程文档了解更多信息](https://docs.paperspace.com/gradient/explore-train-deploy/workflows)。]

您可以将首选的默认参数保存在 config.yaml 文件中，而不是每次想从 Gradient CLI 运行时都键入每个参数。如果您发现自己重复输入相同的命令，并且希望配置可重用(并且具有集成的版本控制)，那么这个工具就是为您准备的。

## 定义默认参数

在 YAML 配置文件中，您可以为特定任务定义最常用或首选的命令参数，并为不同的任务创建多个文件。以单节点实验为例。而不是写下这样的话:

```py
gradient experiments run singlenode \
  --projectId <your-project-id> \
  --name singleEx \
  --experimentEnv "{\"EPOCHS_EVAL\":5,\"TRAIN_EPOCHS\":10,\"MAX_STEPS\":1000,\"EVAL_SECS\":10}" \
  --container tensorflow/tensorflow:1.13.1-gpu-py3 \
  --machineType K80 \
  --command "python mnist.py" \
  --workspaceUrl https://github.com/Paperspace/mnist-sample.git \
  --modelType Tensorflow \
  --modelPath /artifacts
```

使用配置文件，您可以指定您的任务和您想要使用的特定配置。在这种情况下，您只需写:

```py
gradient experiments run singlenode --optionsFile config.yaml
```

## 生成模板配置文件

您可以通过指定想要运行的任务并使用`--createOptionsFile`标志来生成一个新的配置文件:

```py
gradient experiments run singlenode --createOptionsFile config_file_name.yaml
```

当您第一次创建配置文件时，它会自动创建一个所有潜在参数的列表，并将每个字段填充为空值。然后，您可以填写与您的任务相关的每个参数，将其他值保留为空，或者完全删除它们。

> **注意:**您当前不能通过从命令行重新定义来覆盖文件中定义的特定参数。如果您想要进行任何特定的参数更改，您必须更改 config.yaml 本身的值。

## 示例使用案例

您可以为不同的任务使用不同的配置文件。配置文件可用于运行笔记本、实验、TensorBoard，以及基本上任何其他带有可从 CLI 运行的参数的任务。除了允许使用 [GradientCI](https://docs.paperspace.com/gradient/projects/gradientci) 进行集成版本控制之外，这还简化了协作，因为您可以共享您的 config.yaml 文件供其他人使用。

例如，如果您想要部署您的模型，您可能会生成以下配置文件:

```py
deploymentType: TFServing
imageUrl: tensortensorflow/serving:latest-gpuflow/
instanceCount: 1
machineType: K80
modelId: mos3vkbikxc6c38
name: tfserving deployment
```

另一方面，如果您想要运行单节点实验，您的配置文件可能如下所示:

```py
command: python mnist.py
container: tensorflow/tensorflow:1.13.1-gpu-py3
experimentEnv:
  EPOCHS_EVAL: 5
  EVAL_SECS: 10
  MAX_STEPS: 1000
  TRAIN_EPOCHS: 10
machineType: K80
modelPath: /artifacts
modelType: Tensorflow
name: mnist-cli-local-workspace
projectId: pr64qlxl0
tensorboard: false
workspace: 'https://github.com/Paperspace/mnist-sample.git'
```

更多信息请查看文档。