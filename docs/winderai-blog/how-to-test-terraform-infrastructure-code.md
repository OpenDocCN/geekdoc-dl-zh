# 如何测试 Terraform 基础设施代码

> 原文：<https://winder.ai/how-to-test-terraform-infrastructure-code/>

基础设施作为代码已经成为一种范式，但是基础设施脚本通常只编写和运行一次。这适用于简单的基础架构需求(例如 k8s 部署)。但是，当需要更多不同的基础设施或更大的弹性时，测试基础设施代码就成了一项要求。这篇博文介绍了一个当前的项目，该项目已经找到了处理这个问题的工具和模式。

![How to test terraform infrastructure code](img/4a67850f8dcbf33509edfdc4d804ca48.png)

## 介绍

我一直在做一个项目，这个项目包含了大量各种各样的基础设施。因为规模和快速发展，总有失败的成分。这导致了对代码完全缺乏信心。我们需要测试。

关键问题是基础设施代码几乎是不可测试的。我们可以执行的唯一测试是手动启动整个集群(或使用 CI 工具)，然后检查是否有任何错误。完整的集群需要几个小时才能建立起来，因为我们需要对一个遗留系统进行一些非常糟糕的引导。即使没有错误，也不能保证它能正常工作。谢天谢地，现在有工具可以帮忙了。

与基础设施专家交流

Winder 是云基础设施开发和运营方面的专家。我们所有的客户都依赖我们的专业知识来交付数据驱动的云原生产品。[联系我们获得我们的帮助](https://winder.ai/about/contact/)。

## terra Test:terra form 测试工具

我们花了几个月的时间在 github 上搜索 Terraform 测试框架。大约在 2018 年 4 月，我们获得了第一次成功。我们在 [Gruntworks](https://gruntworks.io) 的朋友开发了一种叫做 [Terratest](https://github.com/gruntwork-io/terratest/) 的新工具。目标是通过使用单元测试框架使 Terraform 代码可测试。他们选择使用 Go 的单元测试框架(他们可以选择任何东西)，然后开发 Go 库来执行常见的任务，比如 SSH-ing 到机器中。

## 如何使用 Terratest

要运行测试，您需要提供一个 Terraform 文件和一个 Go 测试。然后，您只需简单地调用`go test`，Terratest 就会负责对您的基础设施进行`init`和`apply`操作。然后它会执行一些测试。当然，这不是魔术。这个过程编码在 Go 测试文件中。但它是测试基础设施的一个简洁的可重复模式。

让我们来看一个浓缩的例子。更多示例可以在 Terratest 的[示例目录中找到。](https://github.com/gruntwork-io/terratest/tree/master/examples)

### 测试 Terraform 基础设施的示例

首先，我们需要用 Terraform 代码定义我们的基础设施。这包含了所有常见的东西，示例通常使用 AWS 资源，但任何云提供商都应该工作。

```py
resource "aws_instance" "example" {
  ami           = "${data.aws_ami.ubuntu.id}"
  instance_type = "t2.micro"

  tags {
    Name = "${var.instance_name}"
  }
}

... 
```

接下来，我们使用新的 Terratest 框架在 Go 中开发我们的测试。

```py
 terraformOptions := &terraform.Options{
		// The path to where our Terraform code is located  TerraformDir: "../examples/terraform-http-example",

		// Variables to pass to our Terraform code using -var options  Vars: map[string]interface{}{
			"aws_region":    awsRegion,
			"instance_name": instanceName,
			"instance_text": instanceText,
		},
	}

	// At the end of the test, run `terraform destroy` to clean up any resources that were created  defer terraform.Destroy(t, terraformOptions)

	// This will run `terraform init` and `terraform apply` and fail the test if there are any errors  terraform.InitAndApply(t, terraformOptions)

	// Run `terraform output` to get the value of an output variable  instanceURL := terraform.Output(t, terraformOptions, "instance_url")

	// It can take a minute or so for the Instance to boot up, so retry a few times  maxRetries := 30
	timeBetweenRetries := 5 * time.Second

	// Verify that we get back a 200 OK with the expected instanceText
 http_helper.HttpGetWithRetry(t, instanceURL, 200, instanceText, maxRetries, timeBetweenRetries) 
```

该测试初始化一些传递给 Terraform 的选项。接下来，我们使用 Go 的`defer`来推迟销毁函数的执行，直到作用域中的当前函数结束。接下来执行`apply`。最后，我们有一些来自 Terratest 库的帮助器代码，用于反复 ping web 服务器以获得响应。如果在要求的时间内没有响应，测试将失败。

Terratest 提供了一整套 [Go 模块](https://github.com/gruntwork-io/terratest/tree/master/modules)来帮助测试。例如 SSH 到一台机器，运行一个命令，复制文件或检查文件是否存在，或者提交 POST HTTP 请求。然而，还有进一步帮助功能的空间。我们的第一个 PR 支持使用 ssh-agent 而不是密钥来 SSH 到一台机器。

## 如何对地形模块进行单元测试

在更大的项目中，测试仍然是痛苦的。开发基础设施代码的主要问题是反馈周期太长。执行一次迭代可能需要几个小时。

解决一个大问题最常见的方法是把它分解成可以解决的小问题。在这种情况下，我们使用 Terraform 模块，它旨在封装更小的构建块。我对 Terraform 模块的问题是，它们最终看起来总是像巨大的传递函数，因为你需要灵活性；即大多数低级配置仍然由父代码完成，这限制了电源封装。

但是现在忽略这一点，我们仍然可以为单独的模块编写 Terratests。这些构成了我们对 Terraform 代码的“单元测试”。我们编写一个非常简单的`main.tf`来实例化我们的模块，并编写一个 Go 测试文件来测试该模块。目录布局示例可能如下所示:

```py
.
├── main.tf
├── modules
│   ├── bastion
│   │   └── main.tf
│   └── webserver
│       └── main.tf
└── tests
    └── unit
        ├── bastion
        │   ├── main.tf
        │   └── test.go
        └── webserver
            ├── main.tf
            └── test.go 
```

这里我们有一个简化的设置，根目录中的`main.tf`是模块的父模块。在这个文件中，我们导入了`bastion`和`webserver`模块。测试位于`tests`目录中。对于每个模块，我们都有一个由`main.tf`和`test.go`组成的单元测试。`main.tf`包含了导入该模块所需的所有依赖项，并且只包含该模块。`test.go`包含用于`apply`、测试和`destroy`模块的代码。

最后，我们可以使用`cd tests/unit ; go test -v ./...`在 CI 管道中运行所有这些代码，我们将获得一个全局通过/失败。

## 结论

这种设置隔离了基础设施的较小部分，如果您并行运行它们(Go 的测试框架可以并行运行)，那么这将缩短反馈周期。当然，测试只和最慢的测试一样快，所以通过将静态元素放入映像中可以加快部署速度。