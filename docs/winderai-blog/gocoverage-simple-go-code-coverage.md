# Go Coverage——简化 Go 代码覆盖率

> 原文：<https://winder.ai/gocoverage-simple-go-code-coverage/>

Go 在该语言的 1.5 版本中引入了 vendoring。供应商文件夹用作项目的依赖关系缓存。由于 Go 处理依赖关系的独特方式，缓存是来自整个存储库的完整代码；worts 和所有。Go 将在搜索全局 GOPATH 之前搜索 vendor 文件夹中的依赖项。

已经出现了工具来收集供应商文件夹，我最喜欢的工具之一是 [glide](https://glide.sh/) 。然而，有一件事还没有发生，那就是对 Go 的代码覆盖工具的改进。从 1.7 开始，任何为整个项目生成单一覆盖报告的尝试，*不包括*供应商文件夹，是不可能的。单个报告对于代码覆盖服务很重要，比如[套装](https://coveralls.io/)。

所以我开发了 [Gocoverage](https://github.com/philwinder/gocoverage) ，这是一个非常简单的工具，可以递归遍历项目中的所有目录(带有可选的过滤器)，使用标准的 Gocoverage 工具生成覆盖报告，并整理所有报告，以便上传到像工作服这样的服务。这个想法受到了没有过滤功能的 [gover](https://github.com/modocache/gover) 的强烈影响。但值得称赞的是。

例如，生成覆盖报告并上传到工作服非常简单:

```py
go get github.com/philwinder/gocoverage
go get github.com/mattn/goveralls
gocoverage
goveralls -coverprofile=profile.cov -repotoken=${COVERALLS_TOKEN} 
```

仅此而已。该命令默认过滤掉隐藏的目录和供应商文件夹。但是您可以通过 cli 参数随意更改。

[来看看](https://github.com/philwinder/gocoverage)。如果你的围棋项目需要任何帮助，请联系我们。