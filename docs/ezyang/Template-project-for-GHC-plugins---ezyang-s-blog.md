<!--yml

category: 未分类

date: 2024-07-01 18:17:26

-->

# GHC 插件模板项目：ezyang's 博客

> 来源：[`blog.ezyang.com/2012/09/template-project-for-ghc-plugins/`](http://blog.ezyang.com/2012/09/template-project-for-ghc-plugins/)

## GHC 插件模板项目

制作 Core 到 Core 转换的 GHC 插件涉及一些脚手架工作，因此我创建了一个小项目，基于[Max Bolingbroke 的示例](https://github.com/thoughtpolice/strict-ghc-plugin)，这是一个非常好的、干净的模板项目，你可以用它来创建自己的 GHC 插件。特别是，它包含了关于 GHC 源码的文档和指针，以及一个方便的 shell 脚本 `rename.sh MyProjectName`，可以让你轻松地将模板项目重命名为任何你想要的名称。你可以在 [GitHub 上找到它](https://github.com/ezyang/ghc-plugin-template)。随着项目的进展，我可能会继续添加更多内容；如果有任何 bug，请告诉我。
