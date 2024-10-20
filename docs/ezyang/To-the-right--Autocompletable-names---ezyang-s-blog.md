<!--yml

category: 未分类

date: 2024-07-01 18:18:29

-->

# 向右！可自动完成的名称：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/01/to-the-right-autocompletable-names/`](http://blog.ezyang.com/2010/01/to-the-right-autocompletable-names/)

## 向右！可自动完成的名称

在我年轻的时候，MM/DD/YYYY 的风格约定曾经让我困惑；为什么人们会选择这样一个不合逻辑的系统，将月份、日期和年份放在非层次化的顺序中？显然像 YYYY-MM-DD 这样的顺序会更合理：这种格式可以排序，总体来说相当合理。

不过，最终我不情愿地接受了 MM/DD/YYYY，它为了人性化牺牲了机器友好性；毕竟，年份条目很少改变，对人类来说，月份和日期是最重要的信息。通常情况下，上下文足以隐含地指定年份是多少。

但作为一个自动完成用户，我已经意识到，即使涉及计算机时，这种排序方式也能派上用场。考虑一下按层次命名和非按层次命名的文件列表：

```
# hierarchally named
test-algorithm.sh
test-bottles.sh
test-capistrano.sh
utils.sh

# non-hierarchally named
algorithm-test.sh
bottles-test.sh
capistrano-test.sh
utils.sh

```

在层次化情况下，要自动完成`test-algorithms.sh`，我需要输入`t<tab>a<tab>`；总共四个按键。然而，在非层次化情况下，我只需要输入`a<tab>`。如果我经常访问这些文件，额外的按键会累积起来。

因此，我提出一个请求：下次你考虑为存放在目录中的文件制定命名规范时，请考虑将“category”组件移动到末尾，并考虑友好的自动完成名称。你的手指会感谢你的这一举措。

（感谢 GameTeX 给我指引。）
