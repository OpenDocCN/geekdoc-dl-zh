<!--yml

category: 未分类

date: 2024-07-01 18:17:18

-->

# 责备树：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/08/blame-trees/`](http://blog.ezyang.com/2013/08/blame-trees/)

## 责备树

我刚在第 13 届算法与数据结构研讨会上介绍了*责备树*。责备树是一种功能性数据结构，通过融入关于结构任意部分“责备”的信息（类似于`git blame`），支持高效的合并操作。这是一篇理论论文，因此常数因子并不理想，但渐近性能比现代版本控制系统中使用的传统合并算法要好得多。

这是与[大卫·A·威尔逊](http://web.mit.edu/dwilson/www/)、[帕维尔·潘切哈](http://pavpanchekha.com/)和[埃里克·D·德迈恩](http://erikdemaine.org/)共同完成的工作。你可以查看[论文](http://ezyang.com/papers/demaine13-blametrees.pdf)，或查看[幻灯片。](http://ezyang.com/slides/ezyang13-blametrees-slides.pdf) 我还有一份稍早版本的演讲录像在[YouTube (20 minutes)](http://youtu.be/f8e-QE6Gus8)，我用它来从外地合作者那里获取反馈，然后才真正做演讲。还要感谢大卫·马兹雷斯亲自对演示稿提出有用的评论。
