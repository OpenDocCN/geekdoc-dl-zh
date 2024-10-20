<!--yml

category: 未分类

date: 2024-07-01 18:17:56

-->

# 多值逻辑与底部：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/03/many-valued-logics-and-bottom/`](http://blog.ezyang.com/2011/03/many-valued-logics-and-bottom/)

## 多值逻辑与底部

我翻阅了 Graham Priest 的《非经典逻辑导论》，其中的多值逻辑部分引起了我的注意。多值逻辑是具有非传统真值*true*和*false*之外的更多真值的逻辑。克里尼三值逻辑（强）建立了以下真值表，包括 0、1 和 x（被认为是既非真也非假的某个值）：

```
NOT
    1 0
    x x
    0 1

AND
      1 x 0
    1 1 x 0
    x x x 0
    0 0 0 0

OR
      1 x 0
    1 1 1 1
    x 1 x x
    0 1 x 0

IMPLICATION
      1 x 0
    1 1 x 0
    x 1 x x
    0 1 1 1

```

我一直认为多值逻辑有点像是为了处理自指悖论而“应急”的方式，但实际上，克里尼通过思考部分函数在未定义值上的应用来发明他的逻辑：一种符号失效的情况。因此，这些真值表对应于[表示语义预测的并行或与和运算符](http://blog.ezyang.com/2010/12/gin-and-monotonic/)并不令人意外。

读者被邀请考虑是否可以将此逻辑用于柯里-霍华德风格的对应；特别是，在 K3 中排中律是无效的。
