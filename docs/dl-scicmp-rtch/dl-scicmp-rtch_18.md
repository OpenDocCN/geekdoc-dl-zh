# 13 加载数据

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/data.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/data.html)

我们将在下一章看到我们的玩具示例的第三个（也是最后一个）版本，该模型在一个很小的数据集上训练——小到可以一次性将所有观测值传递给模型。如果情况不是这样呢？比如说我们有 10,000 个项目，每个项目都是一个 256 x 256 像素大小的 RGB 图像。即使在非常强大的硬件上，我们也不可能一次性在完整的数据上训练模型。

因此，深度学习框架如 `torch` 包含一个输入管道，允许你以 *批处理* 的形式将数据传递给模型——也就是说，观测值的子集。在这个过程中涉及两个类：`dataset()` 和 `dataloader()`。在我们查看如何构建这些实例之前，让我们通过它们的作用来描述它们。

## 13.1 数据 vs. `dataset()` vs. `dataloader()` – 有何区别？

在本书中，“dataset”（变量宽度字体，无括号），或简称为“数据”，通常指的是像 R 矩阵、`data.frame`以及其中包含的内容。然而，`dataset()`（固定宽度字体，括号），是一个`torch`对象，它知道如何做一件事：*向调用者提供一个* *单个项目*。这个项目通常是一个列表，包含一个输入和一个目标张量。（它可以是任何东西，只要对任务有意义。例如，如果输入和目标是相同的，它可能是一个单独的张量。或者，如果需要将不同的输入传递给不同的模块，它可能包含超过两个张量。）

只要它满足上述声明的合同，`dataset()` 就可以自由地做任何需要做的事情。例如，它可以从互联网上下载数据，将它们存储在某个临时位置，进行一些预处理，并在需要时，以特定类模型期望的形状返回数据的小块。无论它在后台做什么，调用者所关心的只是它返回一个单独的项目。它的调用者，就是 `dataloader()`。

`dataloader()` 的作用是以 *批处理* 的形式向模型提供输入。一个直接的原因是计算机内存：大多数 `dataset()` 都会太大，无法一次性传递给模型。但是，批处理还有其他好处。由于梯度是在每个 *批处理* 中计算（并且模型权重更新）的，因此这个过程具有固有的随机性，这种随机性有助于模型训练。我们将在下一章中更多地讨论这一点。

## 13.2 使用 `dataset()`

`dataset()` 有各种风味，从现成的——由某些包提供，比如 `torchvision` 或 `torchdatasets`，或者任何选择以 `torch` 准备形式提供数据访问的包——到完全自定义（由你制作）。创建 `dataset()` 很简单，因为它们是 R6 对象，并且只需要实现三个方法。这些方法是：

1.  `initialize(...)`. 当实例化 `dataset()` 时传递给 `initialize()` 的参数。可能包括但不限于对 R `data.frame` 的引用、文件系统路径、下载 URL 以及 `dataset()` 预期的任何配置和参数化。

1.  `.getitem(i)`. 这是负责履行合同的方法。它返回的任何内容都算作一个单独的项目。参数 `i` 是一个索引，在许多情况下，它将被用来确定底层数据结构（例如文件系统路径的 `data.frame`）的起始位置。然而，`dataset()` 并不 *必须* 实际使用该参数。例如，对于极其巨大的 `dataset()` 或严重类别不平衡的情况，它可以选择基于 *抽样* 返回项目。

1.  `.length()`。这通常是一行代码，它的唯一目的是告知 `dataset()` 中可用的项目数量。

这里是创建 `dataset()` 的蓝图：

```r
ds <- dataset()(
 initialize = function(...) {
 ...
 },
 .getitem = function(index) {
 ...
 },
 .length = function() {
 ...
 }
)
```

*话虽如此，让我们比较三种获取 `dataset()` 以便使用的方法，从量身定制到最大程度上的省力。

### 13.2.1 自定义的 `dataset()`

假设我们想要基于流行的 `iris` 替代品 `palmerpenguins` 构建一个分类器。

```r
library(torch)
library(palmerpenguins)
library(dplyr)

penguins %>% glimpse()
```

```r
$ species           <fct> Adelie, Adelie, Adelie, Adelie,...
$ island            <fct> Torgersen, Torgersen, Torgersen,...
$ bill_length_mm    <dbl> 39.1, 39.5, 40.3, NA, 36.7, 39.3,...
$ bill_depth_mm     <dbl> 18.7, 17.4, 18.0, NA, 19.3, 20.6,...
$ flipper_length_mm <int> 181, 186, 195, NA, 193, 190, 181,...
$ body_mass_g       <int> 3750, 3800, 3250, NA, 3450, 3650,...
$ sex               <fct> male, female, female, NA, female,...
$ year              <int> 2007, 2007, 2007, 2007, 2007,...
```

在预测 `species` 时，我们只想使用子集的列：`bill_length_mm`、`bill_depth_mm`、`flipper_length_mm` 和 `body_mass_g`。我们构建一个返回所需内容的 `dataset()`：

```r
penguins_dataset <- dataset(
 name = "penguins_dataset()",
 initialize = function(df) {
 df <- na.omit(df)
 self$x <- as.matrix(df[, 3:6]) %>% torch_tensor()
 self$y <- torch_tensor(
 as.numeric(df$species)
 )$to(torch_long())
 },
 .getitem = function(i) {
 list(x = self$x[i, ], y = self$y[i])
 },
 .length = function() {
 dim(self$x)[1]
 }
)
```

*一旦我们实例化了 `penguins_dataset()`，我们应该立即进行一些检查。首先，它是否有预期的长度？

```r
ds <- penguins_dataset(penguins)
length(ds)
```

```r
[1] 333
```

其次，单个元素是否具有预期的形状和数据类型？方便的是，我们可以像访问张量值一样访问 `dataset()` 项目，通过索引：

```r
ds[1]
```

```r
$x
torch_tensor
   39.1000
   18.7000
  181.0000
 3750.0000
[ CPUFloatType{4} ]

$y
torch_tensor
1
[ CPULongType{} ]
```

这也适用于 `dataset()` 中“更深层”的项目——它必须这样做：当索引 `dataset()` 时，后台发生的是对 `.getitem(i)` 的调用，传递所需的 `i` 位置。

真相是，在这种情况下，我们实际上并不需要构建自己的 `dataset()`。由于预处理的工作量很小，有一个替代方案：`tensor_dataset()`。
  
### 13.2.2 `tensor_dataset()`

当你已经有一个张量或可以轻松转换为张量的东西时，你可以使用内置的 `dataset()` 生成器：`tensor_dataset()`。这个函数可以传递任意数量的张量；每个批次的项目随后是一个张量值的列表：

```r
three <- tensor_dataset(
 torch_randn(10), torch_randn(10), torch_randn(10)
)
three[1]
```

```r
[[1]]
torch_tensor
0.522735
[ CPUFloatType{} ]

[[2]]
torch_tensor
-0.976477
[ CPUFloatType{} ]

[[3]]
torch_tensor
-1.14685
[ CPUFloatType{} ]
```

在我们的 `penguins` 场景中，我们最终得到两行代码：

```r
penguins <- na.omit(penguins)
ds <- tensor_dataset(
 torch_tensor(as.matrix(penguins[, 3:6])),
 torch_tensor(
 as.numeric(penguins$species)
 )$to(torch_long())
)

ds[1]
```

*诚然，我们还没有使用数据集的所有列。你需要`dataset()`执行更多预处理，你更有可能想要编写自己的代码。

第三点也是最后一点，这是最简单的方法。**  **### 13.2.3 `torchvision::mnist_dataset()`

当你在`torch`生态系统中的包工作时，它们很可能已经包含了一些`dataset()`，无论是为了演示目的还是为了数据本身。例如，`torchvision`打包了多个经典图像数据集——其中，那个原型中的原型，MNIST。

由于我们将在后面的章节中讨论图像处理，因此在这里我不会对`mnist_dataset()`的参数进行评论；然而，我们确实包含了一个快速检查，以确保提供的数据符合我们的预期：

```r
library(torchvision)

dir <- "~/.torch-datasets"

ds <- mnist_dataset(
 root = dir,
 train = TRUE, # default
 download = TRUE,
 transform = function(x) {
 x %>% transform_to_tensor() 
 }
)

first <- ds[1]
cat("Image shape: ", first$x$shape, " Label: ", first$y, "\n")
```

```r
Image shape:  1 28 28  Label:  6 
```

到目前为止，这就是我们需要了解的关于`dataset()`的所有内容——在这本书的过程中，我们将遇到很多。现在，我们从单一的数据集转向多个数据集。
  
## 13.3 使用 `dataloader()`s

继续使用新创建的 MNIST `dataset()`，我们为它实例化一个`dataloader()`。`dataloader()`将按批次提供图像和标签对：每次 32 个。在每个 epoch 中，它将以不同的顺序返回它们（`shuffle = TRUE`）：

```r
dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)
```

*就像`dataset()`一样，`dataloader()`也可以查询它们的长度：

```r
length(dl)
```

```r
[1] 1875
```

这次，返回的值不是项目数量；它是批次的数量。

要遍历批次，我们首先获取一个迭代器，这是一个知道如何遍历`dataloader()`中元素的对象。调用`dataloader_next()`后，我们可以逐个访问连续的批次：

```r
first_batch <- dl %>%
 # obtain an iterator for this dataloader
 dataloader_make_iter() %>% 
 dataloader_next()

dim(first_batch$x)
dim(first_batch$y)
```

```r
[1] 32  1 28 28
[1] 32
```

如果你比较`x`（图像部分）的批次形状与单个图像的形状（如上所述），你会发现现在前面有一个额外的维度，反映了批次中的图像数量。

下一步是将批次传递给模型。实际上，这以及完整的端到端深度学习工作流程都是下一章的主题。

