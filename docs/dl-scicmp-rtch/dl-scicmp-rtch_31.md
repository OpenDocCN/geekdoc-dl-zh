# 24  矩阵计算：最小二乘问题

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/matrix_computations_leastsquares.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/matrix_computations_leastsquares.html)

在本章和下一章中，我们将探讨 `torch` 允许我们用矩阵做什么。在这里，我们将查看解决最小二乘问题的各种方法。目的是双重的。

首先，这个主题通常很快就会变得相当技术性，或者说，*计算* 非常快。根据你的背景（和目标），这可能是你想要的；你要么非常了解，要么不太关心底层概念。但对于一些人来说，纯技术性的展示，那种不也停留在 *概念* 上，即主题背后的抽象思想，可能无法传达其魅力，无法传达它可能产生的智力吸引力。这就是为什么在本章中，我将尝试以一种方式来呈现事物，这样主要思想就不会被“计算机科学”细节所掩盖（这些细节在任何一本优秀的书中都很容易找到）。

## 24.1 五种求解最小二乘法的方式

你如何计算线性最小二乘回归？在 R 中，使用 `lm()` 函数；在 `torch` 中，有 `linalg_lstsq()` 函数。R 有时会对用户隐藏复杂性，而高性能计算框架如 `torch` 则倾向于要求用户 upfront 做更多的努力，无论是仔细阅读文档，还是尝试一些，或者两者都要。例如，以下是 `linalg_lstsq()` 函数的核心文档，详细说明了函数的 `driver` 参数：

> `driver` 选择将使用的 LAPACK/MAGMA 函数。
> 
> 对于 CPU 输入，有效的值是 ‘gels’，‘gelsy’，‘gelsd’，‘gelss’。
> 
> 对于 CUDA 输入，唯一有效的驱动程序是 ‘gels’，它假设 A 是满秩的。
> 
> 要在 CPU 上选择最佳驱动程序，请考虑：
> 
> +   如果 A 是良态的（其条件数不是太大），或者你不在乎一些精度损失：
> +   
>     +   对于一般矩阵：‘gelsy’（带置换的 QR）（默认）
>     +   
>     +   如果 A 是满秩的：‘gels’（QR）
>     +   
> +   如果 A 不是良态的：
> +   
>     +   ‘gelsd’（三对角化并 SVD）
>     +   
>     +   但如果你遇到内存问题：‘gelss’（满 SVD）。

你是否需要了解这一点将取决于你正在解决的问题。但如果你需要，了解那里在谈论的内容无疑会很有帮助，即使只是从高层次上了解。

在我们下面的示例问题中，我们会很幸运。所有驱动程序都将返回相同的结果——但只有在我们应用了一种“技巧”之后。尽管如此，我们仍将继续深入研究 `linalg_lstsq()` 以及其他一些常用方法所使用的各种方法。具体来说，我们将解决最小二乘问题：

1.  通过所谓的 *正则方程*，这是最直接的方法，从问题的数学陈述中立即得出。

1.  再次，从正则方程开始，但利用 *Cholesky 分解* 来解决它们。

1.  再次，以正则方程为出发点，但通过*LU*分解进行。

1.  第四，采用另一种类型的分解——*QR*分解——它与最终的分解一起，解释了“现实世界”中应用的大多数分解。使用 QR 分解，求解算法不是从正则方程开始的。

1.  第五点也是最后一点，利用*奇异值分解*（SVD）。在这里，正则方程也不需要。

所有方法首先将应用于真实世界的数据集，然后，在以其缺乏稳定性而闻名的基准问题上进行测试。

## 24.2 天气预测的回归

我们将使用的数据集可以从[UCI 机器学习仓库](http://archive.ics.uci.edu/ml/machine-learning-dAtAbases/00514/Bias_correction_ucl.csv)获取。我们使用它的方式并不完全符合收集的原始目的；我们不是用机器学习预测温度，而是原始研究（Cho 等人（2020））真正关于从数值天气预报模型获得的预测的偏差校正。但没关系——我们在这里的关注点是矩阵方法，这个数据集非常适合我们将要进行的探索。

```r
set.seed(777)

library(torch)
torch_manual_seed(777)

library(dplyr)
library(readr)

library(zeallot)

uci <- "https://archive.ics.uci.edu"
ds_path <- "ml/machine-learning-databases/00514"
ds_file <- "Bias_correction_ucl.csv"

# download.file(
#   file.path(uci, ds_path, ds_file),
#   destfile = "resources/matrix-weather.csv"
# )

weather_df <- read_csv("resources/matrix-weather.csv") %>%
 na.omit()
weather_df %>% glimpse()
```

```r
Rows: 7,588
Columns: 25
$ station           <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,…
$ Date              <date> 2013-06-30, 2013-06-30,…
$ Present_Tmax      <dbl> 28.7, 31.9, 31.6, 32.0, 31.4, 31.9,…
$ Present_Tmin      <dbl> 21.4, 21.6, 23.3, 23.4, 21.9, 23.5,…
$ LDAPS_RHmin       <dbl> 58.25569, 52.26340, 48.69048,…
$ LDAPS_RHmax       <dbl> 91.11636, 90.60472, 83.97359,…
$ LDAPS_Tmax_lapse  <dbl> 28.07410, 29.85069, 30.09129,…
$ LDAPS_Tmin_lapse  <dbl> 23.00694, 24.03501, 24.56563,…
$ LDAPS_WS          <dbl> 6.818887, 5.691890, 6.138224,…
$ LDAPS_LH          <dbl> 69.45181, 51.93745, 20.57305,…
$ LDAPS_CC1         <dbl> 0.2339475, 0.2255082, 0.2093437,…
$ LDAPS_CC2         <dbl> 0.2038957, 0.2517714, 0.2574694,…
$ LDAPS_CC3         <dbl> 0.1616969, 0.1594441, 0.2040915,…
$ LDAPS_CC4         <dbl> 0.1309282, 0.1277273, 0.1421253,…
$ LDAPS_PPT1        <dbl> 0.0000000, 0.0000000, 0.0000000,…
$ LDAPS_PPT2        <dbl> 0.000000, 0.000000, 0.000000,…
$ LDAPS_PPT3        <dbl> 0.0000000, 0.0000000, 0.0000000,…
$ LDAPS_PPT4        <dbl> 0.0000000, 0.0000000, 0.0000000,…
$ lat               <dbl> 37.6046, 37.6046, 37.5776, 37.6450,…
$ lon               <dbl> 126.991, 127.032, 127.058, 127.022,…
$ DEM               <dbl> 212.3350, 44.7624, 33.3068, 45.7160,…
$ Slope             <dbl> 2.7850, 0.5141, 0.2661, 2.5348,…
$ `Solar radiation` <dbl> 5992.896, 5869.312, 5863.556,…
$ Next_Tmax         <dbl> 29.1, 30.5, 31.1, 31.7, 31.2, 31.5,…
$ Next_Tmin         <dbl> 21.2, 22.5, 23.9, 24.3, 22.5, 24.0,…
```

我们构建任务的方式，基本上数据集中的每一项都作为（或者如果保留它——下面会更多讨论）预测因子。作为目标，我们将使用`Next_Tmax`，即后续一天达到的最高温度。这意味着我们需要从预测因子集中移除`Next_Tmin`，因为它会成为一个过于强大的线索。我们也会对`station`（天气站 ID）和`Date`（日期）做同样的事情。这使我们剩下二十一個预测因子，包括实际温度的测量（`Present_Tmax`，`Present_Tmin`）、各种变量的模型预测（`LDAPS_*`）以及辅助信息（`lat`，`lon`和`Solar radiation`等）。

```r
weather_df <- weather_df %>%
 select(-c(station, Next_Tmin, Date)) %>%
 mutate(across(.fns = scale))
```

*注意，在上面，我添加了一行来*标准化*预测因子。这就是我上面提到的“技巧”。我们很快就会讨论为什么我们要这样做。

对于`torch`，我们将数据分成两个张量：一个包含所有预测因子的矩阵`A`，以及一个包含目标值的向量`b`。

```r
weather <- torch_tensor(weather_df %>% as.matrix())
A <- weather[ , 1:-2]
b <- weather[ , -1]

dim(A)
```

```r
[1] 7588   21
```

现在，首先让我们确定预期的输出。

### 24.2.1 最小二乘法（I）：使用`lm()`设置期望

如果有一个我们“相信”的最小二乘实现，那肯定就是`lm()`。

```r
fit <- lm(Next_Tmax ~ . , data = weather_df)
fit %>% summary()
```

```r
Call:
lm(formula = Next_Tmax ~ ., data = weather_df)

Residuals:
     Min       1Q   Median       3Q      Max 
-1.94439 -0.27097  0.01407  0.28931  2.04015 

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)        2.605e-15  5.390e-03   0.000 1.000000    
Present_Tmax       1.456e-01  9.049e-03  16.089  < 2e-16 

Present_Tmin       4.029e-03  9.587e-03   0.420 0.674312    
LDAPS_RHmin        1.166e-01  1.364e-02   8.547  < 2e-16 

LDAPS_RHmax       -8.872e-03  8.045e-03  -1.103 0.270154    
LDAPS_Tmax_lapse   5.908e-01  1.480e-02  39.905  < 2e-16 

LDAPS_Tmin_lapse   8.376e-02  1.463e-02   5.726 1.07e-08 

LDAPS_WS          -1.018e-01  6.046e-03 -16.836  < 2e-16 

LDAPS_LH           8.010e-02  6.651e-03  12.043  < 2e-16 

LDAPS_CC1         -9.478e-02  1.009e-02  -9.397  < 2e-16 

LDAPS_CC2         -5.988e-02  1.230e-02  -4.868 1.15e-06 

LDAPS_CC3         -6.079e-02  1.237e-02  -4.913 9.15e-07 

LDAPS_CC4         -9.948e-02  9.329e-03 -10.663  < 2e-16 

LDAPS_PPT1        -3.970e-03  6.412e-03  -0.619 0.535766    
LDAPS_PPT2         7.534e-02  6.513e-03  11.568  < 2e-16 

LDAPS_PPT3        -1.131e-02  6.058e-03  -1.866 0.062056 .  
LDAPS_PPT4        -1.361e-03  6.073e-03  -0.224 0.822706    
lat               -2.181e-02  5.875e-03  -3.713 0.000207 

lon               -4.688e-02  5.825e-03  -8.048 9.74e-16 

DEM               -9.480e-02  9.153e-03 -10.357  < 2e-16 

Slope              9.402e-02  9.100e-03  10.331  < 2e-16 

`Solar radiation`  1.145e-02  5.986e-03   1.913 0.055746 .  
---
Signif. codes:  0 ‘
’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.4695 on 7566 degrees of freedom
Multiple R-squared:  0.7802,    Adjusted R-squared:  0.7796 
F-statistic:  1279 on 21 and 7566 DF,  p-value: < 2.2e-16
```

解释方差为 78%，预测效果相当不错。这是我们想要检查所有其他方法的基线。为此，我们将存储相应的预测和预测误差（后者被操作化为均方根误差，RMSE）。目前，我们只有`lm()`的条目：

```r
rmse <- function(y_true, y_pred) {
 (y_true - y_pred)² %>%
 sum() %>%
 sqrt()
}

all_preds <- data.frame(
 b = weather_df$Next_Tmax,
 lm = fit$fitted.values
)
all_errs <- data.frame(lm = rmse(all_preds$b, all_preds$lm))
all_errs
```

```r
 lm
1 40.8369
```**  **### 24.2.2 最小二乘法（II）：使用`linalg_lstsq()`

现在，让我们暂时假设这并不是在探索不同的方法，而是要快速得到结果。在`torch`中，我们有`linalg_lstsq()`，这是一个专门用于解决最小二乘问题的函数。（这就是我在上面引用的文档中提到的函数。）就像我们使用`lm()`一样，我们可能会直接调用它，并使用默认设置：

```r
x_lstsq <- linalg_lstsq(A, b)$solution

all_preds$lstsq <- as.matrix(A$matmul(x_lstsq))
all_errs$lstsq <- rmse(all_preds$b, all_preds$lstsq)

tail(all_preds)
```

```r
 b         lm      lstsq
7583 -1.1380931 -1.3544620 -1.3544616
7584 -0.8488721 -0.9040997 -0.9040993
7585 -0.7203294 -0.9675286 -0.9675281
7586 -0.6239224 -0.9044044 -0.9044040
7587 -0.5275154 -0.8738639 -0.8738635
7588 -0.7846007 -0.8725795 -0.8725792
```

预测结果与`lm()`非常相似——事实上，如此相似，以至于我们可能会猜测那些微小的差异只是由于从各自的调用堆栈中出现的数值误差。因此，RMSE 也应该相等：

```r
all_errs
```

```r
 lm    lstsq
1 40.8369 40.8369
```

是的；这是一个令人满意的结果。然而，这主要是因为那个“技巧”：归一化。当然，当我说“技巧”时，我并不是真的这么认为。标准化数据是一个常见的操作，尤其是在神经网络中，它通常会被例行使用，以加快训练速度。我想说的是：高性能计算框架，如`torch`，通常会假设用户有更多的领域知识，或者更多的前期分析。

我会解释。**  **### 24.2.3 间奏：如果我们没有标准化数据会怎样？

为了快速比较，让我们创建一个替代的预测矩阵：这次**不**对数据进行归一化。

```r
weather_df_alt <- 
 read_csv("resources/matrix-weather.csv") %>% 
 na.omit() %>%
 select(-c(station, Next_Tmin, Date)) 

weather_alt <- torch_tensor(weather_df_alt %>% as.matrix())
A_alt <- weather_alt[ , 1:-2]
b_alt <- weather_alt[ , -1]
```

*为了设定我们的预期，我们再次调用`lm()`：

```r
fit_alt <- lm(Next_Tmax ~ ., data = weather_df_alt)
all_preds_alt <- data.frame(
 b = weather_df_alt$Next_Tmax,
 lm = fit_alt$fitted.values
)

all_errs_alt <- data.frame(
 lm = rmse(
 all_preds_alt$b,
 all_preds_alt$lm
 )
)

all_errs_alt
```

```r
 lm
1 127.0765
```

现在，我们调用`linalg_lstsq()`，使用与之前相同的默认参数。

```r
x_lstsq_alt <- linalg_lstsq(A_alt, b_alt)$solution

all_preds_alt$lstsq <- as.matrix(A_alt$matmul(x_lstsq_alt))
all_errs_alt$lstsq <- rmse(
 all_preds_alt$b, all_preds_alt$lstsq
)

all_errs_alt
```

```r
 lm    lstsq
1 127.0765 177.9128
```

哇——这里发生了什么？回想一下我引用的那段文档，也许默认参数这次并不奏效。让我们找出原因。

#### 24.2.3.1 调查“问题”

为了有效地解决线性最小二乘问题，`torch`调用 LAPACK，这是一组 Fortran 例程，旨在高效且可扩展地处理线性代数中最常见的任务：求解线性方程组、计算特征向量和特征值，以及确定奇异值。

`linalg_lstsq()`中允许的`driver`对应于不同的 LAPACK 过程¹，这些过程都应用不同的算法来解决该问题——类似于我们下面将要做的。

因此，在调查发生了什么时，第一步是确定使用了哪种方法以及为什么；分析（如果可能的话）为什么结果不尽如人意；确定我们想要使用的 LAPACK 例程，并检查如果我们真的这样做会发生什么。（当然，考虑到所涉及的少量努力，我们可能会尝试所有方法。）

这里涉及的主要概念是矩阵的**秩**。

#### 24.2.3.2 概念（I）：矩阵的秩

*“等等！”* 你可能正在想——从上述引用的文档片段来看，我们首先应该检查的不是秩，而是 *条件数*：矩阵是否“良好条件”。是的，条件数确实很重要，我们很快就会回到这一点。然而，这里还有更基本的东西在起作用，这并不真的“跃入眼帘”。

关键信息位于我们通过 `linalg_lstsq()` 被引用的 LAPACK 文档片段中。在 GELS、GELSY、GELSD 和 GELSS 这四个例程之间，差异不仅限于实现。优化的 *目标* 也不同。其理由如下。在整个过程中，让我们假设我们正在处理一个行数多于列数的矩阵（在最常见的案例中，观察值多于特征值）：

+   如果矩阵是满秩的——这意味着其列是线性无关的——则不存在“完美”的解。问题是超定的。我们能做的只是找到最佳可能的近似。这是通过最小化预测误差来实现的——我们将在讨论正规方程时回到这一点。最小化预测误差是 GELS 例程所做的事情，当我们有一个满秩的预测矩阵时，我们应该使用 GELS。

+   如果矩阵不是满秩的，问题是不确定的；存在无限多个解。所有剩余的例程——GELSY、GELSD 和 GELSS——都适用于这种情况。虽然它们采取不同的步骤，但它们都追求与 GELS 不同的策略：除了预测误差之外，它们还 *也* 最小化系数向量。这被称为寻找最小范数的最小二乘解。

总结来说，GELS（用于满秩矩阵）和 GELSY、GELSD、GELSS 的三个例程（用于矩阵秩不足的情况）有意遵循不同的优化标准。

现在，根据 `linalg_lstsq()` 的文档，当没有明确传递 `driver` 时，会调用 GELSY。如果我们的矩阵是秩不足的，这应该没问题——但是，它是吗？

```r
linalg_matrix_rank(A_alt)
```

```r
torch_tensor
21
[ CPULongType{} ]
```

矩阵有二十一个列；所以如果其秩是二十一个，那么它肯定是一个满秩矩阵。我们肯定希望调用 GELS 例程。*  *#### 24.2.3.3 正确调用 `linalg_lstsq()`

现在我们知道了应该为 `driver` 传递什么，这里是修改后的调用：

```r
x_lstsq_alt <- linalg_lstsq(
 A_alt, b_alt,
 driver = "gels"
)$solution

all_preds_alt$lstsq <- as.matrix(A_alt$matmul(x_lstsq_alt))
all_errs_alt$lstsq <- rmse(
 all_preds_alt$b,
 all_preds_alt$lstsq
)

all_errs_alt
```

```r
 lm    lstsq
1 127.0765 127.9489
```

现在，相应的 RMSE 值非常接近。不过，你可能想知道：为什么在处理 *标准化* 矩阵时，我们没有必要指定 Fortran 例程？*  *#### 24.2.3.4 为什么标准化有帮助？

对于我们的矩阵，标准化所做的就是显著减少了奇异值所跨越的范围。对于 `A`，标准化矩阵，最大的奇异值大约是最小的十倍：

```r
svals_normalized_A <- linalg_svdvals(A)/linalg_svdvals(A)[1]
svals_normalized_A %>% as.numeric()
```

```r
[1] 1.0000000 0.7473214 0.5929527 0.5233989 0.5188764 0.4706140
[7] 0.4391665 0.4249273 0.4034659 0.3815900 0.3621315 0.3557949
[13] 0.3297923 0.2707912 0.2489560 0.2229859 0.2175170 0.1852890
[19] 0.1627083 0.1553169 0.1075778
```

而与 `A_alt` 相比，它要大上百万倍：

```r
svals_normalized_A_alt <- linalg_svdvals(A_alt) /
 linalg_svdvals(A_alt)[1]
svals_normalized_A_alt %>% as.numeric()
```

```r
[1] 1.000000e+00 1.014369e-02 6.407313e-03 2.881966e-03
[5] 2.236537e-03 9.633782e-04 6.678377e-04 3.988165e-04
[9] 3.584047e-04 3.137257e-04 2.699152e-04 2.383501e-04
[13] 2.234150e-04 1.803384e-04 1.625245e-04 1.300101e-04
[17] 4.312536e-05 3.463851e-05 1.964120e-05 1.689913e-05
[18] 8.419599e-06
```

这为什么重要？正是在这里，我们最终回到了 *条件数*。**  **#### 24.2.3.5 概念（II）：条件数

一个矩阵的所谓 *条件数* 越高，我们在使用它进行计算时遇到数值稳定性问题的可能性就越大。在 torch 中，使用 `linalg_cond()` 来获取条件数。让我们比较 `A` 和 `A_alt` 的条件数。

```r
linalg_cond(A)
linalg_cond(A_alt)
```

```r
torch_tensor
9.2956
[ CPUFloatType{} ]

torch_tensor
118770
[ CPUFloatType{} ]
```

这是一种相当大的差异！它是如何产生的？

条件数定义为矩阵 `A` 的矩阵范数除以其逆的范数。可以使用不同的范数；默认是 2-范数。在这种情况下，条件数可以从矩阵的奇异值中计算出来：即 `A` 的 2-范数等于最大的奇异值，而其逆的范数由最小的奇异值给出。

我们可以使用 `linalg_svdvals()` 来验证这一点，就像之前一样：

```r
linalg_svdvals(A)[1]/linalg_svdvals(A)[21]
linalg_svdvals(A_alt)[1]/linalg_svdvals(A_alt)[21]
```

```r
torch_tensor
9.29559
[ CPUFloatType{} ]

torch_tensor
118770
[ CPUFloatType{} ]
```

再次强调，这是一个实质性的差异。顺便问一下，你还记得在 `A_alt` 的情况下，即使使用适当的程序 GELS，`linalg_lstsq()` 的 RMSE 比使用 `lm()` 稍微差一点吗？鉴于两者基本上使用相同的算法（QR 分解，很快就会介绍），这很可能是因为 `A_alt` 的高条件数导致的数值误差。

到目前为止，我可能已经说服你，使用 `torch` 的 `linalg` 组件，了解最常用的最小二乘算法的工作原理是有帮助的。让我们熟悉一下。
  
### 24.2.4 最小二乘法（III）：正则方程

我们首先明确目标。给定一个矩阵 $\mathbf{A}$，其列包含特征，行包含观测值，以及一个观测结果的向量 $\mathbf{b}$，我们希望找到回归系数，每个特征一个，以便尽可能好地近似 $\mathbf{b}$。将回归系数的向量称为 $\mathbf{x}$。为了获得它，我们需要解一个联立方程组，在矩阵表示法中，它看起来像

$$ \mathbf{Ax} = \mathbf{b} $$

如果 $\mathbf{b}$ 是一个可逆的方阵，那么解可以直接计算为 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$。这几乎是不可能的；我们（希望）总是有比预测因子更多的观测值。另一种方法是直接从问题陈述开始。

当我们使用 $\mathbf{A}$ 的列来近似 $\mathbf{b}$ 时，这种近似必然是在 $\mathbf{A}$ 的列空间中。另一方面，$\mathbf{b}$ 通常不会。我们希望这两个尽可能接近；换句话说，我们希望最小化它们之间的距离。选择 2-范数作为距离，这产生了目标

$$ minimize \ ||\mathbf{Ax}-\mathbf{b}||² $$

这个距离是预测误差向量的（平方）长度。这个向量必然与$\mathbf{A}$本身正交。也就是说，当我们用$\mathbf{A}$乘以它时，我们得到零向量：

$$ \mathbf{A}^T(\mathbf{Ax} - \mathbf{b}) = \mathbf{0} $$

对这个方程的重新排列得到所谓的*正则方程*：

$$ \mathbf{A}^T \mathbf{A} \mathbf{x} = \mathbf{A}^T \mathbf{b} $$

这些可以解出$\mathbf{x}$，计算$\mathbf{A}^T\mathbf{A}$的逆：

$$ \mathbf{x} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b} $$

$\mathbf{A}^T\mathbf{A}$是一个方阵。它仍然可能不可逆，在这种情况下，将计算所谓的伪逆。在我们的情况下，这将是必要的；我们已知$\mathbf{A}$具有满秩，同样$\mathbf{A}^T\mathbf{A}$也是。

因此，我们从正则方程中推导出计算$\mathbf{b}$的配方。让我们将其付诸实践，并与我们从`lm()`和`linalg_lstsq()`中得到的结果进行比较。

```r
AtA <- A$t()$matmul(A)
Atb <- A$t()$matmul(b)
inv <- linalg_inv(AtA)
x <- inv$matmul(Atb)

all_preds$neq <- as.matrix(A$matmul(x))
all_errs$neq <- rmse(all_preds$b, all_preds$neq)

all_errs
```

```r
 lm   lstsq     neq
1 40.8369 40.8369 40.8369
```

确认直接方法可行后，我们可以允许自己一些复杂性。四种不同的矩阵分解将出现：Cholesky、LU、QR 和奇异值分解。在每种情况下，目标都是避免计算（伪）逆。这是所有方法共有的。然而，它们不仅在矩阵分解的方式上有所不同，而且在分解的矩阵上也有所不同。这与各种方法施加的约束有关。粗略地说，它们列出的顺序反映了预条件的下降斜率，或者换句话说，是一般性的上升斜率。由于涉及的约束，前两种（Cholesky 以及 LU 分解）将在$\mathbf{A}^T\mathbf{A}$上执行，而后两种（QR 和 SVD）直接在$\mathbf{A}$上操作。使用它们，永远不需要计算$\mathbf{A}^T\mathbf{A}$。*  *### 24.2.5 最小二乘法（IV）：Cholesky 分解

在 Cholesky 分解中，一个矩阵被分解成两个相同大小的三角矩阵，其中一个矩阵是另一个矩阵的转置。这通常可以写成

$$ \mathbf{A} = \mathbf{L} \mathbf{L}^T $$ 或者

$$ \mathbf{A} = \mathbf{R}^T\mathbf{R} $$

这里符号$\mathbf{L}$和$\mathbf{R}$分别表示下三角矩阵和上三角矩阵。

为了进行 Cholesky 分解，一个矩阵必须既是对称的又是正定的。这些条件相当严格，在实践中通常不会经常满足。在我们的情况下，$\mathbf{A}$不是对称的；这立即意味着我们必须在$\mathbf{A}^T\mathbf{A}$上操作。由于$\mathbf{A}$已经是正定的，我们知道$\mathbf{A}^T\mathbf{A}$也是。

在`torch`中，我们使用`linalg_cholesky()`函数来获取矩阵的 Cholesky 分解。默认情况下，这个调用将返回$\mathbf{L}$，一个下三角矩阵。

```r
# AtA = L L_t
AtA <- A$t()$matmul(A)
L <- linalg_cholesky(AtA)
```

*让我们检查一下我们是否可以从 $\mathbf{L}$ 中重建 $\mathbf{A}$：

```r
LLt <- L$matmul(L$t())
diff <- LLt - AtA
linalg_norm(diff, ord = "fro")
```

```r
torch_tensor
0.00258896
[ CPUFloatType{} ]
```

这里，我计算了原始矩阵与其重建之间的 Frobenius 范数的差。Frobenius 范数分别求和所有矩阵项，并返回平方根。理论上，我们希望在这里看到零；但在数值误差的存在下，结果足以表明分解工作得很好。

现在我们有了 $\mathbf{L}\mathbf{L}^T$ 而不是 $\mathbf{A}^T\mathbf{A}$，这如何帮助我们呢？正是在这里发生了魔法，你还会在剩下的三种方法中找到同样的魔法。其思路是，由于某种分解，出现了一种更高效的方式来解决构成给定任务的一组方程。

在 $\mathbf{L}\mathbf{L}^T$ 的情况下，关键是 $\mathbf{L}$ 是三角形的，当这种情况发生时，线性系统可以通过简单的替换来解决。这最好用一个小的例子来说明：

$$ \begin{bmatrix} 1 & 0 & 0\\ 2 & 3 & 0\\ 3 & 4 & 1 \end{bmatrix} \begin{bmatrix} x1\\ x2\\ x3 \end{bmatrix} = \begin{bmatrix} 1\\ 11\\ 15 \end{bmatrix} $$

从最上面一行开始，我们立即看到 $x1$ 等于 $1$；一旦我们知道 *这一点*，从第二行就可以直接计算出 $x2$ 必须是 $3$。最后一行告诉我们 $x3$ 必须是 $0$。

在代码中，`torch_triangular_solve()` 用于高效地计算一个线性方程组的解，其中预测矩阵是下三角或上三角。一个额外的要求是矩阵必须是对称的——但这个条件我们已经在能够使用 Cholesky 分解的情况下满足了。

默认情况下，`torch_triangular_solve()` 预期矩阵是上三角（而不是下三角）；但有一个函数参数，`upper`，允许我们纠正这个预期。返回值是一个列表，其第一个元素包含所需的解。为了说明，这里展示了 `torch_triangular_solve()`，它应用于我们手动解决的玩具示例：

```r
some_L <- torch_tensor(
 matrix(c(1, 0, 0, 2, 3, 0, 3, 4, 1), nrow = 3, byrow = TRUE)
)
some_b <- torch_tensor(matrix(c(1, 11, 15), ncol = 1))

x <- torch_triangular_solve(
 some_b,
 some_L,
 upper = FALSE
)[[1]]
x
```

```r
torch_tensor
 1
 3
 0
[ CPUFloatType{3,1} ]
```

回到我们的运行示例，正则方程现在看起来是这样的：

$$ \mathbf{L}\mathbf{L}^T \mathbf{x} = \mathbf{A}^T \mathbf{b} $$

我们引入一个新的变量 $\mathbf{y}$，用来表示 $\mathbf{L}^T \mathbf{x}$。

$$ \mathbf{L}\mathbf{y} = \mathbf{A}^T \mathbf{b} $$

并计算这个系统的解：

```r
Atb <- A$t()$matmul(b)

y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
)[[1]]
```

*现在我们有了 $y$，我们回顾一下它是如何定义的：

$$ \mathbf{y} = \mathbf{L}^T \mathbf{x} $$

为了确定 $\mathbf{x}$，我们再次可以使用 `torch_triangular_solve()`：

```r
x <- torch_triangular_solve(y, L$t())[[1]]
```

*就这样了。

如同往常，我们计算预测误差：

```r
all_preds$chol <- as.matrix(A$matmul(x))
all_errs$chol <- rmse(all_preds$b, all_preds$chol)

all_errs
```

```r
 lm   lstsq     neq    chol
1 40.8369 40.8369 40.8369 40.8369
```

现在你已经看到了 Cholesky 分解背后的原理——正如已经提到的，这个想法也适用于所有其他分解——你可能想利用一个专门的便利函数，`torch_cholesky_solve()`，来节省一些工作。这将使对 `torch_triangular_solve()` 的两次调用变得过时。

以下代码行产生的输出与上面的代码相同——但当然，它们确实隐藏了背后的魔法。

```r
L <- linalg_cholesky(AtA)

x <- torch_cholesky_solve(Atb$unsqueeze(2), L)

all_preds$chol2 <- as.matrix(A$matmul(x))
all_errs$chol2 <- rmse(all_preds$b, all_preds$chol2)
all_errs
```

```r
 lm   lstsq     neq    chol   chol2
1 40.8369 40.8369 40.8369 40.8369 40.8369
```

让我们继续下一个方法——相当于下一个分解方法。
  
### 24.2.6 最小二乘法（V）：LU 分解

LU 分解以它引入的两个因子命名：一个下三角矩阵$\mathbf{L}$，以及一个上三角矩阵$\mathbf{U}$。在理论上，LU 分解没有限制：如果我们允许行交换，实际上将$\mathbf{A} = \mathbf{L}\mathbf{U}$转换为$\mathbf{A} = \mathbf{P}\mathbf{L}\mathbf{U}$（其中$\mathbf{P}$是一个置换矩阵），我们就可以分解任何矩阵。

在实践中，尽管如此，如果我们想使用`torch_triangular_solve()`，输入矩阵必须是对称的。因此，在这里我们也必须处理$\mathbf{A}^T\mathbf{A}$，而不是直接处理$\mathbf{A}$。（这就是为什么我在展示 Cholesky 分解之后紧接着展示 LU 分解的原因——它们在让我们做什么方面相似，但在精神上却完全不同。）

使用$\mathbf{A}^T\mathbf{A}$意味着我们再次从正则方程开始。我们分解$\mathbf{A}^T\mathbf{A}$，然后解两个三角系统以得到最终解。以下是步骤，包括有时不需要的置换矩阵$\mathbf{P}$：

$$ \begin{aligned} \mathbf{A}^T \mathbf{A} \mathbf{x} &= \mathbf{A}^T \mathbf{b} \\ \mathbf{P} \mathbf{L}\mathbf{U} \mathbf{x} &= \mathbf{A}^T \mathbf{b} \\ \mathbf{L} \mathbf{y} &= \mathbf{P}^T \mathbf{A}^T \mathbf{b} \\ \mathbf{y} &= \mathbf{U} \mathbf{x} \end{aligned} $$

我们看到，当$\mathbf{P}$ *需要*时，会有额外的计算：遵循与 Cholesky 分解相同的策略，我们希望将$\mathbf{P}$从左边移到右边。幸运的是，可能看起来很昂贵——计算逆——实际上并不昂贵：对于置换矩阵，其转置会反转操作。

在代码方面，我们已经熟悉了我们需要做的绝大多数内容。唯一缺少的部分是`torch_lu()`。`torch_lu()`返回一个包含两个张量的列表，第一个是三个矩阵$\mathbf{P}$、$\mathbf{L}$和$\mathbf{U}$的压缩表示。我们可以使用`torch_lu_unpack()`来解压缩它：

```r
lu <- torch_lu(AtA)

c(P, L, U) %<-% torch_lu_unpack(lu[[1]], lu[[2]]) 
```

*我们将$\mathbf{P}$移到另一边：

```r
Atb <- P$t()$matmul(Atb)
```

*剩下要做的就是解两个三角系统，我们就完成了：

```r
y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
)[[1]]
x <- torch_triangular_solve(y, U)[[1]]

all_preds$lu <- as.matrix(A$matmul(x))
all_errs$lu <- rmse(all_preds$b, all_preds$lu)
all_errs[1, -5]
```

```r
 lm   lstsq     neq    chol      lu
1 40.8369 40.8369 40.8369 40.8369 40.8369
```

与 Cholesky 分解一样，我们可以避免调用两次`torch_triangular_solve()`。`torch_lu_solve()`接受分解，并直接返回最终解：

```r
lu <- torch_lu(AtA)
x <- torch_lu_solve(Atb$unsqueeze(2), lu[[1]], lu[[2]])

all_preds$lu2 <- as.matrix(A$matmul(x))
all_errs$lu2 <- rmse(all_preds$b, all_preds$lu2)
all_errs[1, -5]
```

```r
 lm   lstsq     neq    chol      lu      lu
1 40.8369 40.8369 40.8369 40.8369 40.8369 40.8369
```

现在，我们来看两种不需要计算$\mathbf{A}^T\mathbf{A}$的方法。
  
### 24.2.7 最小二乘法（VI）：QR 分解

任何矩阵都可以分解成一个正交矩阵 $\mathbf{Q}$ 和一个上三角矩阵 $\mathbf{R}$。QR 分解可能是解决最小二乘问题最流行的方法；实际上，它是 R 的`lm()`函数使用的方法。那么，它以什么方式简化了任务？

至于 $\mathbf{R}$，我们已经知道它的用途：由于是三角矩阵，它定义了一个可以通过逐步替换求解的方程组。而 $\mathbf{Q}$ 更好。一个正交矩阵的列是正交的——这意味着，它们的点积都是零——并且具有单位范数；这样的矩阵的好处是它的逆等于它的转置。一般来说，逆矩阵难以计算；然而，转置矩阵是容易计算的。鉴于逆矩阵的计算——求解 $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$——是最小二乘法中的核心任务，这立即表明了其重要性。

与我们通常的方案相比，这导致了一个略微简化的步骤。不再有“虚拟”变量 $\mathbf{y}$。相反，我们直接将 $\mathbf{Q}$ 移到另一边，计算转置（这实际上是逆）。然后剩下的就是回代。此外，由于每个矩阵都有一个 QR 分解，我们现在直接从 $\mathbf{A}$ 开始，而不是从 $\mathbf{A}^T\mathbf{A}$ 开始：

$$ \begin{aligned} \mathbf{A}\mathbf{x} &= \mathbf{b}\\ \mathbf{Q}\mathbf{R}\mathbf{x} &= \mathbf{b}\\ \mathbf{R}\mathbf{x} &= \mathbf{Q}^T\mathbf{b}\\ \end{aligned} $$

在`torch`中，`linalg_qr()`函数为我们提供了矩阵 $\mathbf{Q}$ 和 $\mathbf{R}$。

```r
c(Q, R) %<-% linalg_qr(A)
```

*在右侧，我们曾经有一个“便利变量”来保存 $\mathbf{A}^T\mathbf{b}$；在这里，我们跳过这一步，而是做些“立即有用”的事情：将 $\mathbf{Q}$ 移到另一边。

```r
Qtb <- Q$t()$matmul(b)
```

*现在唯一剩下的步骤就是解决剩余的三角系统。

```r
x <- torch_triangular_solve(Qtb$unsqueeze(2), R)[[1]]

all_preds$qr <- as.matrix(A$matmul(x))
all_errs$qr <- rmse(all_preds$b, all_preds$qr)
all_errs[1, -c(5,7)]
```

```r
 lm   lstsq     neq    chol      lu      qr
1 40.8369 40.8369 40.8369 40.8369 40.8369 40.8369 
```

到现在为止，你可能期望我结束这一节时说“在`torch`/`torch_linalg`中也有一个专门的求解器，即……”）。好吧，不是字面上的；但实质上，是的。如果你调用`linalg_lstsq()`并传递`driver = "gels"`，将会使用 QR 分解。
  
### 24.2.8 最小二乘法（VII）：奇异值分解（SVD）

按照真正的气候顺序，我们最后讨论的分解方法是最高效的、最多样化的、最具有语义意义的：*奇异值分解（SVD）*。第三个方面，虽然很迷人，但与我们当前的任务无关，所以在这里我不会深入探讨。在这里，重要的是其普遍适用性：每个矩阵都可以按照 SVD 风格分解成组件。

奇异值分解将输入 $\mathbf{A}$ 分解为两个正交矩阵，称为 $\mathbf{U}$ 和 $\mathbf{V}^T$，以及一个对角矩阵，称为 $\symbf{\Sigma}$，使得 $\mathbf{A} = \mathbf{U} \symbf{\Sigma} \mathbf{V}^T$。在这里，$\mathbf{U}$ 和 $\mathbf{V}^T$ 是*左*和*右*奇异向量，而 $\symbf{\Sigma}$ 包含*奇异值*。

$$ \begin{aligned} \mathbf{A}\mathbf{x} &= \mathbf{b}\\ \mathbf{U}\symbf{\Sigma}\mathbf{V}^T\mathbf{x} &= \mathbf{b}\\ \symbf{\Sigma}\mathbf{V}^T\mathbf{x} &= \mathbf{U}^T\mathbf{b}\\ \mathbf{V}^T\mathbf{x} &= \mathbf{y}\\ \end{aligned} $$

我们首先使用 `linalg_svd()` 获取分解。参数 `full_matrices = FALSE` 告诉 `torch` 我们想要一个与 $\mathbf{A}$ 维度相同的 $\mathbf{U}$，而不是扩展到 7588 x 7588。

```r
c(U, S, Vt) %<-% linalg_svd(A, full_matrices = FALSE)

dim(U)
dim(S)
dim(Vt)
```

```r
[1] 7588   21
[1] 21
[1] 21 21
```

我们将 $\mathbf{U}$ 移到另一边——由于 $\mathbf{U}$ 是正交的，这是一个便宜的操作。

```r
Utb <- U$t()$matmul(b)
```

*由于 $\mathbf{U}^T\mathbf{b}$ 和 $\symbf{\Sigma}$ 都是相同长度的向量，我们可以使用逐元素乘法对 $\symbf{\Sigma}$ 做同样的事情。我们引入一个临时变量 `y` 来保存结果。

```r
y <- Utb / S
```

*现在剩下最后一个系统需要解决，$\mathbf{\mathbf{V}^T\mathbf{x} = \mathbf{y}}$，我们再次得益于正交性——这次是矩阵 $\mathbf{V}^T$ 的正交性。

```r
x <- Vt$t()$matmul(y)
```

*总结一下，让我们计算预测和预测误差：

```r
all_preds$svd <- as.matrix(A$matmul(x))
all_errs$svd <- rmse(all_preds$b, all_preds$svd)

all_errs[1, -c(5, 7)]
```

```r
 lm   lstsq     neq    chol      lu     qr      svd
1 40.8369 40.8369 40.8369 40.8369 40.8369 40.8369 40.8369
```

这就结束了我们对重要最小二乘算法的巡礼。总结这个例子，我们快速看一下性能。
  
### 24.2.9 检查执行时间

正如我所说的，本章的重点是概念，而不是性能。但一旦你开始处理更大的数据集，你不可避免地会关心速度。而且，看看这些方法有多快也是很有趣的！所以，让我们快速进行一次性能基准测试。只是，请不要从这些结果外推——相反，请在你关心的数据上运行类似的代码。

为了计时，我们需要将所有算法封装在其各自的功能中。这里它们是：

```r
# normal equations
ls_normal_eq <- function(A, b) {
 AtA <- A$t()$matmul(A)
 x <- linalg_inv(AtA)$matmul(A$t())$matmul(b)
 x
}

# normal equations and Cholesky decomposition (done manually)
# A_t A x = A_t b
# L L_t x = A_t b
# L y = A_t b 
# L_t x = y
ls_cholesky_diy <- function(A, b) {
 AtA <- A$t()$matmul(A)
 Atb <- A$t()$matmul(b)
 L <- linalg_cholesky(AtA)
 y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
 )[[1]]
 x <- torch_triangular_solve(y, L$t())[[1]]
 x
}

# torch's Cholesky solver
ls_cholesky_solve <- function(A, b) {
 AtA <- A$t()$matmul(A)
 Atb <- A$t()$matmul(b)
 L <- linalg_cholesky(AtA)
 x <- torch_cholesky_solve(Atb$unsqueeze(2), L)
 x
}

# normal equations and LU factorization (done manually)
# A_t A x = A_t b
# P L U x = A_t b
# L y = P_t A_t b          # where y = U x
# U x = y
ls_lu_diy <- function(A, b) {
 AtA <- A$t()$matmul(A)
 Atb <- A$t()$matmul(b)
 lu <- torch_lu(AtA)
 c(P, L, U) %<-% torch_lu_unpack(lu[[1]], lu[[2]]) 
 Atb <- P$t()$matmul(Atb)
 y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
 )[[1]]
 x <- torch_triangular_solve(y, U)[[1]]
 x
}

# torch's LU solver
ls_lu_solve <- function(A, b) {
 AtA <- A$t()$matmul(A) 
 Atb <- A$t()$matmul(b)
 lu <- torch_lu(AtA)
 m = lu[[1]]
 pivots = lu[[2]]
 x <- torch_lu_solve(Atb$unsqueeze(2), m, pivots)
 x
}

# QR factorization
# A x = b
# Q R x = b
# R x = Q_t b 
ls_qr <- function(A, b) {
 c(Q, R) %<-% linalg_qr(A)
 Qtb <- Q$t()$matmul(b)
 x <- torch_triangular_solve(Qtb$unsqueeze(2), R)[[1]]
 x
}

# SVD
# A x = b
# U S V_ x = b
# S V_t x = U_t b
# S y = U_t b 
# V_t x = y
ls_svd <- function(A, b) {
 c(U, S, Vt) %<-% linalg_svd(A, full_matrices = FALSE)
 Utb <- U$t()$matmul(b)
 y <- Utb / S
 x <- Vt$t()$matmul(y)
 x
}

# torch's general least squares solver
ls_lstsq <- function(A, b) {
 x <- linalg_lstsq(A, b)
 x
}
```

*我们使用`bench`包来分析这些方法。`mark()`函数的功能远不止跟踪时间；然而，在这里我们只是简单看一下执行时间的分布（图 24.1）：

```r
set.seed(777)
torch_manual_seed(777)
library(bench)
library(ggplot2)

res <- mark(ls_normal_eq(A, b),
 ls_cholesky_diy(A, b),
 ls_cholesky_solve(A, b),
 ls_lu_diy(A, b),
 ls_lu_solve(A, b),
 ls_qr(A, b),
 ls_svd(A, b),
 ls_lstsq(A, b)$solution,
 min_iterations = 1000)

autoplot(res, type = "ridge") + theme_minimal()
```

*![每种方法的执行时间密度图，每行一个。这里不深入细节，因为我只是想展示基准测试是如何进行的。](img/f4e746f40f968adcda1540fa0621b449.png)

图 24.1：通过示例计时最小二乘算法。

总之，我们看到了不同的矩阵分解方式如何有助于解决最小二乘问题。我们还快速展示了一种计时这些策略的方法；然而，速度并不是唯一重要的。我们还想确保解决方案的可靠性。这里的术语是*稳定性*。
  
## 24.3 快速了解稳定性

我们已经讨论过条件数了。稳定性概念在精神上相似，但指的是一个 *算法* 而不是 *矩阵*。在两种情况下，想法是计算输入的微小变化应导致输出的微小变化。已经有许多书籍专门讨论这个主题，所以我就不深入细节了²。

相反，我将使用一个病态的最小二乘问题示例——这意味着矩阵是病态的——以便我们形成对我们所讨论的算法稳定性的概念³。

预测矩阵是一个 100 x 15 的 Vandermonde 矩阵，创建方式如下：

```r
set.seed(777)
torch_manual_seed(777)

m <- 100
n <- 15
t <- torch_linspace(0, 1, m)$to(dtype = torch_double())

A <- torch_vander(t, N = n, increasing = TRUE)$to(
 dtype = torch_double()
)
```

*条件数非常高：

```r
linalg_cond(A)
```

```r
torch_tensor
2.27178e+10
[ CPUDoubleType{} ]
```

当我们将它与其转置相乘时，条件数更高——记住，一些算法实际上需要与 *这个* 矩阵一起工作：

```r
linalg_cond(A$t()$matmul(A))
```

```r
torch_tensor
7.27706e+17
[ CPUDoubleType{} ]
```

接下来，我们有预测目标：

```r
b <- torch_exp(torch_sin(4*t))
b <- b/2006.787453080206
```

*在上面的例子中，我们所有方法都得到了相同的 RMSE。这里会发生什么将很有趣。我将限制自己查看之前展示的方法中的“DIY”方法。这里它们再次列出，以方便起见：

```r
# normal equations
ls_normal_eq <- function(A, b) {
 AtA <- A$t()$matmul(A)
 x <- linalg_inv(AtA)$matmul(A$t())$matmul(b)
 x
}

# normal equations and Cholesky decomposition (done manually)
# A_t A x = A_t b
# L L_t x = A_t b
# L y = A_t b 
# L_t x = y
ls_cholesky_diy <- function(A, b) {
 # add a small multiple of the identity matrix 
 # to counteract numerical instability
 # if Cholesky decomposition fails in your 
 # setup, increase eps
 eps <- 1e-10
 id <- eps * torch_diag(torch_ones(dim(A)[2]))
 AtA <- A$t()$matmul(A) + id
 Atb <- A$t()$matmul(b)
 L <- linalg_cholesky(AtA)
 y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
 )[[1]]
 x <- torch_triangular_solve(y, L$t())[[1]]
 x
}

# normal equations and LU factorization (done manually)
# A_t A x = A_t b
# P L U x = A_t b
# L y = P_t A_t b          # where y = U x
# U x = y
ls_lu_diy <- function(A, b) {
 AtA <- A$t()$matmul(A)
 Atb <- A$t()$matmul(b)
 lu <- torch_lu(AtA)
 c(P, L, U) %<-% torch_lu_unpack(lu[[1]], lu[[2]]) 
 Atb <- P$t()$matmul(Atb)
 y <- torch_triangular_solve(
 Atb$unsqueeze(2),
 L,
 upper = FALSE
 )[[1]]
 x <- torch_triangular_solve(y, U)[[1]]
 x
}

# QR factorization
# A x = b
# Q R x = b
# R x = Q_t b 
ls_qr <- function(A, b) {
 c(Q, R) %<-% linalg_qr(A)
 Qtb <- Q$t()$matmul(b)
 x <- torch_triangular_solve(Qtb$unsqueeze(2), R)[[1]]
 x
}

# SVD
# A x = b
# U S V_ x = b
# S V_t x = U_t b
# S y = U_t b 
# V_t x = y
ls_svd <- function(A, b) {
 c(U, S, Vt) %<-% linalg_svd(A, full_matrices = FALSE)
 Utb <- U$t()$matmul(b)
 y <- Utb / S
 x <- Vt$t()$matmul(y)
 x
}
```

*让我们看看吧！

```r
algorithms <- c(
 "ls_normal_eq",
 "ls_cholesky_diy",
 "ls_lu_diy",
 "ls_qr",
 "ls_svd"
)

rmses <- purrr::map(
 algorithms,
 function(m) {
 rmse(
 as.numeric(b),
 as.numeric(A$matmul(get(m)(A, b)))
 )
 }
)

rmse_df <- data.frame(
 method = algorithms,
 rmse = unlist(rmses)
)

rmse_df
```

```r
 method         rmse
1    ls_normal_eq 2.882399e-03
2 ls_cholesky_diy 1.373906e-06
3       ls_lu_diy 1.274305e-07
4           ls_qr 3.436749e-08
5          ls_svd 3.436749e-08
```

这相当令人印象深刻！我们清楚地看到，尽管正常方程很简单，但当问题不再是有良好条件的时，可能并不是最佳选择。Cholesky 分解以及 LU 分解表现更好；然而，明显的“赢家”是 QR 分解和 SVD。难怪这两个（各有两个变体）被 `linalg_lstsq()` 所使用。

Cho, Dongjin, Cheolhee Yoo, Jungho Im, and Dong-Hyun Cha. 2020\. “Comparative Assessment of Various Machine Learning-Based Bias Correction Methods for Numerical Weather Prediction Model Forecasts of Extreme Air Temperatures in Urban Areas.” *Earth and Space Science* 7 (4): e2019EA000740\. https://doi.org/[`doi.org/10.1029/2019EA000740`](https://doi.org/10.1029/2019EA000740).Trefethen, Lloyd N., and David Bau. 1997\. *Numerical Linear Algebra*. SIAM.
 
 * *

1.  上文中引用的 `driver` 的文档基本上是 [LAPACK](https://www.netlib.org/lapack/lug/node27.html) 中相应文档的摘录，我们可以很容易地验证这一点，因为相关的页面已经被方便地链接到了 `linalg_lstsq()` 的文档中。↩︎

1.  要了解更多信息，可以考虑查阅这些书籍之一，例如，广泛使用的（且简洁的）Trefethen 和 Bau 的处理方法（1997）。↩︎

1.  这个例子取自上面脚注中提到的 Trefethen 和 Bau 的书籍。感谢 Rachel Thomas，她通过在她的 [数值线性代数课程](https://github.com/fastai/numerical-linear-algebra)中使用它，让我注意到了这一点。↩︎

