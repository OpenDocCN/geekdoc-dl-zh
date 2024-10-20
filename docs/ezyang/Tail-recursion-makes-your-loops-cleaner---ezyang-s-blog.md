<!--yml

category: 未分类

date: 2024-07-01 18:17:46

-->

# 尾递归使您的循环更清晰：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/05/tail-recursion-makes-your-loops-cleaner/`](http://blog.ezyang.com/2011/05/tail-recursion-makes-your-loops-cleaner/)

## 尾递归使您的循环更清晰

递归是函数式编程语言擅长的事情之一，但让人有点失望的是，在许多情况下，您必须将美丽的递归函数转换回迭代形式。毕竟，迭代是命令式语言最擅长的，对吧？

实际上，在函数式编程语言中，显式尾递归函数可以非常美丽：事实上，在复杂循环的情况下，它们甚至可以比它们的命令式对应物更漂亮。以这个中点画线算法为例：

```
circleMidpoint d r = go 0 (-r) k0
    where k0 = 5 - 4 * r
          x1 = ceiling (fromIntegral r / sqrt 2)
          go x y k | x > x1    = return ()
                   | k > 0     = d (x,y) >> go (x+1) (y+1) (k+8*x+8*y+20)
                   | otherwise = d (x,y) >> go (x+1)  y    (k+8*x+12)

```

有三个循环变量：`x`、`y` 和 `k`，根据不同的条件，其中一些变量以不同的方式更新。`x` 是一个标准的循环变量；老式的 C 风格的 `for` 循环可以很好地处理它。但是 `y` 和 `k` 根据一些循环条件以不同的方式更新。但由于它们是 `go` 辅助函数的参数，总是清楚地知道哪些经常变化的变量。在命令式翻译中，您会失去这种良好的结构：

```
// global variables and loop variables are all mixed together
int k = 5 - 4 * r;
int y = -r;
int x1 = ceil(r/sqrt(2));
for (int x = 0; x <= x1; x++) { // only x is obviously an index var
  draw(x, y);
  if (k > 0) {
    y++;
    k += 8*x + 8*y + 20;
  } else {
    k += 8*x + 12;
  }
  // does it ever make sense for any code to live here?
}

```

我在这个过程中还设法引入了一个错误...
