<!--yml

类别：未分类

日期：2024-07-01 18:17:55

-->

# 使用错误变量导致的错误：ezyang's 博客

> 来源：[`blog.ezyang.com/2011/04/bugs-from-using-the-wrong-variable/`](http://blog.ezyang.com/2011/04/bugs-from-using-the-wrong-variable/)

我原本应该在今天发布关于 Hoopl 的另一篇文章，但当我写的一个示例程序触发了我认为是 Hoopl 的一个 bug 时（如果这不是一个 bug，那么我的关于 Hoopl 内部工作方式的心理模型严重错误，我也不应该写这个），所以今天的文章将是关于所谓的 Hoopl 遇到的 bug：使用错误变量导致的 bug。

如果我没记错，使用了错误的变量就是缺少撇号：

```
          ; let (cha', fbase') = mapFoldWithKey
-                                (updateFact lat lbls)
+                                (updateFact lat lbls')
                                 (cha,fbase) out_facts

```

实际上，这种 bug 在函数式代码中经常发生。这里是最近我与 Simon Marlow 一起修复的 GHC 本地代码生成后端中的一个 bug：

```
-       return (Fixed sz (getRegisterReg use_sse2 reg) nilOL)
+       return (Fixed size (getRegisterReg use_sse2 reg) nilOL)

```

几周前，当我在处理 abcBridge 时，由于类似以下原因导致无限循环：

```
      cecManVerify :: Gia_Man_t -> Cec_ParCec_t_ -> IO Int
-     cecManVerify a b = handleAbcError "Cec_ManVerify" $ cecManVerify a b
+     cecManVerify a b = handleAbcError "Cec_ManVerify" $ cecManVerify' a b

```

如何防范这些错误？有多种选择：

### 将旧变量变异/遮蔽掉

对于任何命令式程序员来说，这是经典的解决方案：如果某个值不再使用，用新值覆盖它。因此，您会得到这样的代码：

```
$string = trim($string);
$string = str_replace('/', '_', $string);
$string = ...

```

在函数式编程语言中，您可以通过重新使用名称来创建新的绑定，这将*遮蔽*旧的绑定。但是这种做法有些不鼓励，因为`-Wall`可能会建议：

```
test.hs:1:24:
    Warning: This binding for `a' shadows the existing binding
               bound at test.hs:1:11

```

### 使用点无关风格消除变量

特定情况下，如果变量只在一个地方使用，在这种管道样式中通过一系列函数可以相对直接地消除它，将代码移至点无关风格（“点”在“点无关”中指的是变量名）：

```
let z = clipJ a . clipI b . extendIJ $ getIJ (q ! (i-1) ! (j-1))

```

但是当中间值需要多次使用时，这种方式通常效果不佳。通常可以安排一种方法，但是“多次使用”通常是点无关风格变得难以理解的一个很好的指标。

### 视图模式

视图模式是一种相当巧妙的语言扩展，允许您避免编写类似这样的代码：

```
f x y = let x' = unpack x
        in ... -- using only x'

```

使用 `{-# LANGUAGE ViewPatterns #-}`，您可以改写为：

```
f (unpack -> x) y = ... -- use x as if it were x'

```

因此避免了创建可能会意外使用的临时名称的需要，同时允许您使用名称。

### 打开警告

只需要几秒钟的凝视就能看出这段代码有什么问题：

```
getRegister (CmmReg reg)
  = do use_sse2 <- sse2Enabled
       let
         sz = cmmTypeSize (cmmRegType reg)
         size | not use_sse2 && isFloatSize sz = FF80
              | otherwise                      = sz
       --
       return (Fixed sz (getRegisterReg use_sse2 reg) nilOL)

```

是的，`size`在函数体中从未被使用。 GHC 会提醒您这一点：

```
test.hs:1:24: Warning: Defined but not used: `b'

```

不幸的是，有人把它关闭了（眩光）：

```
{-# OPTIONS -w #-}
-- The above warning supression flag is a temporary kludge.
-- While working on this module you are encouraged to remove it and fix
-- any warnings in the module. See
--     http://hackage.haskell.org/trac/ghc/wiki/Commentary/CodingStyle#Warnings
-- for details

```

### 使用描述性的变量名和类型

Haskell 程序员倾向于使用像`f, g, h, x, y, z`这样的短、数学风格的名称，当变量的作用域不是很大时。命令式编程者倾向于觉得这有些奇怪和难以维护。在 Haskell 中，这种风格能够被维护的原因在于静态类型系统：如果我写的函数是`compose f g`，其中`f :: a -> b`，`g :: b -> c`，我可以确定不会意外地在`f`的位置使用`g`：它会导致类型错误！如果所有关于变量内容的语义信息都包含在类型中，重复说明似乎没有多大意义。当然，不要在这个方向上走得太远是好的：当有两个变量都具有`Int`类型时，类型检查器将无法帮助你很多。在这种情况下，最好稍微多加一点描述。相反，如果你调整类型使得这两个变量再次具有不同的类型，错误的可能性再次消失。
