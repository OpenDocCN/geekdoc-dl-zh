<!--yml

类别：未分类

日期：2024-07-01 18:18:15

-->

# 使用 get 和 set 进行数据编组：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/06/marshalling-with-get-and-set/`](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/)

这是[六部分介绍 c2hs 的第五部分](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/)。今天，我们解释如何对 C 结构体进行数据编组。

*重要提示。* `struct foo`和`foo`之间有区别；c2hs 仅认为后者是类型，因此您可能需要添加一些形式为`typedef struct foo foo`的 typedef 以便 c2hs 识别这些结构体。

*Get.* Haskell FFI 对 C 结构体一无所知；Haskell 读取结构体成员的想法是查看某个内存位置的字节偏移量，这是您手动计算的。这很可怕，而`hsc2hs`有`#peek`可以让您摆脱这种非可移植的枯燥工作。`c2hs`更简单：您可以指定`{#get StructName->struct_field #}`，c2hs 将其替换为正确类型的 lambda，执行正确类型的 peek 操作：`(\ptr -> do {peekByteOff ptr 12 ::IO CInt})`（在 IO 单子中！）请注意以下陷阱：

+   您需要手动将生成的原始 C 类型转换为更友好的 Haskell 类型，

+   表达式的左侧是*类型*或*结构名*，而不是包含您想要 peek 的指针/结构的 Haskell 变量。通常这将放在 lambda 的右侧。

`get`指令实际上比仅仅结构体访问更通用：它可以解引用指针（`*StructName`）或访问成员而不解引用（`StructName.struct_field`）。

*Set.* 与`get`相反，`set`允许您将值填充到任意内存位置。与`get`不同，传入的值需要是指针（语法使用点号）。`{#set StructName.struct_field #}`扩展为`(\ptr val -> do {pokeByteOff ptr 12 (val::CInt)})`；指针是第一个参数，值是第二个。您还需要手动转换输入值。

*定义可存储性。* 如果您不是在不透明指针中单独获取和设置结构体中的字段，创建`Storable`实例是一个好方法。然而，由于`get`和`set`创建的所有 lambda 都在 IO 单子中，组合它们可能会稍微复杂一些。审慎使用单子提升和适用实例可以使代码变得更简单，但是：

```
data StructName = StructName
  { struct_field1'StructName :: Int
  , struct_field2'StructName :: Int
  }
instance Storable StructName where
  sizeOf _ = {#sizeof StructName #}
  alignment _ = 4
  peek p = StructName
    <$> liftM fromIntegral ({#get StructName->struct_field1 #} p)
    <*> liftM fromIntegral ({#get StructName->struct_field2 #} p)
  poke p x = do
    {#set StructName.struct_field1 #} p (fromIntegral $ struct_field1'StructName x)
    {#set StructName.struct_field2 #} p (fromIntegral $ struct_field2'StructName x)

```

`StructName`中的奇怪命名约定是为了考虑不同结构体可能共享字段名，而 Haskell 字段名可能不行。

*注意。* 最近为`alignment`指令增加了 c2hs 支持，用于计算 C 数据结构的对齐。然而，截至 0.6.12，这尚未对一般公众发布。

*请求。* 描述 c2hs 的论文陈述如下：“[将复合 C 值马歇尔化为 Haskell 值] 更普遍地有用；然而，我们通常并不真正希望将整个 C 结构体马歇尔化为 Haskell。” 不幸的是，当前的 c2hs 版本没有提供任何可选功能来减少编写“直接” Storable 实例的繁琐性，这将是非常可爱的。bindings-dsl 和 GreenCard 在这方面似乎表现更好。

*下次见。* [调用和乐趣：调用重置](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)
