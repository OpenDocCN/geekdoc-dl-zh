<!--yml

category: 未分类

date: 2024-07-01 18:18:12

-->

# 如何在 Haskell 中选择你的字符串库：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/08/strings-in-haskell/`](http://blog.ezyang.com/2010/08/strings-in-haskell/)

## 如何在 Haskell 中选择你的字符串库

*注意。* 在来自布莱恩·奥沙利文的批评后，我重构了页面。

“不同的文本处理库如何比较，我们在什么情况下应该使用哪个包？” [克里斯·艾德霍夫提问](http://blog.ezyang.com/2010/07/suggestion-box/#comment-787)。后一个问题更容易回答。使用[bytestring](http://hackage.haskell.org/package/bytestring)处理二进制数据——原始的位和字节，没有关于语义含义的明确信息。使用[text](http://hackage.haskell.org/package/text)处理表示人类书面语言的 Unicode 数据，通常表示为带有字符编码的二进制数据。两者（尤其是 bytestring）广泛使用，并且很可能会成为——如果它们还没有成为的话——标准。

但是，在 Hackage 上还有很多更专业的字符串处理库。由于没有在实际项目中使用过所有这些库，我不会对它们的稳定性或实现进行评判；相反，我们将根据它们填补的特定需求对它们进行分类。有几个维度可以用来分类字符串库或模块：

+   *二进制还是文本？* 二进制是原始的位和字节：它不包含关于`0`或`0x0A`的明确信息。文本用于表示人类语言，通常是带有字符编码的二进制数据。这是程序员需要了解的[最重要的区别](http://www.joelonsoftware.com/articles/Unicode.html)。

+   如果是文本，*ASCII、8 位或 Unicode？* ASCII 简单但只支持英语；8 位（例如 Latin-1）无处不在，经常因向后兼容性而必需；Unicode 是“正确的方式”但稍微复杂。Unicode 进一步问，*内存编码是什么？* UTF-16 易于处理，而 UTF-8 对英文文本可能会节省一倍的内存。大多数语言选择 Unicode 和 UTF-16 供程序员使用。

+   *解包还是打包？* 解包字符串是本地选择，只是字符的链表。打包字符串是经典的 C 数组，允许高效的处理和内存使用。大多数语言使用打包字符串：Haskell 以其使用链表而闻名（或者说是臭名昭著）。

+   *懒惰还是严格？* 懒惰更灵活，允许诸如流式处理之类的操作。严格字符串必须完全保存在内存中，但在整个字符串需要计算的情况下可能更快。打包的懒惰表示通常使用分块来减少生成的惰性求值。毋庸置疑，严格字符串是经典解释，尽管懒惰字符串在流式处理中有用。

根据这些问题，以下是 Hackage 字符串库的分类：

除了内存编码外，还涉及到源和目标编码的问题：希望是正常的东西，但偶尔你会遇到 Shift_JIS 文本，需要对其进行处理。你可以用 [encoding](http://hackage.haskell.org/package/encoding)（处理 `String` 或严格/懒惰 `ByteString`，可以通过 `ByteSource` 和 `ByteSink` 扩展）或者 [iconv](http://hackage.haskell.org/package/iconv)（处理严格/懒惰 `ByteString`）。

*Unicode 笑话。*

```
Well done, mortal!  But now thou must face the final Test...--More--

Wizard the Evoker         St:10 Dx:14 Co:12 In:16 Wi:11 Ch:12  Chaotic
Dlvl:BMP  $:0  HP:11(11) Pw:7(7) AC:9  Xp:1/0 T:1

```

*Alt 文本。* 是的，我到了补充特殊用途平面，但后来被 TAG LATIN CAPITAL LETTER A 给干掉了。看起来像是普通的 A，所以我以为它只是一个 Archon……
