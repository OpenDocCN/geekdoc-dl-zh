<!--yml

类别：未分类

日期：2024-07-01 18:18:11

-->

# 类型卡塔：区分具有相同基础表示的不同数据：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/08/type-kata-newtypes/`](http://blog.ezyang.com/2010/08/type-kata-newtypes/)

*双关语是最低级的幽默形式。也是无尽的错误来源。*

*命令式。* 在编程中，语义上不同的数据可能具有相同的表示（类型）。使用这些数据需要手动跟踪关于数据的额外信息。当另一种解释大部分时间是正确的时，这是危险的；不完全理解所有额外条件的程序员会被安全感所蒙蔽，可能编写看似正常工作但实际上存在微妙错误的代码。以下是一些真实世界的例子，其中很容易混淆语义。

*变量和字面值。* 以下是布尔变量（`x, y, z`）和布尔字面值（`x`或`not x`）的空间高效表示。布尔变量简单地从零开始计数，但布尔字面值被左移，最低有效位用于存储补码信息。

```
int Gia_Var2Lit( int Var, int fCompl )  { return Var + Var + fCompl; }
int Gia_Lit2Var( int Lit )              { return Lit >> 1;           }

```

那么，考虑以下函数：

```
int Gia_ManHashMux( Gia_Man_t * p, int iCtrl, int iData1, int iData0 )

```

不清楚`iCtrl`、`iData1`和`iData0`参数是否对应字面值或变量：只有了解这个函数的作用（禁止带补码输入的多路复用器是没有意义的）或检查函数体才能确定这个问题（函数体调用`Gia_LitNot`）。幸运的是，由于移位错误地将字面值解释为变量（或反之），通常会导致致命错误。（来源：[ABC](http://www.eecs.berkeley.edu/~alanmi/abc/)）

*指针位。* 众所周知，指针的低两位通常是未使用的：在 32 位系统上，32 位整数是最细粒度的对齐方式，这迫使任何合理的内存地址必须能被四整除。空间高效的表示可能会使用这两个额外位来存储额外信息，但在解引用指针时需要屏蔽这些位。在我们之前的例子基础上，考虑一个变量和字面值的指针表示：如果普通指针表示一个变量，那么我们可以使用最低位来指示变量是否被补码，以实现字面值表示。

考虑以下函数：

```
Gia_Obj_t *  Gia_ObjFanin0( Gia_Obj_t * pObj );

```

其中`iDiff0`是`Gia_Obj_t`结构体中的一个`int`字段。不清楚输入指针或输出指针是否可能带补码。实际上，输入指针不得带补码，输出指针永远不会带补码。

最初可能会误将输出指针的互补视为无害：所有发生的只是将低两位掩码掉，这在正常指针上是无操作的。然而，这实际上是一个关键逻辑错误：它假设返回的指针的最低位说出了风扇输入是否被补充，而实际上返回的位将始终为零。（来源：[ABC](http://www.eecs.berkeley.edu/~alanmi/abc/)）

*物理内存和虚拟内存。* 在构建操作系统的过程中，内存管理是其中的一个重要步骤。在实现此过程时，关键的区别在于物理内存（实际硬件上的内容）和虚拟内存（由 MMU 翻译的内容）。以下代码来自学生构建的玩具操作系统框架：

```
/* This macro takes a kernel virtual address -- an address that points above
 * KERNBASE, where the machine's maximum 256MB of physical memory is mapped --
 * and returns the corresponding physical address.  It panics if you pass it a
 * non-kernel virtual address.
 */
#define PADDR(kva)                                          \
({                                                          \
        physaddr_t __m_kva = (physaddr_t) (kva);            \
        if (__m_kva < KERNBASE)                                     \
                panic("PADDR called with invalid kva %08lx", __m_kva);\
        __m_kva - KERNBASE;                                 \
})

/* This macro takes a physical address and returns the corresponding kernel
 * virtual address.  It panics if you pass an invalid physical address. */
#define KADDR(pa)                                           \
({                                                          \
        physaddr_t __m_pa = (pa);                           \
        uint32_t __m_ppn = PPN(__m_pa);                             \
        if (__m_ppn >= npage)                                       \
                panic("KADDR called with invalid pa %08lx", __m_pa);\
        (void*) (__m_pa + KERNBASE);                                \
})

```

请注意，尽管代码使用了类型同义词`uintptr_t`（虚拟地址）和`physaddr_t`（物理地址）来区分，但编译器不会阻止学生混淆两者。（来源：[JOS](http://pdos.csail.mit.edu/6.828/2009/overview.html)）

*字符串编码。* 给定任意字节序列，没有一个标准的解释说明这些字节应该如何理解成人类语言。解码器根据超出带外数据（如 HTTP 标头）或带内数据（如元标签）来确定字节的可能含义，然后将字节流转换为更结构化的内部内存表示（在 Java 中为 UTF-16）。然而，在许多情况下，原始的字节序列是数据的最有效表示方式：考虑 UTF-8 和 UCS-32 在拉丁文本上的空间差异。这鼓励开发人员使用本机字节串来传递数据（PHP 的字符串类型只是一个字节串），但如果不跟踪适当的编码，这可能会导致[无尽的问题](http://en.wikipedia.org/wiki/Mojibake)。Unicode 规范化形式的存在进一步加剧了这一问题，这使得不能对不在同一规范化形式中的 Unicode 字符串进行有意义的相等性检查（或可能完全未规范化）。

*字节序。* 给定四个字节对应的 32 位整数，没有一个标准的“数值”可以分配给这些字节：你得到的数字取决于系统的字节序。字节序列`0A 0B 0C 0D`可以被解释为`0x0A0B0C0D`（大端序）或`0x0D0C0B0A`（小端序）。

*数据验证。* 给定一个表示人类的数据结构，包含“真实姓名”、“电子邮件地址”和“电话号码”等字段，可能有两种不同的数据解释：数据被信任为正确，可以直接用于执行操作（例如发送电子邮件），或者数据未经验证，直到处理后才能被信任。程序员必须记住数据的状态，或者强制特定表示永远不包含未验证的数据。“污点”是一种语言特性，动态跟踪此数据的验证/未验证状态。

*挑战任务。* 每当数据结构（无论简单还是复杂）可能有多种解释时，对每种解释都进行一次`newtype`。

```
newtype GiaLit = GiaLit { unGiaLit :: CInt }
newtype GiaVar = GiaVar { unGiaVar :: CInt }

-- accessor functions omitted for brevity; they should be included

newtype CoGia_Obj_t = CoGia_Obj_t (Gia_Obj_t)

newtype PhysAddr a = PhysAddr (Ptr a)
newtype VirtualAddr a = VirtualAddr (Ptr a)

newtype RawBytestring = RawBytestring ByteString
-- where e is some Encoding
newtype EncodedBytestring e = EncodedBytestring ByteString
-- where n is some Normalization
newtype UTF8Bytestring n = UTF8Bytestring ByteString
type Text = UTF8Bytestring NFC

-- where e is some endianness
newtype EndianByteStream e = EndianByteStream ByteString

newtype Tainted c = Tainted c
newtype Clean c = Clean c

```

辨别数据可能存在多种解释的情况可能并不明显。如果你处理的是你没有创建的底层表示，请仔细查看变量命名和看起来在同一类型之间进行相互转换的函数。如果你设计高性能数据结构，请确定*你的*原始数据类型（这些类型不同于`int`、`char`、`bool`，通用编程语言的原始类型）。随着代码增加新功能，多种解释可能逐渐出现：要愿意重构（可能破坏 API 兼容性）或者推测性地为重要的用户可见数据创建新类型。

有关新类型的常见抱怨是类型的包装和解包。虽然这部分是必要之恶，但普通用户通常不需要包装和解包新类型：内部表示应保持隐藏！（这是一个密切相关但正交的属性，新类型有助于强制执行。）尽量不要导出新类型构造函数；而是导出智能构造函数和解构函数，进行运行时合理性检查，并以`unsafe`为前缀。

当底层值被新类型包装时，你告诉编译器你相信该值在该新类型下有一个有意义的解释：当你包装某些东西时要做好你的功课！相反，你应假设传入的新类型具有由该新类型隐含的不变量（它是有效的 UTF-8 字符串，其最低有效位为零等）：让静态类型检查器为你完成这项工作！新类型在编译时没有运行时开销：它们严格在编译时检查。

*适用性。* 新类型不能替代适当的数据结构：不要试图在 HTML 的字节字符串上进行 DOM 转换。即使在底层表示仅有一种解释的情况下，新类型也可能有用——然而，即时的好处主要来自封装。然而，当表达方式存在多种解释时，新类型是*必不可少的*：不要出门忘记它们！
