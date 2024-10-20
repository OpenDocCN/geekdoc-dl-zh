<!--yml

分类：未分类

日期：2024-07-01 18:18:12

-->

# 缓冲流和迭代器：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/08/buffered-streams-and-iteratee/`](http://blog.ezyang.com/2010/08/buffered-streams-and-iteratee/)

在试图找出如何更深入地解释惰性与严格字节串而不会让我的读者半途而废时，我偶然发现了在命令式语言中缓冲流的标准实现与函数式语言中的迭代器之间存在的一个有趣的对比。

没有自重的输入/输出机制会缺少*缓冲*。通过将读取或写入操作分组以便作为单个单元执行，缓冲可以提高效率。在 C 中，一个简单的读取缓冲区可以像这样实现（当然，使用静态变量封装到数据结构中...并且对`read`中的错误条件进行适当处理...）：

```
static char buffer[512];
static int pos = 0;
static int end = 0;
static int fd = 0;

int readChar() {
  if (pos >= end && feedBuf() == 0) {
    return EOF;
  }
  return (int) buffer[pos++];
}

int feedBuf() {
  pos = 0;
  end = read(fd, buffer, sizeof(buffer));
  assert(end != -1);
  return end;
}

```

导出的接口是`readChar`，每次用户调用时提供一个转换为`int`的单个`char`，但在幕后仅在缓冲区耗尽时实际从输入中读取（`pos >= end`）。

对于大多数应用程序来说，这已经足够好了：底层行为的粗糙性被一个简单而明了的函数隐藏起来。此外，我们的函数并不*过于*简单：如果我们将所有标准输入读入一个巨大的缓冲区中，直到`EOF`到来之前我们将无法做任何其他事情。在这里，我们可以在输入到来时做出反应。

在纯函数设置中，这样一组函数会是什么样子呢？一个明显的困难是`buffer`被重复地突变，因为我们执行读取操作。在持久性的精神中，我们应该尽量避免在最初填充之后对缓冲区进行变异。使缓冲区持久化意味着我们在读取更多数据时也可以保存数据而不必复制它（你可以称之为零拷贝）。我们可以使用一些简单的方式将缓冲区链接在一起：比如说，一个链表。

链表？让我们查看惰性和严格字节字符串的定义（稍作编辑，以适应你，读者）：

```
data Strict.ByteString = PS !(ForeignPtr Word8) !Int !Int
data Lazy.ByteString = Empty | Chunk !Strict.ByteString Lazy.ByteString

```

在 C 中，这些将是：

```
struct strict_bytestring {
  char *pChar;
  int offset;
  int length;
}

struct lazy_bytestring {
  struct strict_bytestring *cur;
  int forced;
  union {
    struct lazy_bytestring *next;
    void (*thunk)(struct lazy_bytestring*);
  }
}

```

严格字节串只不过是一个精心设计的、内存管理的缓冲区：两个整数跟踪偏移量和长度。在持久性存在的情况下，选择偏移量是一个特别好的选择：从字符串中取子串不再需要复制：只需创建一个新的严格字节串，适当设置偏移量和长度，并使用相同的基指针。

那么什么是`Lazy.ByteString`呢？嗯，它是一种显赫的懒惰严格字节串的惰性链表——只需将`Chunk`理解为`Cons`，将`Empty`理解为`Null`：惰性源自于对`Chunk`的第二个参数的非严格性（注意没有感叹号，这是一种严格性注释）。这种惰性是我们在`lazy_bytestring`结构中有`thunk`联合和`forced`布尔值的原因：当调用时，此 API 将新的`lazy_bytestring` scribble 到函数指针上（这与 GHC 的工作方式非常相似；少了一层或更多的间接层）。如果忽略惰性，这听起来有点像我们之前描述的缓冲区链表。

然而，有一个重要的区别。`Lazy.ByteString`是纯的：我们无法调用原始的`read`函数（一个系统调用，这使得它几乎是最 IO 的操作）。因此，当我们有一些纯计算（比如马尔可夫过程）可以生成无限量的文本时，懒惰字节串是合适的选择，但在缓冲输入方面则显得不足。

“没问题！”你可能会说，“只需将数据类型更改为持有`IO Lazy.ByteString`而不是`Lazy.ByteString`即可：

```
data IO.ByteString = Empty | Chunk !Strict.ByteString (IO IO.ByteString)

```

但是这种数据类型有些问题：没有人阻止多次调用`IO IO.ByteString`。事实上，将 IO 操作放在`Chunk`值中没有任何意义：由于文件描述符的状态性质，每次都是相同的代码：`hReadByteString handle`。我们又回到基于句柄的 IO。

`IO.ByteString`作为列表的想法是一个重要的直觉。关键的洞察力在于：谁说我们必须将 IO 操作的列表提供给用户？相反，*倒转控制*，使得用户不再调用迭代器：迭代器调用用户并将 IO 的结果返回给用户。用户反过来可以启动其他 IO 操作，或将迭代器组合在一起（我们还没有讨论过），以从一个迭代器流向另一个。

此时，我推荐参考 Oleg 的[优秀注释幻灯片（PDF）](http://okmij.org/ftp/Haskell/Iteratee/IterateeIO-talk-notes.pdf)进一步解释迭代器（不是开玩笑，幻灯片写得非常好），以及多种[迭代器](http://ianen.org/articles/understanding-iteratees/) [教程](http://cdsmith.wordpress.com/2010/05/23/iteratees-step-by-step-part-1/)。我希望对由 IO 操作生成的“缓冲区链表”进行重视，引起对迭代器本质的注意：这是一个在 IO 操作列表之上的抽象。

总结一下：

+   使用*严格的字节串*作为构建更有趣的结构的原语，这些结构具有缓冲区（尽管避免重新实现惰性字节串或迭代器）。当数据量较小、全部可以一次性初始化或随机访问、切片和其他非线性访问模式很重要时，请使用它们。

+   使用*lazy bytestrings*作为表示纯计算生成的无限数据流的机制。考虑在执行主要适合于惰性列表的操作（`concat`、`append`、`reverse`等）时使用它们。尽管模块上写着，避免在惰性 IO 时使用它们。

+   使用*iteratees*表示可以逐步处理的来自 IO 源的数据：这通常意味着大型数据集。迭代器特别适合多层逐步处理：它们可以自动且安全地“融合”处理。
