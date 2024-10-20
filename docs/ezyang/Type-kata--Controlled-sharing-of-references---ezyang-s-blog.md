<!--yml

category: 未分类

date: 2024-07-01 18:18:11

-->

# Type kata: Controlled sharing of references : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/08/type-kata-controlled-sharing-of-references/`](http://blog.ezyang.com/2010/08/type-kata-controlled-sharing-of-references/)

*命令式.* 具有许多子对象的可变数据结构经常迫使任何给定的子对象与一个给定的父数据结构相关联：

```
class DOMNode {
  private DOMDocument $ownerDocument;
  // ...
  public void appendNode(DOMNode n) {
    if (n.ownerDocument != this.ownerDocument) {
      throw DOMException("Cannot append node that "
        "does not belong to this document");
    }
    // ...
  }
}

```

客户端代码必须小心，不要混淆属于不同所有者的子对象。对象可以通过特殊函数从一个所有者复制到另一个所有者。

```
class DOMDocument {
  public DOMNode importNode(DOMNode node) {
    // ...
  }
}

```

有时，这种风格的函数只能在特定情况下调用。如果复制了可变数据结构，并且你想引用新结构中的一个子对象，但你只有对原始结构的引用，实现可以让你转发这样的指针，但前提是目标结构是最新的副本。

```
class DOMNode {
  private DOMNode $copy;
}

```

*技巧.* [ST 单子](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Monad-ST.html)风格的幻影类型允许静态强制将不同单子所有者的子对象分离开来。

```
{-# LANGUAGE Rank2Types #-}
-- s is the phantom type
newtype DOM s a = ...
newtype Node s = ...
runDom :: (forall s. DOM s ()) -> Document
getNodeById :: Id -> DOM s (Node s)
deleteNode :: Node s -> DOM s ()

-- Does not typecheck, the second runDom uses a fresh
-- phantom variable which does not match node's
runDom $ do
  node <- getNodeById "myNode"
  let alternateDocument = runDom $ do
    deleteNode node

```

要允许任何单子的值在另一个单子中使用，请实现一个在两个幻影类型中都是多态的函数：

```
importNode :: Node s -> DOM s2 (Node s2)
setRoot :: Node s -> DOM s ()

-- This now typechecks
runDom $ do
  node <- getNodeById "myNode"
  let alternateDocument = runDom $ do
    node' <- importNode node
    setRoot node'

```

函数可能是单子的，因为实现需要知道`Node`被转换为什么所有者。

仅在特定情况下允许翻译，使用类型构造函数（可以使用空数据声明来获取这些）在幻影类型上：

```
{-# LANGUAGE EmptyDataDecls #-}
data Dup n
getNewNode :: Node s -> DOM (Dup s) (Node (Dup s))
dupDom :: DOM s () -> DOM s (DOM (Dup s) ())

-- This typechecks, and does not recopy the original node
runDom $ do
  node <- getNodeById "myNode"
  dupDom $ do
    node' <- getNewNode node
    ...

```

*适用性.* Haskell 的从业者被鼓励实现和使用纯数据结构，其中共享使得这种关于所有权的细致管理变得不必要。尽管如此，当你通过 FFI 与需要这些不变式的库进行接口时，这种技术仍然是有用的。
