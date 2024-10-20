<!--yml

category: 未分类

date: 2024-07-01 18:17:16

-->

# Rust 开发者都应该了解的借用检查器中的两个 bug：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/12/two-bugs-in-the-borrow-checker-every-rust-developer-should-know-about/`](http://blog.ezyang.com/2013/12/two-bugs-in-the-borrow-checker-every-rust-developer-should-know-about/)

如果是这样的话，你可能已经遇到了借用检查器中两个臭名昭著的 bug 之一。在这篇文章中，我想描述这两个 bug，给出它们可能出现的情况，并描述一些解决方法。希望这类文章很快就会过时，但它们的修复方法相当复杂，如果你今天尝试在 Rust 中编程，不可避免地会遇到这些 bug。

### 可变借用过于急切（#6268）

*总结。* 当你使用 `&mut`（无论是显式还是隐式）时，Rust 会立即将 lvalue 视为借用，并强加其限制（例如，lvalue 不能再次借用）。然而，在许多情况下，借用指针直到后来才会被使用，因此立即强加限制可能会导致错误。当存在 *隐式* 使用 `&mut` 时，这种情况最有可能发生。（[Bug #6268](https://github.com/mozilla/rust/issues/6268)）

*症状。* 你会收到错误消息“因为它也作为不可变借用，所以无法借用 `foo`”，但报告的第二次借用是对象调度方法调用，或者在标记的借用发生时看起来不应该被借用。

*示例。* 原始的 bug 报告描述了嵌套方法调用的情况，其中外部方法调用在其签名中有 `&mut self`：

```
fn main() {
  let mut map = std::hashmap::HashMap::new();
  map.insert(1, 2);
  map.insert(2, *map.get(&1)); // XXX
}

test.rs:4:17: 4:20 error: cannot borrow `map` as immutable because it is also borrowed as mutable
test.rs:4   map.insert(2, *map.get(&1)); // XXX
                           ^~~
test.rs:4:2: 4:5 note: second borrow of `map` occurs here
test.rs:4   map.insert(2, *map.get(&1)); // XXX
            ^~~

```

这段代码希望获取键为 `1` 的值并存储在键为 `2` 的位置。为什么会失败呢？考虑签名 `fn insert(&mut self, key: K, value: V) -> bool`：在尝试评估其参数之前，`insert` 方法调用会立即对 `map` 获取一个 `&mut` 借用。如果我们展开方法调用，顺序就变得清楚了：`HashMap::insert(&mut map, 2, *map.get(&1))`（注意：此语法尚未实现）。因为 Rust 会从左到右评估参数，这等效于：

```
let x_self : &mut HashMap<int> = &mut map;
let x_arg1 : int = 2;
let x_arg2 : int = *map.get(&1); // XXX
HashMap::insert(x_self, x_arg1, x_arg2);

```

意味着在调用 `map.get` 时存在活跃的借用。通过进行轻微的重写可以解决该问题：

```
fn main() {
  let mut map = std::hashmap::HashMap::new();
  map.insert(1, 2);
  let x = *map.get(&1);
  map.insert(2, x);
}

```

敏感到参数顺序的问题，即使没有涉及方法调用。下面是另一个例子，其中没有方法调用：

```
fn g(x: &mut int) -> int { *x }
fn f(x: &mut int, y: int) { *x += y; }
fn main() {
    let mut a = 1;
    f(&mut a, g(&mut a));
}

```

*讨论。* 幸运的是，这个 bug 很容易解决，虽然有点恼人：在不幸的可变借用之前将所有子表达式移动到 let 绑定中（请参见示例以获取详细操作）。注意：这些子表达式中发生的借用确实必须是临时的；否则，你会遇到合法的“无法两次借用可变”的错误。

### 借用范围不应总是按词法作用域处理（#6393）

*摘要.* 当您借用一个指针时，Rust 为其分配一个构成其生命周期的词法范围。这个范围可以小到一个语句，也可以大到整个函数体。然而，Rust 无法计算非词法的生命周期，例如，一个借用的指针仅在函数的一半之前有效。因此，借用可能比用户预期的时间更长，导致借用检查器拒绝某些语句。([Bug #6393](https://github.com/mozilla/rust/issues/6393))

*症状.* 您收到“因为它也作为不可变/可变的借用而无法将 foo 借用为不可变/可变”的错误，但您认为先前的借用应该已经过期了。

*例子.* 这个问题在各种情况下都会出现。引发此错误的最简单的示例如下所示：

```
fn main() {
    let mut x = ~1;
    let y = &mut *x;
    *y = 1;
    let z = &mut *x;
    *z = 1;
}

test.rs:5:12: 5:19 error: cannot borrow `*x` as mutable more than once at a time
test.rs:5     let z = &mut *x;
                      ^~~~~~~
test.rs:3:12: 3:19 note: second borrow of `*x` as mutable occurs here
test.rs:3     let y = &mut *x;
                      ^~~~~~~

```

显然，在`*y = 1`之后，`y`已经无效了，但是借用检查器无法看到这一点。幸运的是，在这种情况下，很容易添加一个新的词法范围来解决这个问题：

```
fn main() {
    let mut x = ~1;
    {
        let y = &mut *x;
        *y = 1;
    }
    let z = &mut *x;
    *z = 1;
}

```

那么，这实际上何时成为问题呢？通常的罪魁祸首是`match`语句。这里是涉及映射的一些常见代码，您可能希望编写：

```
extern mod extra;
fn main() {
    let mut table = extra::treemap::TreeMap::new();
    let key = ~"test1";
    match table.find_mut(&key) {
        None    => table.insert(key.clone(), ~[1]), // XXX
        Some(v) => { v.push(1); false }
    };
}

test.rs:6:19: 6:24 error: cannot borrow `table` as mutable more than once at a time
test.rs:6         None    => table.insert(key.clone(), ~[1]), // XXX
                             ^~~~~
test.rs:5:10: 5:15 note: second borrow of `table` as mutable occurs here
test.rs:5     match table.find_mut(&key) {
                    ^~~~~

```

`table`是整数键到向量的映射。代码在`key`处进行插入：如果映射中没有条目，则创建一个新的单元素向量并将其插入该位置；否则，只需将值`1`推送到现有向量中。为什么`table`在`None`分支中被借用？直觉上，对于`table.find_mut`的借用应该是无效的，因为我们不再使用任何结果；然而对于 Rust 来说，它只能将借用指针分配给整个`match`语句的词法范围，因为借用指针在`Some`分支中继续使用（请注意，如果删除`Some`分支，则此借用检查）。不幸的是，无法像前面的示例那样插入新的词法范围。（在发布时，我找不到仅使用`if`的小示例。）

有时，与变量相关的*生命周期*可能会强制将其分配给比您预期的更大的词法范围。[Issue #9113](https://github.com/mozilla/rust/issues/9113)提供了一个很好的例子（以下是代码摘录）：

```
pub fn read1<'a>(&'a mut self, key: int) -> Option<&'a Data> {
    match self.cache.find(&key) {
        Some(data) => return Some(data),
        None => ()
    };
    match self.db.find(&key) {
        Some(data) => {
            let result: &Data = self.cache.find_or_insert(key, data.clone());
            Some(result)
        },
        None => None
    }
}

test.rs:22:36: 22:46 error: cannot borrow `(*self).cache` as mutable because it is also borrowed as immutable
test.rs:22                 let result: &Data = self.cache.find_or_insert(key, data.clone());
                                               ^~~~~~~~~~
test.rs:15:14: 15:24 note: second borrow of `(*self).cache` occurs here
test.rs:15         match self.cache.find(&key) {
                         ^~~~~~~~~~

```

这段代码试图执行数据库查找；它首先查看缓存并返回缓存的条目（如果有）。否则，它在数据库中查找该值，并在此过程中缓存该值。通常情况下，您希望在第一个匹配中对`self.cache`的借用仅扩展到第一个表达式。然而，`return`语句却对此产生了影响：它强制`data`的生命周期为`'a`，包含整个函数体。借用检查器因此得出结论，在函数的任何地方都存在借用，即使函数在获取此借用后立即返回。

*讨论.* 解决方法取决于导致问题的范围的性质。当涉及`match`时，通常可以安排执行不良借用操作，该操作位于`match`语句之外，位于一个新的、非重叠的词法范围内。当相关分支不依赖于模式匹配中的任何变量时，可以使用短路控制运算符：

```
extern mod extra;
use extra::treemap::TreeMap;
fn main() {
    let mut table: TreeMap<~str,~[int]> = TreeMap::new();
    let key = ~"test1";
    match table.find_mut(&key) {
        None    => {},
        Some(v) => { v.push(1); return }
    };
    table.insert(key.clone(), ~[1]); // None-case
}

```

或者，与其直接返回，`match`语句可以分配一个布尔值，以指示是否应运行`None`情况：

```
extern mod extra;
use extra::treemap::TreeMap;
fn main() {
    let mut table: TreeMap<~str,~[int]> = TreeMap::new();
    let key = ~"test1";
    let is_none = match table.find_mut(&key) {
        None    => true,
        Some(v) => { v.push(1); false }
    };
    if is_none {
        table.insert(key.clone(), ~[1]);
    }
}

```

可以将布尔值详细说明为一个`enum`，其中包含可能需要的模式匹配中的任何非引用。请注意，对于借用引用，这种方法不起作用；但在这种情况下，借用确实仍然活跃！

关于生命周期问题的解决方法要困难一些，因为在函数中没有指针不“被借用”的地方。在某些情况下可以起作用的一个技巧是将函数转换为延续传递风格：即，不是返回借用的指针，而是接受一个函数参数，并在函数中调用它。[pnkfelix](https://github.com/mozilla/rust/issues/9113#issuecomment-24407530)描述了如何修复第三个例子。这消除了变量的生命周期约束并解决了问题。

分配给借用的词法范围可能对代码扰动非常敏感，因为删除对借用的使用可能会导致 Rust 分配（更）小的词法范围给借用，这可能会消除错误。有时，可以通过避免借用来完全避免问题。
