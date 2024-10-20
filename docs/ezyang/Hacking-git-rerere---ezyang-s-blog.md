<!--yml

category: 未分类

date: 2024-07-01 18:18:29

-->

# [Hacking git-rerere](http://blog.ezyang.com/2010/01/hacking-git-rerere/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/01/hacking-git-rerere/`](http://blog.ezyang.com/2010/01/hacking-git-rerere/)

Git 的一个非常规工作流，[Wizard](http://scripts.mit.edu/wizard)广泛使用，是单个开发者需要在大量工作副本中执行合并。通常，维护者会从他关心的分支拉取，并将大量工作分配给那些有兴趣贡献补丁的人。然而，Wizard 正在使用 Git 为那些不了解并且不感兴趣学习 Git 的人提供服务，因此我们需要推送更新并为他们合并他们的软件。

在进行大量合并时遇到的问题是“重复解决相同的冲突”。至少对于经典案例来说，解决方法是使用`git rerere`。此功能保存冲突的解决方案，然后如果再次遇到冲突，则自动应用这些解决方案。如果查看`man git-rerere`，您可以了解到这些信息。

不幸的是，这个合并解决数据存储在每个`.git`目录中，具体在`rr-cache`子目录中，因此需要一些适度的聪明才能使其在多个仓库中正常工作。幸运的是，将所有`rr-cache`目录符号链接到一个共同的目录的简单解决方案既有效又在最初合并时安全（写出解决方案时不是竞争安全，但我认为这种低竞争足以忽略不计）。

为什么这个解决方案是竞争安全的？初看`rerere.c`中的代码，这似乎并非如此：如果发生两次合并以生成相同的合并冲突（这正是 git rerere 的用例），则以下代码将使用相同的`hex`值执行：

```
ret = handle_file(path, sha1, NULL);
if (ret < 1)
        continue;
hex = xstrdup(sha1_to_hex(sha1));
string_list_insert(path, rr)->util = hex;
if (mkdir(git_path("rr-cache/%s", hex), 0755))
        continue;
handle_file(path, NULL, rerere_path(hex, "preimage"));

```

最后三行访问了（现在共享的）`rr-cache`目录，并且`handle_file`将尝试写出文件`rr-cache/$HEX/preimage`的预影像内容；如果两个实例同时运行`handle_file`，则此文件将被覆盖。

但事实证明，我们并不在乎；除非发生 SHA-1 碰撞，否则两个实例将写出相同的文件。`handle_file`的签名是：

```
static int handle_file(const char *path,
         unsigned char *sha1, const char *output)

```

第一个参数是从中读取冲突标记的路径，是必需的。`sha1`和`output`是可选的；如果`output`不为 NULL，则其包含整个文件的内容，*减去*任何 diff3 冲突部分（由`|||||||`和`=======`分隔）；如果`sha1`不为 NULL，则写入其内容的 20 字节二进制摘要，这些内容`output`本来会收到。于是，世界恢复了平衡。

*附录*

Anders 提出了一个有趣的问题，即两个进程是否将相同内容写入同一个文件是否真的是竞争安全的。事实上，有一个非常相似的情况涉及到两个进程将相同内容写入同一个文件，这是竞争条件的一个经典例子：

```
((echo "a"; sleep 1; echo "b") & (echo "a"; sleep 1; echo "b")) > test

```

在正常情况下，测试的内容是：

```
a
a
b
b

```

但是偶尔会出现其中一个进程输掉比赛的情况，你会得到：

```
a
a
b

```

因为写入和更新文件偏移量的非原子组合。

但是，这个例子与 Git 的情况的区别在于，在这个例子中只有一个文件描述符。然而，在 Git 的情况下，由于每个进程都独立调用 `open`，所以有两个文件描述符。一个更类似的 shell 脚本可能是：

```
((echo "a"; sleep 1; echo "b") > test & (echo "a"; sleep 1; echo "b") > test)

```

其内容（据我所知）无疑是：

```
a
b

```

现在，POSIX 实际上并没有说明如果两个写入相同偏移量相同内容的情况会发生什么。然而，简单的测试似乎表明，Linux 和 ext3 能够更强地保证写入相同值不会导致随机损坏（注意，如果文件的内容不同，则可能会有任何组合，这是实际中的情况）。
