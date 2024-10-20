<!--yml

category: 未分类

date: 2024-07-01 18:18:25

-->

# 用 Haskell 替换小型 C 程序：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/`](http://blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/)

## 用 Haskell 替换小型 C 程序

C 作为小型程序的经典选择，速度非常快。当[scripts.mit.edu](http://scripts.mit.edu/)需要一个小程序来实现[一个增强版的 cat](http://scripts.mit.edu/trac/browser/trunk/server/common/oursrc/execsys/static-cat.c.pre)，并且在输出开头添加有用的 HTTP 头时，毫无疑问：它将会用 C 语言编写，并且速度非常快；我们静态内容的服务速度取决于它！（技术细节：我们的 Web 服务器基于网络文件系统，并且我们希望避免在 Apache 被入侵时给予它过多的凭证。因此，我们修改了内核以强制执行额外的规定，即必须以某个用户 ID 的身份运行才能从文件系统中读取这些文件。Apache 以自己的用户身份运行，因此我们需要另一个*小*程序作为中间人。）

它也是一个 Frankenscript，这是一个根据我们项目非常特定需求而发展出来的程序，在世界上其他地方找不到。因此，保证程序简洁和定义清晰非常重要；这两个特性在 C 代码中很难达到。当您想要添加功能时情况会变得更糟。有一些小功能（最后修改头，字节范围），以及一些大功能（FastCGI 支持）。没有开发团队愿意考虑将 C 文件的大小加倍以添加所有这些增强功能，并且将程序重写为脚本语言将导致性能下降。用 Perl CGI 替换脚本的基准测试使脚本慢了十倍（这在进行端到端 Apache 测试时转化为四倍慢）。

但是还有另一种方式！Anders 写道：

> 所以我意识到：用编译的 Haskell CGI 脚本替换它可能会让我们保持相同的性能。而且由于 Haskell 的 FastCGI 库具有相同的接口，易于移植到 FastCGI。

几周后，呈现出：[Haskell 中的 static-cat](http://andersk.mit.edu/gitweb/scripts-static-cat.git)。然后我们看到以下基准测试：

```
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.cgi/hello/hello.html
Requests per second:    15.68 [#/sec] (mean)
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.perl.cgi/hello/hello.html
Requests per second:    7.50 [#/sec] (mean)
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.c.cgi/hello/hello.html
Requests per second:    16.59 [#/sec] (mean)

```

微基准测试显示没有 Apache 时有 4ms 的差异，Anders 怀疑这是由于 Haskell 可执行文件的大小。肯定需要进行一些性能调查，但 Haskell 版本在端到端测试中比 Perl 版本快两倍以上。

更一般地说，编译成本地代码的语言类别（Haskell 只是其中之一）似乎越来越成为取代具有高性能要求的紧凑 C 程序的吸引力选择。这是相当令人兴奋的，尽管这取决于您是否能说服开发团队将 Haskell 引入您使用的语言组合中是一个好主意。关于这一点，我将在另一篇博客文章中详细讨论。
