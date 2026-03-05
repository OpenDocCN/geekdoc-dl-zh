# 惊讶于 Golang 的 CPU 霸占

> 原文：<https://winder.ai/cpu-hogging-in-golang/>

在我的一个应用程序中，由于各种原因，我们现在有一个类似批处理的进程和一个基于 HTTP 的 REST 应用程序在同一个二进制文件中运行。今天，我遇到了一个问题，当批处理进程运行时，HTTP 延迟大约为 10 秒。

经过一些调试后，出现这种情况的原因是，尽管两者运行在不同的 Go 例程中，批处理进程不允许调度程序调度 HTTP 请求，直到批处理进程完成。

这是一个令人费解的例子。它运行两个 goroutines，并将线程数强制设置为 1。第一个 goroutine 是一个“cpu 密集型”例程。第二个是打印行。运行这个程序，注意它是如何等待 cpu 密集型例程完成，然后再调度其他任务的。

凭直觉，根据我们在其他语言中操作系统线程的经验，我们会期望它会产生一个线程，并立即转移到下一行。但事实并非如此。[https://play.golang.org/p/PDsbUphk5Q](https://play.golang.org/p/PDsbUphk5Q)

```py
package main

import "fmt"
import "runtime"
import "time"

func cpuIntensive(p *int64) {
  for i := int64(1); i <= 10000000; i++ {
    *p = i
  }
  fmt.Println("Done intensive thing")
}

func printVar(p *int64) {
  fmt.Printf("print x = %d.\n", *p)
}

func main() {
  runtime.GOMAXPROCS(1)

  x := int64(0)
  go cpuIntensive(&x) // This should go into background
 go printVar(&x)

  // This won't get scheduled until everything has finished.
 time.Sleep(1 * time.Nanosecond) // Wait for goroutines to finish } 
```

```py
Done intensive thing
print x = 10000000. 
```

这是因为需要手动调用 golang 调度程序来执行调度事件。现在在第 10 行添加代码:`runtime.Gosched()`。在 for 循环中。这将手动调用调度程序。现在首先打印变量，用一个较小的数字，因为我们已经手动调用了一个调度事件。

```py
print x = 1.
Done intensive thing 
```

现在，很明显，go base 代码中充满了`Gosched`调用，所以无论何时你调用`time.Sleep`或`printf`或其他什么，它都会为你安排。

所以，如果我们尝试一些更现实的东西，比如解码 JSON:[https://play.golang.org/p/qyN2iFrJED](https://play.golang.org/p/qyN2iFrJED)

```py
package main

import "fmt"
import "runtime"
import "time"
import "encoding/json"
import "strings"

const testBytes = `{ "Test": "value" }`

type Message struct {
  Test string
}

func cpuIntensive(p *Message) {
  for i := int64(1); i <= 1000; i++ {
	json.NewDecoder(strings.NewReader(testBytes)).Decode(p)
  }
  fmt.Println("Done intensive thing")
}

func printVar(p *Message) {
  fmt.Printf("print x = %v.\n", *p)
}

func main() {
  runtime.GOMAXPROCS(1)

  x := Message{}
  go cpuIntensive(&x) // This should go into background
 go printVar(&x)

  // This won't get scheduled until everything has finished.
 time.Sleep(1 * time.Nanosecond) // Wait for goroutines to finish } 
```

低一看，同样的问题。加个`Gosched`就搞定了。

有趣的是，即使您将 GOMAXPROCS 强制为大于 1 的值，它仍然会发生。我想这是因为操场只允许一个等价的 CPU。并且调度是在 cpu 的基础上发生的。

所以，简单来说，对于每个 CPU，Golang 总是表现得像在单线程上运行一样。除非你自己调用`Gosched`,或者调用其他调用，否则 Go 程序结束。无论您指定多少线程或启动多少 Goroutines，它仍然会一次做一件 CPU 密集型的事情。

令人惊讶！