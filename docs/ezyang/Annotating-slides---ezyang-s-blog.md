<!--yml

category: 未分类

date: 2024-07-01 18:18:10

-->

# 注释幻灯片：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/09/annotating-slides/`](http://blog.ezyang.com/2010/09/annotating-slides/)

## 注释幻灯片

一个小技巧供你参考：在生成了幻灯片堆栈并将其打印为 PDF 后，你可能想要用评论来注释幻灯片。这是一个好主意，有几个原因：

+   如果你的幻灯片构造得内容较少，它们可能会被优化用于展示，但并不适合稍后阅读。（“嗯，这里是个图表，我真希望我知道演讲者在这一点上说了什么。”）

+   编写与幻灯片配套的对话是一种无声地练习你的演示的方式！

但是你如何将幻灯片页面与你的注释交叉排列？利用`enscript`和`pdftk`的强大功能，你可以完全自动地完成这一过程，甚至无需离开终端！以下是具体方法。

1.  创建一个“annotations”文本文件（我们将其称为`annot.txt`）。其中包含了与幻灯片配套的文字评论。首先写出解释你的第一张幻灯片的文字，然后插入一个*换页*（`^L`，你可以在 vim 中按`C-l`（插入模式）或在 emacs 中按`C-q C-l`来实现）。接着写出第二张幻灯片的文字。如此反复。

1.  现在，我们希望将此渲染为一个 PDF 文件，并与幻灯片堆栈具有相同的尺寸。找出你的幻灯片尺寸为多少像素，然后编辑你的`~/.enscriptrc`文件，加入以下行：

    ```
    Media: Slide width height llx lly urx ury

    ```

    其中 ll 表示左下，ur 表示右上：这四个数字表示文字的边界框。这些数字的一个可能组合是：

    ```
    Media: Slide 576 432 18 17 558 415

    ```

    现在我们可以调用 enscript 来生成我们注释的一个尺寸合适的 PostScript 文件，使用`enscript annot.txt -p annot.ps -M Slide -B -f Palatino-Roman14`（如果你愿意，可以选择不同的字体。）

1.  将生成的 PostScript 文件转换为 PDF，使用`ps2pdf annot.ps`。

1.  现在，使用 pdftk，我们将分割我们的注释 PDF 和幻灯片 PDF 成为单独的页面，然后将它们合并成一个 PDF。我们可以使用`burst`来输出页面，并建议命名输出文件以便它们正确地交叉排列：

    ```
    mkdir stage
    pdftk slides.pdf burst output stage/%02da.pdf
    pdftk annot.pdf burst output stage/%02db.pdf

    ```

    然后我们将它们合并回来：

    ```
    pdftk stage/*.pdf cat output annotated-slides.pdf

    ```

这是完整的脚本：

```
#!/bin/sh
set -e
ANNOT="$1"
SLIDES="$2"
OUTPUT="$3"
if [ -z "$3" ]
then
    echo "usage: $0 annot.txt slides.pdf output.pdf"
    exit 1
fi
TMPDIR="$(mktemp -d)"
enscript "$ANNOT" -p "$ANNOT.ps" -M Slide -B -f Palatino-Roman14
ps2pdf "$ANNOT.ps" "$ANNOT.pdf"
pdftk "$SLIDES" burst output "$TMPDIR/%03da.pdf"
pdftk "$ANNOT.pdf" burst output "$TMPDIR/%03db.pdf"
pdftk "$TMPDIR"/*.pdf cat output "$OUTPUT"
rm -Rf "$TMPDIR"

```

不要忘记在你的`.enscriptrc`文件中定义`Slide`，并愉快地进行注释吧！
