<!--yml

category: 未分类

date: 2024-07-01 18:17:59

-->

# 建议 indent/php.vim : ezyang's 博客

> 来源：[`blog.ezyang.com/2011/02/inden-php-vim/`](http://blog.ezyang.com/2011/02/inden-php-vim/)

## 建议 indent/php.vim

收件人：[John Wellesz](http://www.2072productions.com/)

首先，我要感谢您编写了 php.vim 缩进插件。最近使用了一些其他缩进插件的经历使我意识到没有好的缩进插件编辑会很烦人，多年来 php.vim 大部分时间都为我服务良好。

但是，我对`PHP_autoformatcomment`的默认行为有一个建议。当此选项启用（默认情况下启用），它设置了'w'格式选项，根据尾随换行符进行段落格式化。不幸的是，此选项可能会产生许多不利影响，除非您特别留意尾随换行符的情况，否则可能并不明显：

+   当您输入注释并自动换行时，Vim 会留下单个尾随空格，以示“这不是段落的结尾！”

+   如果您选择几个相邻的注释，例如：

    ```
    // Do this, but if you do that then
    // be sure to frob the wibble

    ```

    然后输入 'gq'，期望重新换行，但什么也不会发生。这是因为这些行缺少尾随空格，因此 Vim 认为它们是单独的句子。

我还认为缩进插件应该无条件地设置 'comments' 选项，因为您加载了 'html' 插件，这会覆盖任何预先存在的值（例如由 .vim/indent/php.vim 文件指定的值）。

请告诉我您对这些更改的看法。我还查看了 Vim 默认提供的所有其他缩进脚本，并注意到它们都不编辑 formatoptions。
