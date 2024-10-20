<!--yml

类别：未分类

日期：2024-07-01 18:18:02

-->

# 每个计算机科学家都应该知道的整数数列？：ezyang's 博客

> 来源：[`blog.ezyang.com/2010/11/integer-sequences-every-computer-scientist-should-know/`](http://blog.ezyang.com/2010/11/integer-sequences-every-computer-scientist-should-know/)

[整数数列在线百科全书](http://oeis.org/Seis.html) 是一个非常棒的网站。假设你在解决一个问题，然后得到了以下的整数数列：`0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2...` 然后你自己想：“嗯，这是什么数列？”嗯，只需[输入它](http://oeis.org/search?q=0%2C+1%2C+0%2C+2%2C+0%2C+1%2C+0%2C+3%2C+0%2C+1%2C+0%2C+2&sort=&language=english&go=Search)，答案就出来了：[A007814](http://oeis.org/A007814)，还有各种有趣的小贴士，如构造方式，封闭形式，数学属性等等。甚至像[二的幂](http://oeis.org/A000079)这样简单的数列都有成千上万种不同的解释和生成方式。

这让我想知道：每个计算机科学家都应该知道哪些整数数列？也就是说，他们应该能够看到前几项并想：“哦，我知道这个数列！”然后稍微费点脑筋去记住构造方式、封闭形式或一些关键的属性。例如，几乎所有有基本数学背景的人都会认识数列[1, 2, 3, 4, 5](http://oeis.org/A000027)；[0, 1, 4, 9, 16](http://oeis.org/A000290)；或[1, 1, 2, 3, 5](http://oeis.org/A000045)。我在本文中引用的第一个数列对我来说有特殊意义，因为我在为《Monad.Reader》写作的文章[Adventures in Three Monads](http://blog.ezyang.com/2010/01/adventures-in-three-monads/)中意外地推导出它。也许不那么熟悉的数列可能是[1, 1, 2, 5, 14, 42](http://oeis.org/A000108)或[3, 7, 31, 127, 8191, 131071, 524287, 2147483647](http://oeis.org/A000668)，但它们对计算机科学家来说仍然非常重要。

那么，每个计算机科学家都应该知道哪些整数数列？（或者说，你最喜欢的整数数列是什么？）
