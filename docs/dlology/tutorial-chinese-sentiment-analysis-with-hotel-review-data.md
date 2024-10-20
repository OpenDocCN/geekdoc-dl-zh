# 利用酒店评论数据进行中文情感分析的简易指南

> 原文：<https://www.dlology.com/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/>

###### 发布者:[程维](/blog/author/Chengwei/) 5 年 3 个月前

([评论](/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/#disqus_thread))

###### 本教程使用的源代码和数据集，请查看我的[githubrepo](https://github.com/Tony607/Chinese_sentiment_analysis)。

## 依赖性

Python 3.5、numpy、pickle、keras、tensorlow、[【jiba】](https://github.com/fxsjy/jieba)

## 关于日期

客户酒店评论，包括

2916 条正面评价和 3000 条负面评价

### 绘图可选

皮拉布， scipy

## 与英文数据集相比的关键差异

### 文件编码

有些数据文件包含编码 GB2312 会报错的异常编码字符。解决方案:按字节读取，然后逐行解码为 GB2312，跳过编码异常的行。我们还将任何繁体中文字符转换为简体中文字符。

### 从繁体中文转换为简体中文(繁体转简体)

从下载这两个文件

[朗康夫](https://github.com/skydark/nstools/blob/master/zhtools/langconv.py)。py

[zh_wiki](https://github.com/skydark/nstools/blob/master/zhtools/zh_wiki.py) 。py

下面这两行将把字符串" **line"** 从繁体中文转换成简体中文。

### 记号

使用 [街霸](https://github.com/fxsjy/jieba) 对中文句子进行标记，然后加入标记列表，用空格分隔。

然后我们将字符串输入 Keras Tokenizer，Keras Tokenizer】期望每个句子都有单词记号用空格分隔。

### 中文停用词

首先从文件 **中获取停用词列表** ，然后对照该列表检查每个分词后的中文词

### 结果

Keras 训练 20 个纪元，用 GPU (GTX 1070)耗时 7 分 14 秒

升息:0.9726

#### 尝试一些新的评论

对于 Python Jupyter 笔记本源代码和数据集，查看我的 [github repo](https://github.com/Tony607/Chinese_sentiment_analysis) 。

对于一个更新的单词级英语模型，看看我的另一个博客: [简单的股票情绪分析与 Keras](https://www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/) 的新闻数据。

*   标签:
*   [情感分析](/blog/tag/sentiment-analysis/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/&text=An%20easy%20guide%20to%20Chinese%20Sentiment%20analysis%20with%20hotel%20review%20data) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/)

*   [↓【解决方案】Pyinstaller UnicodeDecodeError 错误:“utf-8”编解码器无法解码字节](/blog/solution-pyinstaller-unicodedecodeerror-utf-8-codec-cant-decode-byte/)
*   [如何使用 Keras 对患者查询进行分类(1 分钟培训)→](/blog/tutorial-medical-triage-with-patient-query/)