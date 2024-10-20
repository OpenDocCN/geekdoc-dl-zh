# 如何在 Python 3 中加载 Python 2 PyTorch 检查点

> 原文：<https://www.dlology.com/blog/how-to-load-python-2-pytorch-checkpoint-in-python-3-1/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 2 个月前

([评论](/blog/how-to-load-python-2-pytorch-checkpoint-in-python-3-1/#disqus_thread))

![checkpoint](img/19fd666c8ff0ddc92b6837e97cf42a07.png)

本教程展示了一个快速转换 Python 2 中训练的 PyTorch 检查点文件的方法。x 转换成 Python 3.x 兼容格式。当您尝试调用 `torch.load()` 时，它会解决类似这样的错误信息。

```py
UnicodeDecodeError: 'ascii' codec can't decode byte 0x8c in position 16: ordinal not in range(128)
```

## 第一步

加载并保存 Python 2.X 中的 `state_dict` 中的

在下面的例子中，我们使用 Kaggle 数据科学碗 2017 年获奖者模型进行演示，该模型可在[https://github.com/lfz/DSB2017/tree/master/model](https://github.com/lfz/DSB2017/tree/master/model)找到。

## 第二步

`load_state_dict`在 Python 3.X 中

`casenet`是`torch.nn.``Module`的子类实例。

### 可选步骤 3

或者，您可以将整个检查点文件转换为 Python 3。x 兼容。

1.从 Python 2 加载并清理检查点文件。x 转换为二进制格式。

2.在 Python 3 中加载 pickled 检查点。X

3.迭代解码和转换所有二进制字典键。

这里有一个完整的例子来展示它是如何做的。

下面是 Google Colab 上的 Python 2.x [笔记本](https://colab.research.google.com/drive/1HTLXRWFAdJToz8_T11gbMDNfHdnj5jaG)供你参考。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-load-python-2-pytorch-checkpoint-in-python-3-1/&text=How%20to%20load%20Python%202%20PyTorch%20checkpoint%20in%20Python%203) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-load-python-2-pytorch-checkpoint-in-python-3-1/)

*   [←如何在 TensorFlow 中运行 GPU 加速信号处理](/blog/how-to-run-gpu-accelerated-signal-processing-in-tensorflow/)
*   [如何将训练好的 Keras 模型转换成单个 TensorFlow。pb 文件并进行预测→](/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/)