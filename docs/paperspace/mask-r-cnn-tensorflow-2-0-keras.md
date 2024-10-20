# 基于 TensorFlow 2.0 和 Keras 的掩模 R-CNN 目标检测

> 原文：<https://blog.paperspace.com/mask-r-cnn-tensorflow-2-0-keras/>

在[之前的一个教程](https://blog.paperspace.com/mask-r-cnn-in-tensorflow-2-0/)中，我们看到了如何在 Keras 和 TensorFlow 1.14 中使用开源 GitHub 项目 [Mask_RCNN](https://github.com/matterport/Mask_RCNN) 。在本教程中，将对项目进行检查，以将 TensorFlow 1.14 的功能替换为与 TensorFlow 2.0 兼容的功能。

具体来说，我们将涵盖:

*   使用 TensorFlow 2.0 通过掩膜 R-CNN 进行四次编辑以进行预测
*   使用 TensorFlow 2.0 对训练掩模 R-CNN 进行五次编辑
*   要进行的所有更改的摘要
*   结论

开始之前，查看[之前的教程](https://blog.paperspace.com/mask-r-cnn-in-tensorflow-2-0/)下载并运行 [Mask_RCNN](https://github.com/matterport/Mask_RCNN) 项目。

## **使用 TensorFlow 2.0 进行编辑以使用掩膜 R-CNN 进行预测**

[Mask_RCNN](https://github.com/matterport/Mask_RCNN) 项目仅适用于 TensorFlow $\geq$ 1.13。由于 TensorFlow 2.0 提供了更多的功能和增强，开发人员希望迁移到 TensorFlow 2.0。

一些工具可能有助于自动将 TensorFlow 1.0 代码转换为 TensorFlow 2.0，但它们不能保证生成功能完整的代码。查看 Google 提供的[升级脚本](https://www.tensorflow.org/guide/upgrade)。

在本节中，将讨论 Mask R-CNN 项目所需的更改，以便它完全支持 TensorFlow 2.0 进行**预测**(即当`mrcnn.model.MaskRCNN`类构造函数中的`mode`参数设置为`inference`)。在后面的部分中，将应用更多的编辑来训练 TensorFlow 2.0 中的 Mask R-CNN 模型(即当`mrcnn.model.MaskRCNN`类构造函数中的`mode`参数设置为`training`)。

如果安装了 TensorFlow 2.0，运行以下代码块来执行推理将引发异常。请考虑从[此链接](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)下载训练过的重量`mask_rcnn_coco.h5`。

```py
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)

model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)

image = cv2.imread("sample2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)

r = r[0]

mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
```

接下来的小节讨论支持 TensorFlow 2.0 和解决所有异常所需的更改。

### 1.**T2`tf.log()`**

运行前面的代码，在`mrcnn.model.log2_graph()`函数中的这一行引发一个异常:

```py
return tf.log(x) / tf.log(2.0)
```

异常文本如下所示。表示 TensorFlow 没有名为`log()`的属性。

```py
...
File "D:\Object Detection\Tutorial\code\mrcnn\model.py", in log2_graph
  return tf.log(x) / tf.log(2.0)

AttributeError: module 'tensorflow' has no attribute 'log'
```

在 TensorFlow $\geq$ 1.0 中，`log()`函数在库的根目录下可用。由于 TensorFlow 2.0 中一些函数的重组，`log()`函数被移到了`tensorflow.math`模块中。所以，与其使用`tf.log()`，不如简单地使用`tf.math.log()`。

要解决这个问题，只需找到`mrcnn.model.log2_graph()`函数。下面是它的代码:

```py
def log2_graph(x):
    """Implementation of Log2\. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)
```

将每个`tf.log`替换为`tf.math.log`。新的功能应该是:

```py
def log2_graph(x):
    """Implementation of Log2\. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)
```

### 2.**T2`tf.sets.set_intersection()`**

再次运行代码后，当在`mrcnn.model.refine_detections_graph()`函数中执行这一行时，会引发另一个异常:

```py
keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
```

例外情况如下。

```py
File "D:\Object Detection\Tutorial\code\mrcnn\model.py", line 720, in refine_detections_graph
  keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),

AttributeError: module 'tensorflow_core._api.v2.sets' has no attribute 'set_intersection'
```

出现此问题是因为 TensorFlow $\geq$ 1.0 中的`set_intersection()`函数在 TensorFlow 2.0 中被重命名为`intersection()`。

要解决这个问题，只需使用`tf.sets.intersection()`而不是`tf.sets.set_intersection()`。新的一行是:

```py
keep = tf.sets.set.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
```

注意`tf.sets.set_intersection()`在另一个位置使用。因此，搜索它的所有出现，并用`tf.sets.intersection()`替换每一个。

### 3.**T2`tf.sparse_tensor_to_dense()`**

以上两个问题解决后，再次运行代码会在`mrcnn.model.refine_detections_graph()`函数的这一行中给出一个异常:

```py
keep = tf.sparse_tensor_to_dense(keep)[0]
```

这是个例外。TensorFlow $\geq$ 1.0 中的函数`sparse_tensor_to_dense()`可通过`tf.sparse`模块(`tf.sparse.to_dense`)访问。

```py
File "D:\Object Detection\Tutorial\code\mrcnn\model.py", in refine_detections_graph
  keep = tf.sparse_tensor_to_dense(keep)[0]

AttributeError: module 'tensorflow' has no attribute 'sparse_tensor_to_dense'
```

要解决这个问题，请将每次出现的`tf.sparse_tensor_to_dense`替换为`tf.sparse.to_dense`。新行应该是:

```py
keep = tf.sparse.to_dense(keep)[0]
```

对所有出现的`tf.sparse_tensor_to_dense`都这样做。

### 4.**T2`tf.to_float()`**

由于`mrcnn.model.load_image_gt()`函数中的这行代码，还引发了另一个异常:

```py
tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis]
```

出现以下异常是因为 TensorFlow $\geq$ 1.0 中的`to_float()`函数在 TensorFlow 2.0 中不存在。

```py
File "D:\Object Detection\Tutorial\code\mrcnn\model.py", in refine_detections_graph
  tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],

AttributeError: module 'tensorflow' has no attribute 'to_float'
```

作为 TensorFlow 2.0 中`to_float()`函数的替代，使用`tf.cast()`函数如下:

```py
tf.cast([value], tf.float32)
```

要修复异常，请用下一行替换上一行:

```py
tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis]
```

## **推断变化汇总**

要在 TensorFlow 2.0 中使用 Mask R-CNN 进行预测，需要对`mrcnn.model`脚本进行 4 处更改:

1.  将`tf.log()`替换为`tf.math.log()`
2.  将`tf.sets.set_intersection()`替换为`tf.sets.intersection()`
3.  将`tf.sparse_tensor_to_dense()`替换为`tf.sparse.to_dense()`
4.  将`tf.to_float()`替换为`tf.cast([value], tf.float32)`

完成所有这些更改后，我们在本文开头看到的代码可以在 TensorFlow 2.0 中成功运行。

## **使用 TensorFlow 2.0 编辑训练掩码 R-CNN**

假设您安装了 TensorFlow 2.0，运行下面的代码块在 Kangaroo 数据集上训练 Mask R-CNN 将引发许多异常。本节检查 TensorFlow 2.0 中对训练掩码 R-CNN 所做的更改。

请考虑从[这个链接](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)下载[袋鼠数据集](https://github.com/experiencor/kangaroo)以及权重`mask_rcnn_coco.h5`。

```py
import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class KangarooDataset(mrcnn.utils.Dataset):

	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "kangaroo")

		images_dir = dataset_dir +img/'
		annotations_dir = dataset_dir + '/annots/'

		for filename in os.listdir(images_dir):
			image_id = filename[:-4]

			if image_id in ['00090']:
				continue

			if is_train and int(image_id) >= 150:
				continue

			if not is_train and int(image_id) < 150:
				continue

			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'

			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	def extract_boxes(self, filename):
		tree = xml.etree.ElementTree.parse(filename)

		root = tree.getroot()

		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)

		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	def load_mask(self, image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8')

		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

class KangarooConfig(mrcnn.config.Config):
    NAME = "kangaroo_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 131

train_set = KangarooDataset()
train_set.load_dataset(dataset_dir='kangaroo', is_train=True)
train_set.prepare()

valid_dataset = KangarooDataset()
valid_dataset.load_dataset(dataset_dir='kangaroo', is_train=False)
valid_dataset.prepare()

kangaroo_config = KangarooConfig()

model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=kangaroo_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set, 
            val_dataset=valid_dataset, 
            learning_rate=kangaroo_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

model_path = 'Kangaro_mask_rcnn.h5'
model.keras_model.save_weights(model_path)
```

### 1.**T2`tf.random_shuffle()`**

运行前面的代码后，当执行`mrcnn.model.detection_targets_graph()`函数中的下一行时，会引发一个异常。

```py
positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
```

下面给出了一个例外，它表明没有函数被命名为`random_shuffle()`。

```py
File "D:\mrcnn\model.py", in detection_targets_graph
  positive_indices = tf.random_shuffle(positive_indices)[:positive_count]

AttributeError: module 'tensorflow' has no attribute 'random_shuffle'
```

由于 TensorFlow 2.0 函数的新组织，TensorFlow 1.0 中的`tf.random_shuffle()`函数被`tf.random`模块中的`shuffle()`方法所取代。因此，`tf.random_shuffle()`应该换成`tf.random.shuffle()`。

前一行应该是:

```py
positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
```

请检查所有出现的`tf.random_shuffle()`并进行必要的更改。

### 2.**T2`tf.log`**

在`mrcnn.utils.box_refinement_graph()`函数的下一行出现了一个异常。

```py
dh = tf.log(gt_height / height)
```

下面给出的异常表明函数`tf.log()`不存在。

```py
File "D:\mrcnn\utils.py", in box_refinement_graph
  dh = tf.log(gt_height / height)

AttributeError: module 'tensorflow' has no attribute 'log'
```

在 TensorFlow 2.0 中，`log()`函数被移到了`math`模块中。因此，`tf.log()`应该换成`tf.math.log()`。前一行应该是:

```py
dh = tf.math.log(gt_height / height)
```

对所有出现的`tf.log()`功能进行更改。

### 3.张量隶属度

在执行`mrccn.model.MaskRCNN`类的`compile()`方法中的下一个`if`语句时，会引发异常。

```py
if layer.output in self.keras_model.losses:
```

下面列出了例外情况。我们来解释一下它的意思。

`layer.output`和`self.keras_model.losses`都是张量。前一行检查了`self.keras_model.losses`张量中`layer.output`张量的成员。隶属度运算的结果是另一个张量，Python 把它作为`bool`类型，这是不可能的。

```py
File "D:\mrcnn\model.py", in compile
  if layer.output in self.keras_model.losses:
	...
OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution. Use Eager execution or decorate this function with @tf.function.
```

根据下面的代码，`if`语句的目的是检查层的损失是否存在于`self.keras_model.losses`张量内。如果不是，那么代码把它附加到`self.keras_model.losses`张量。

```py
...

loss_names = ["rpn_class_loss",  "rpn_bbox_loss",
              "mrcnn_class_loss", "mrcnn_bbox_loss", 
              "mrcnn_mask_loss"]

for name in loss_names:
    layer = self.keras_model.get_layer(name)

    if layer.output in self.keras_model.losses:
        continue

    loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
    self.keras_model.add_loss(loss)

...
```

在`for`循环之外，`self.keras_model.losses`张量为空。因此，它根本没有损失函数。因此，`if`声明可能会被忽略。解决方法是根据下一段代码注释`if`语句。

```py
...

loss_names = ["rpn_class_loss",  "rpn_bbox_loss",
              "mrcnn_class_loss", "mrcnn_bbox_loss", 
              "mrcnn_mask_loss"]

for name in loss_names:
    layer = self.keras_model.get_layer(name)

    # if layer.output in self.keras_model.losses:
    #     continue

    loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
    self.keras_model.add_loss(loss)

...
```

让我们讨论另一个例外。

### 4.**T2`metrics_tensors`**

在执行`mrccn.model.MaskRCNN`类中的`compile()`方法内的下一行后，会引发一个异常。

```py
self.keras_model.metrics_tensors.append(loss)
```

根据下面的错误，`keras_model`属性中没有名为`metrics_tensors`的属性。

```py
File "D:\mrcnn\model.py", in compile
  self.keras_model.metrics_tensors.append(loss)

AttributeError: 'Model' object has no attribute 'metrics_tensors'
```

解决方法是在`compile()`方法的开头加上`metrics_tensors`。

```py
class MaskRCNN():
    ...
    def compile(self, learning_rate, momentum):
        self.keras_model.metrics_tensors = []
        ...
```

下一节讨论最后要做的更改。

### 5.**保存培训日志**

在执行`mrcnn.model.MaskRCNN`类中的`set_log_dir()`方法的下一行后，会引发一个异常。

```py
self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))
```

下面给出了异常，它表明创建目录时出现了问题。

```py
NotFoundError: Failed to create a directory: ./kangaroo_cfg20200918T0338\train\plugins\profile\2020-09-18_03-39-26; No such file or directory
```

可以通过手动指定有效的目录来解决该异常。例如，下一个目录对我的电脑有效。请尝试为您的目录指定一个有效的目录。

```py
self.log_dir = "D:\\Object Detection\\Tutorial\\logs"
```

这是最后要做的更改，以便 Mask_RCNN 项目可以在 TensorFlow 2.0 中训练 Mask R-CNN 模型。之前准备的训练代码现在可以在 TensorFlow 2.0 中执行。

## **tensor flow 2.0 中列车屏蔽 R-CNN 变更汇总**

为了在 TensorFlow 2.0 中使用 [Mask_RCNN](https://github.com/matterport/Mask_RCNN) 项目训练 Mask R-CNN 模型，需要对`mrcnn.model`脚本进行 5 处更改:

1.  将`tf.random_shuffle()`替换为`tf.random.shuffle()`
2.  将`tf.log()`替换为`tf.math.log()`
3.  在`compile()`方法中注释掉一个`if`语句。
4.  在`compile()`方法的开始初始化`metrics_tensors`属性。
5.  给`self.log_dir`属性分配一个有效的目录。

## **结论**

本教程编辑了开源 [Mask_RCNN](https://github.com/matterport/Mask_RCNN) 项目，以便 Mask R-CNN 模型能够使用 TensorFlow 2.0 进行训练和执行推理。

为了在 TensorFlow 2.0 中训练 Mask R-CNN 模型，总共应用了 9 个更改:4 个用于支持进行预测，5 个用于启用训练。