# 使用 TensorFlow 对象检测 API 深入足球分析

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/dive-into-football-analytics-with-tensorflow-object-detection-api>

说到足球，看到一个球队如何能够战胜一个更强大的对手赢得比赛是令人惊讶的。有时，观众可以通过观察队员(他们的能力和实力)来预测比赛的比分。建立一个能够跟踪球场上团队球员的自动化机器学习模型，这样我们就可以预测球员的下一步行动，这难道不是很有趣吗？

为了进一步证明这一点，让我们来看看如何直观地应用对象检测/跟踪等计算机视觉技术来监控足球场上的团队球员。

这部作品的笔记本可以在这里找到[。这项工作中实现的所有代码都是在 colab 上完成的。](https://web.archive.org/web/20221201165104/https://github.com/elishatofunmi/Computer-Vision/blob/master/football%20analytics/chealsea-mancity-liverpool%20analytics4.ipynb)

下面是我们将要研究的内容的概要:

1.  数据来源
2.  标签
3.  数据准备(导出为张量流记录)
4.  模型管道
5.  建模、培训(和记录)
6.  模型评估
7.  结果和结论

## 数据来源

为了进行足球分析，需要有算法将从中学习的源数据。在这个项目中，源数据是从这里的[得到的](https://web.archive.org/web/20221201165104/https://www.youtube.com/watch?v=EasWagnJvSI)。这里有一些你需要从比赛中了解的信息。这是一场切尔西(2)对曼城(1)的比赛。这个视频包含了真实足球比赛的精彩部分。切尔西能够在这场比赛中击败曼城，靠的是球员们一些独特的战术。

从视频中提取了 152 幅图像，并使用下面的代码进行了处理。

```py
vidcap = cv2.VideoCapture(ChelseaManCity)
count = 0
def getFrame(sec):
   vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
   hasFrames,image = vidcap.read()
   if hasFrames:
       cv2.imwrite("images/frame"+str(count)+".jpg", image)     
   return hasFrames
sec = 2
frameRate = 2 
count=88
success = getFrame(sec)
while success:
   count = count + 1
   sec = sec + frameRate
   sec = round(sec, 2)
   success = getFrame(sec)

```

生成的图像保存在一个文件夹中。由于目标是能够跟踪足球场上的各种队友(对象)，因此对数据进行标记非常重要，以便算法可以将正确的图像映射到实际目标。

机器学习标记是使算法直观地捕捉数据中呈现的信息的重要步骤。我们总是想把我们的问题作为监督学习问题(让机器从输入数据和目标标签中学习)呈现给一个算法；而不是无监督学习(让机器在输入数据中找出模式，没有目标标签)。

这样做的一个好处是，它减少了太多的计算，增强了学习，也使评估变得容易得多。因为我们正在处理图像数据或视频帧，所以需要利用好的注释工具箱来有效地检测足球场上存在的各种对象。在这个项目中，labelImg 被用作标签工具包。

## 标签

Labellmg 是一个用于图像处理和注释的开源图形标签工具。本地使用它的源代码可以在[这里](https://web.archive.org/web/20221201165104/https://github.com/tzutalin/labelImg)找到。在足球场上，我们确实有以下这些:

1.  来自两个对手球队的球员(在这种情况下，切尔西和曼城)，
2.  裁判们，
3.  守门员们，

从上面可以看出，从视频中提取的 152 个图像中，我们有 4 个不同的类别要标记。我们可以训练并使算法跟踪所有 4 个类别并进行分类。为了简单起见，我决定分成 3 个类，这样我就有了:

1.  切尔西—0 级
2.  曼城—1 级
3.  裁判和守门员——2 级

所以我最后上了 3 节课。这是因为 0 级和 1 级的标签比裁判和守门员的标签多。由于在 152 帧中，裁判和守门员的标签都没有得到充分的代表，我们可以将他们作为一个单独的类别，称为**‘未知’**。

既然已经正确标记了帧中的所有类，我们可以继续使用名为 **efficientDet** 的对象检测架构进行建模。请注意，在我们建模之前，我们需要将这些带标签的数据以及生成的 xml 文件(为每一帧生成的文件，包含图像中每个标签的边界框)转换为 TensorFlow 记录。

## 数据准备(导出为张量流记录)

既然我们已经能够方便地标记数据，我们可以继续将数据导出为 TensorFlow 记录。为此，我们将使用 roboflow.com 平台。你需要做的就是遵循这些步骤:

1.  创建 roboflow.com 帐户后，继续点击创建数据集。
2.  填写您的数据集详细信息，上传数据(图像和。xml 文件)。确保在为计算机视觉建模选择数据类型时选择对象检测。
3.  如果需要，您可以继续添加预处理步骤和增强步骤，最后，
4.  上传后，点击右上角的生成张量流记录。继续选择“生成为代码”，这将为您提供一个链接，供您下载列车数据的张量流记录数据。

### **模型管道**

既然我们已经为训练和测试数据导出了张量流记录；下一步是使用 EfficientDet 建模。在使用 EfficientDet 建模之前，需要满足模型管道的某些先决条件，包括以下内容:

1.  安装 TensorFlow 对象检测 API。
2.  设置对象检测架构
3.  设置配置文件。

### TensorFlow 对象检测 API 的安装

以下信息和步骤演示了如何在 Colab 上进行培训时安装 TensorFlow 2 对象检测 API。首先，使用下面的代码在 GitHub 上克隆 TensorFlow 模型库:

```py
import os
import pathlib

if "models" in pathlib.Path.cwd().parts:
 while "models" in pathlib.Path.cwd().parts:
   os.chdir('..')
elif not pathlib.Path('models').exists():
 !git clone --depth 1 https://github.com/tensorflow/models

```

接下来是使用以下命令安装 TensorFlow 对象检测 API。

```py
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

```

安装了 TensorFlow 对象检测 API 后，下面的代码帮助我们确认已经安装了该 API。

```py
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder

%matplotlib inline

```

既然 API 已经正确安装，我们可以继续训练数据，但是在此之前，让我们使用下面的代码构建模型测试器。model tester 文件有助于确认任何对象检测问题的正确建模所需的库的安装和导入。为此，请尝试实现下面的代码。

```py
!python /content/models/research/object_detection/builders/model_builder_tf2_test.py

```

继续建模流程，在导入提议的架构之前，我们可以使用下面的代码从 roboflow.com 下载 tfrecords 格式的导出数据。

务必注意，在使用[roboflow.com](https://web.archive.org/web/20221201165104/http://roboflow.ai/)平台生成 tfrecords 格式的数据时；数据可以导出为链接，下载到 Colab 中。将导出的链接插入下面的程序并运行，以便将数据下载到 Colab 中。

```py

%cd /content
!curl -L "[insert Link]" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

如果你能走到这一步，这是迄今为止完成的伟大工作。在设置配置文件之前，还有一件事，我们需要指定训练和测试数据的目录(这在设置配置文件时是需要的)。

刚刚从 roboflow.com 下载的训练和测试数据的目录可以在当前目录中找到。您的目录应该如下所示:

```py
train_record_fname = '/content/train/foot.tfrecord'
test_record_fname = '/content/test/foot.tfrecord'
label_map_pbtxt_fname = '/content/train/foot_label_map.pbtxt'

```

恭喜你已经走了这么远，为顺利的训练做好了准备。现在，让我们继续在培训前设置配置文件。

### 设置对象检测架构

对于这个问题，期望的对象检测架构是 EfficientDet。该架构有 4 个变体(D0、D1、D2 和 D3)。以下代码显示了 D0-D3 的模型配置，以及它们各自的模型名称和 base_pipeline_file(配置文件)。

```py
MODELS_CONFIG = {
   'efficientdet-d0': {
       'model_name': 'efficientdet_d0_coco17_tpu-32',
       'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
       'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
       'batch_size': 16
   },
   'efficientdet-d1': {
       'model_name': 'efficientdet_d1_coco17_tpu-32',
       'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
       'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
       'batch_size': 16
   },
   'efficientdet-d2': {
       'model_name': 'efficientdet_d2_coco17_tpu-32',
       'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
       'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
       'batch_size': 16
   },
       'efficientdet-d3': {
       'model_name': 'efficientdet_d3_coco17_tpu-32',
       'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
       'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
       'batch_size': 16
   }
}

```

在本教程中，我们实现了轻量级、最小的艺术级 EfficientDet 模型(D0)。扩大到更高效的模型；你将需要更强的计算能力。对于训练，你可以从 5000 步开始(如果损失函数仍在下降，你可能想增加)。每步的评估次数也设置为 500。这意味着，在 500 步之后执行评估。

下面的代码演示了上面的模型设置。

```py
chosen_model = 'efficientdet-d0'
num_steps = 5000 
num_eval_steps = 500 
model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size'] 

```

完成这些后，让我们继续下载指定架构的预训练权重，如上面的代码所示(D0、D1、D2 和 D3)。下面的代码帮助我们做到这一点:

```py
%mkdir /content/models/research/deploy/
%cd /content/models/research/deploy/
import tarfile
download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
!wget {download_tar}
tar = tarfile.open(pretrained_checkpoint)
tar.extractall()
tar.close()

```

让我们继续编写我们的自定义配置文件。

### 设置配置文件

配置文件是表示为. config 的文件扩展名。该文件包含成功训练对象检测模型/架构所需的所有信息。这包括以下参数:

1.  训练的步数。
2.  训练和 label_maps 数据集的目录。
3.  微调检查点。
4.  SSD 模型参数，如锚点 _ 生成器、图像 _ 大小调整器、框 _ 预测器、特征 _ 提取器等。

默认情况下，培训所需的每个期望的体系结构都有一个配置文件。那些配置文件中需要更新的是文件检查点、label_map、列车数据(在 tfrecords 中)和测试数据的目录。

为了实现上述目标，让我们采取以下步骤。首先，让我们使用下面的代码下载定制配置文件。

```py
%cd /content/models/research/deploy
download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + base_pipeline_file
!wget {download_config}

```

完成以上工作后，我们可以继续设置管道文件名和模型检查点目录。下面的代码说明了这一点。此外，您可以使用函数 get_num_classes 确认从 label_map_pbtxt 文件中提取的类的数量，如下所示。

```py
pipeline_fname = '/content/models/research/deploy/' + base_pipeline_file
fine_tune_checkpoint = '/content/models/research/deploy/' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
   from object_detection.utils import label_map_util
   label_map = label_map_util.load_labelmap(pbtxt_fname)
   categories = label_map_util.convert_label_map_to_categories(
       label_map, max_num_classes=90, use_display_name=True)
   category_index = label_map_util.create_category_index(categories)
   return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)

```

对于这个问题，类的数量是 3，即:

1.  切尔西-0 级
2.  人-城市-1 级
3.  未知(裁判、守门员和其他)-2 级

现在，让我们将以下信息写入自定义配置文件:

1.  列车方向，
2.  测试方向。
3.  标签图。
4.  检查点文件目录。

下面的代码帮助我们读取配置文件并将文件目录写入文件。

```py
import re
%cd /content/models/research/deploy
print('writing custom configuration file')
with open(pipeline_fname) as f:
   s = f.read()
with open('pipeline_file.config', 'w') as f:

   s = re.sub('fine_tune_checkpoint: ".*?"',
              'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

   s = re.sub(
       '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
   s = re.sub(
       '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

   s = re.sub(
       'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

   s = re.sub('batch_size: [0-9]+',
              'batch_size: {}'.format(batch_size), s)

   s = re.sub('num_steps: [0-9]+',
              'num_steps: {}'.format(num_steps), s)

   s = re.sub('num_classes: [0-9]+',
              'num_classes: {}'.format(num_classes), s)

   s = re.sub(
       'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

   f.write(s)

```

您可以通过运行下面的代码来确认 dir 已被写入文件:

```py
%cat /content/models/research/deploy/pipeline_file.config

```

现在我们有了一个配置文件，让我们开始训练吧。但是在训练之前，让我们记下配置文件的目录以及保存所有训练参数的目录。它们应该是这样的:

```py
pipeline_file = '/content/models/research/deploy/pipeline_file.config'
model_dir = '/content/training/'

```

## 建模和培训

在所有条件都相同的情况下，我们可以通过运行 model_main_tf2.py 文件来训练数据。此文件用于运行与 TensorFlow 2 相关的任何对象检测问题。要成功运行，您只需指定以下内容:

1.  管道配置路径。
2.  型号目录。
3.  训练步骤数。
4.  评估步骤数。

下面的代码帮助我们做到这一点。

```py
!python /content/models/research/object_detection/model_main_tf2.py
   --pipeline_config_path={pipeline_file}
   --model_dir={model_dir}
   --alsologtostderr
   --num_train_steps={num_steps}
   --sample_1_of_n_eval_examples=1
   --num_eval_steps={num_eval_steps}
```

这样做了之后，我不得不通过改变训练步数来不断地跑，直到我得到想要的结果。在完成 20000 步训练后，当我确信我的训练不再减少时，我不得不停止。这需要大约 5-6 个小时的培训。

经过这么长时间的训练，这里有一个张量板来说明学习速度。

![football tensorboard training](img/b51600356eb8e5d82e028136c95638f3.png)

让我们通过运行这段代码来导出模型的训练推理图。

```py
%ls '/content/training/'

```

下一步是运行一个转换脚本，导出模型参数作为推理，以便在实时预测需要时重新加载。

```py
import re
import numpy as np
output_directory = '/content/fine_tuned_model'

last_model_path = '/content/training/'
print(last_model_path)
!python /content/models/research/object_detection/exporter_main_v2.py
   --trained_checkpoint_dir {last_model_path}
   --output_directory {output_directory}
   --pipeline_config_path {pipeline_file}

```

### **模型评估**

要评估该模型，需要实施以下步骤:

1.  训练后加载最后一个模型检查点。
2.  读入视频帧。
3.  识别每个帧中的对象(边界框和精确度)。
4.  将视频帧转换为视频。

若要在训练后加载最后一个模型检查点，请运行以下代码:

```py
pipeline_config = pipeline_file

model_dir = '/content/training/ckpt-6'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
     model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(
     model=detection_model)
ckpt.restore(os.path.join('/content/training/ckpt-6'))

def get_model_detection_function(model):
 """Get a tf.function for detection."""

 @tf.function
 def detect_fn(image):
   """Detect objects in image."""

   image, shapes = model.preprocess(image)
   prediction_dict = model.predict(image, shapes)
   detections = model.postprocess(prediction_dict, shapes)

   return detections, prediction_dict, tf.reshape(shapes, [-1])

 return detect_fn

detect_fn = get_model_detection_function(detection_model)

```

下一步是读入视频帧，并将其通过对象检测模型进行边界框识别和正确类别的预测。下面的代码帮助我们方便地做到这一点:

```py
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
   label_map,
   max_num_classes=label_map_util.get_max_label_map_index(label_map),
   use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

import random

TEST_IMAGE_PATHS = glob.glob('/content/test/*.jpg')
image_path = random.choice(TEST_IMAGE_PATHS)
image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(
   np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
     image_np_with_detections,
     detections['detection_boxes'][0].numpy(),
     (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
     detections['detection_scores'][0].numpy(),
     category_index,
     use_normalized_coordinates=True,
     max_boxes_to_draw=200,
     min_score_thresh=.5,
     agnostic_mode=False,
)

```

## 结果和结论

成功完成上述所有过程后，该模型能够识别来自同一支球队的球员，并将球场上的人归类为另一个称为“未知”的类别，即裁判或中场休息时来自支持队的人。下面是一个完整的视频演示对象检测架构的行动视频帧。

[https://web.archive.org/web/20221201165104if_/https://www.youtube.com/embed/NiBL6K3jiJM?feature=oembed](https://web.archive.org/web/20221201165104if_/https://www.youtube.com/embed/NiBL6K3jiJM?feature=oembed)

视频

总之，计算机视觉作为一个深度学习领域，在足球分析中有着可行的应用。可以做的更多事情包括:

1.  球跟踪，
2.  用计算机视觉分析场上球员的情绪，
3.  预测下一个接球的球员的强化学习，等等。

如果这篇文章有助于你详细了解计算机视觉，请与朋友分享。感谢阅读！

## 参考

1.  [roboflow.com](https://web.archive.org/web/20221201165104/http://roboflow.ai/)
2.  [heart beat . fritz . ai/end-to-end-object-detection-using-efficient det-on-raspberry-pi-3-part-2-bb 5133646630](https://web.archive.org/web/20221201165104/https://heartbeat.fritz.ai/end-to-end-object-detection-using-efficientdet-on-raspberry-pi-3-part-2-bb5133646630)
3.  [github.com/tzutalin/labelImg](https://web.archive.org/web/20221201165104/https://github.com/tzutalin/labelImg)
4.  [medium . com/@ iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a 128 a 481](https://web.archive.org/web/20221201165104/https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481)
5.  [github.com/microsoft/VoTT](https://web.archive.org/web/20221201165104/https://github.com/microsoft/VoTT)
6.  [cvat.org](https://web.archive.org/web/20221201165104/https://cvat.org/)