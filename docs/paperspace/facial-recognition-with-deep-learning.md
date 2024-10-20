# 用深度学习理解面部识别

> 原文：<https://blog.paperspace.com/facial-recognition-with-deep-learning/>

在这篇文章中，我们将以一种简单的方式来分解深度学习的面部识别，以便您理解。阅读本文后，您将能够:

*   将面部识别和深度学习联系起来
*   定义并解释面部识别的工作原理
*   用 Python 写一个函数，实际看看人脸识别是如何工作的。

**注意:**本文假设您已经掌握了 python 编码的基本知识

# 介绍

面部识别已经发展成为人类识别技术中最合适和最合理的技术。在所有像签名、手形图、声音识别和语音这样的技术中，面部识别由于其无接触的特性而被优先使用。使我们能够使用面部识别的新应用程序基于深度学习技术。

那样的话，我们就要明白**什么是深度学习了。**你听说过人工智能吗？这不是我们今天的话题，但是当你想到“人工”这个词的时候，**人工智能**是通过计算机应用模拟人类智能。

**深度学习**在**人工智能**下运行。因此，它**T5 是人工智能的一个元素，它模拟人脑的数据分析过程和模式创建，以便做出最终决定。**

## 面部识别的定义。

面部识别是一种通过从图片、视频镜头或实时分析人脸来识别人类的技术。

直到最近，面部识别一直是计算机视觉的一个问题。深度学习技术的引入能够掌握大数据人脸，并分析丰富而复杂的人脸图像，这使得这项新技术变得更容易，使这项新技术变得高效，后来在面部识别方面甚至比人类视觉更好。

## 面部识别是如何工作的

我们将用于面部识别的程序很容易执行。我们的目标是开发一个神经网络，它给出一组代表人脸的数字，这些数字被称为人脸编码。

让我们说，你有同一个人的几个图像，神经网络应该为这两个图像产生相关的输出，以显示它们是同一个人。

或者，当你有两个人的图像时，神经网络应该产生非常独特的结果来显示他们是两个不同的人。

因此，神经网络应该被训练成自动识别人脸，并根据差异和相似性计算数字。

为了简化我们的工作，我们将从[](http://dlib.net/)****访问一个预先构建的训练模型。****

**这个教程应该给我们一个面部识别是如何工作的基本概念。我们将使用以下步骤:**

1.  **使用来自 [dlib](http://dlib.net/) 的训练模型来识别图像中的人脸。**
2.  **为识别的面部测量面部布局。**
3.  **使用步骤 1 和 2 的输出计算人脸编码。**
4.  **比较已识别人脸和未识别人脸的编码。** 

# **现在，让我们开始建设。**

**在开始之前，您需要在同一个项目文件夹中包含以下内容。我为这个项目创建了一个文件夹，命名为人脸识别**

1.  **获取 JPEG 格式的图片。确保所有图像只有一张脸，并且不是集体照。在人脸识别下创建另一个文件夹，并重命名文件夹图片。**
2.  **获取同一个人的多张照片，将它们放在人脸识别下的一个新文件夹中，并将文件夹重命名为 test。图像中相同人物的不同图像将包含在该文件夹中**
3.  **确保你已经安装了 *Python 2.7* 和 *pip* 。 *Python 2.7* 和 *pip* 可以使用 *Anaconda 2* [*这里*](https://docs.continuum.io/anaconda/install/) *(* 一个 Python 发行版)安装。**
4.  **安装完 *Python 2.7* 和 *pip* 之后，使用下面的命令在终端中安装 [**dlib**](http://dlib.net/) 。**

****窗户****

**`pip install --user numpy imageio dlib`**

****Mac、Linux****

**`sudo pip install --user numpy imageio dlib`******

****纸空间梯度****

**运行提供的笔记本中的第一个单元以完成安装。在笔记本创建过程中，您可以通过在高级选项>工作区 URL 字段中放置[这个 Github URL](https://github.com/gradient-ai/face-recognition-dlib) 来创建自己的笔记本，或者通过将[提供的笔记本](https://console.paperspace.com/te72i7f1w/notebook/r8zbc0fz0nrddx7?file=face_recog.ipynb)分支到您自己的渐变团队空间。**

> **注意:无论您的平台是什么，您都可以将 repo 克隆到您的计算机上以快速完成设置。如果您这样设置，请确保仍然运行笔记本中的第一个单元。**

1.  **现在是时候下载面部识别的现有训练模型了。你需要两个模型。第一个模型预测人脸的布局和位置，而第二个模型分析人脸并产生人脸编码。使用以下步骤下载您需要的文件。**

*   **下载 dlib _ face _ recognition _ resnet _ model _ v1 . bz2[这里](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)和 shape _ predictor _ 68 _ face _ landmarks . dat . bz2[这里](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)。(使用 wget 进行渐变)**
*   **下载成功后，您将从 zip 文件夹中提取文件。**
*   **继续将下载的名为 dlib _ face _ recognition _ resnet _ model _ v1 . dat 和 shape _ predictor _ 68 _ face _ landmarks . dat 的文件复制到主项目文件夹中(在我们的例子中，它应该被命名为 face-recognition)。**

**现在我们已经设置好了一切，是时候开始编码了。这总是有趣的部分，对不对？**

 **# 编码！

使用您喜欢的文本编辑器打开您的文件夹(在本教程中，我将使用 VS 代码)。

在文件夹下添加一个新文件，并将其命名为 face.py。在这个 face.py 文件中，我们将编写与两个不同的文件夹 images 和 test 中的人脸相匹配的所有代码。您可以从终端执行这个脚本，或者如果您选择了克隆 repo，也可以在笔记本中执行。克隆 repo 将使您能够访问已经制作好的 face.py 文件，以及样本图像。

**第一步:配置和开始:**在这一部分，我们集成适当的库，并为我们的面部识别建立对象/参数。

```py
import dlib
import imageio
import numpy as np
import os
# Get Face Detector from dlib
# This allows us to detect faces in image
face_detector = dlib.get_frontal_face_detector()
# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# Use a smaller value to avoid hits
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.6
```

**步骤 2:从 JPEG 中获取面部编码:**我们正在构建代码，该代码将接受图像标题并返回图片的面部编码。

```py
# Using the neural network, this code takes an image and returns its ace encodings
def get_face_encodings(path_to_image):
   # Load image using imageio
   image = imageio.imread(path_to_image)
   # Face detection is done with the use of a face detector
   detected_faces = face_detector(image, 1)
   # Get the faces' poses/landmarks
   # The code that calculates face encodings will take this as an argument   # This enables the neural network to provide similar statistics for almost the same people's faces independent of camera angle or face placement in the picture
   shapes_faces = [shape_predictor(image, face) for face in detected_faces]
   # Compile the face encodings for each face found and return
   return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces] 
```

第三步:面对面的比较:我们将编写一段代码，将一个特定的人脸编码与一组相似的人脸编码进行比较。它将给出一个布尔值(真/假)数组，指示是否匹配。

```py
# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
   # Finds the difference between each known face and the given face (that we are comparing)
   # Calculate norm for the differences with each known face
   # Return an array with True/Face values based on whether or not a known face matched with the given face
   # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
   return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)
```

**第四步:寻找匹配:**我们将编写一个函数，它采用一些已知的人脸编码、一个人名列表(对应于获得的人脸编码集合)和一个要匹配的人脸。它将使用 3 中的代码来检索人脸与当前人脸相匹配的人的姓名。

```py
# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
   # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
   matches = compare_face_encodings(known_faces, face)
   # Return the name of the first match
   count = 0
   for match in matches:
       if match:
           return names[count]
       count += 1
   # Return not found if no match found
   return 'Not Found'
```

**步骤 5:获取子文件夹- images 中所有照片的人脸编码**

```py
# Get path to all the known images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
# Sort in alphabetical order
image_filenames = sorted(image_filenames)
# Get full paths to images
paths_to_images = ['images/' + x for x in image_filenames]
# List of face encodings we have
face_encodings = []
# Loop over images to get the encoding one by one
for path_to_image in paths_to_images:
   # Get face encodings from the image
   face_encodings_in_image = get_face_encodings(path_to_image)
   # Make sure there's exactly one face in the image
   if len(face_encodings_in_image) != 1:
       print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
       exit()
   # Append the face encoding found in that image to the list of face encodings we have
   face_encodings.append(get_face_encodings(path_to_image)[0])
```

**步骤 6:识别测试子文件夹中每幅图像中的已识别人脸(一个接一个)**

```py
# Get path to all the test images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))
# Get full paths to test images
paths_to_test_images = ['test/' + x for x in test_filenames]
# Get list of names of people by eliminating the .JPG extension from image filenames
names = [x[:-4] for x in image_filenames]
# Iterate over test images to find match one by one
for path_to_image in paths_to_test_images:
   # Get face encodings from the test image
   face_encodings_in_image = get_face_encodings(path_to_image)
   # Make sure there's exactly one face in the image
   if len(face_encodings_in_image) != 1:
       print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
       exit()
   # Find match for the face encoding found in this test image
   match = find_match(face_encodings, names, face_encodings_in_image[0])
   # Print the path of test image and the corresponding match
   print(path_to_image, match)
```

一旦您完成了步骤 1 到 6，您将能够使用您的终端使用下面的命令运行代码。

```py
cd {PROJECT_FOLDER_PATH}
python face.py
```

您将获得以下输出

```py
('test/1.jpg', 'Sham')
('test/2.jpg', 'Not Found')
('test/3.jpg', 'Traversy')
('test/4.jpg', 'Maura')
('test/5.jpg', 'Mercc')
```

文件名旁边的名字是与指定人脸匹配的人的名字。请记住，这可能并不适用于所有照片。

现在你已经看到了它是如何完成的，下一步你应该试着利用你自己的照片。使用此代码时，照片中人的脸部完全可见，以获得最佳效果。**