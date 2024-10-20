# 使用深度学习的面部识别

> 原文：<https://blog.paperspace.com/facial-recognition-using-deep-learning/>

## 卷积神经网络(CNN)和特征提取

卷积神经网络允许我们从图像中提取广泛的特征。事实证明，我们也可以将这种特征提取的想法用于人脸识别！这就是我们在本教程中要探索的，使用深度 conv 网进行人脸识别。注意:这是人脸识别(即实际说出是谁的脸)，而不仅仅是检测(即识别图片中的人脸)。

> 如果你不知道什么是深度学习(或者什么是神经网络)，请阅读我的帖子[初学者的深度学习](https://medium.com/towards-data-science/intro-to-deep-learning-d5caceedcf85#.p61qh0vz8)。如果你想尝试一个使用卷积神经网络进行图像分类的基础教程，你可以尝试[这个教程](https://medium.com/@sntaus/image-classification-using-deep-learning-hello-world-tutorial-a47d02fd9db1#.52f0684bj)。请记住，本教程假设您有基本的编程经验(最好是 Python)，并且您了解深度学习和神经网络的基本思想。

我们将要使用的人脸识别方法相当简单。这里的关键是让一个深度神经网络产生一串描述一张脸的数字(称为人脸编码)。当您传入同一个人的两个不同图像时，网络应该为这两个图像返回相似的输出(即更接近的数字)，而当您传入两个不同人的图像时，网络应该为这两个图像返回非常不同的输出。这意味着需要训练神经网络来自动识别面部的不同特征，并基于此计算数字。神经网络的输出可以被认为是特定人脸的标识符——如果你传入同一个人的不同图像，神经网络的输出将非常相似/接近，而如果你传入不同人的图像，输出将非常不同。

谢天谢地，我们不必经历训练或建立自己的神经网络的麻烦。我们可以通过我们可以使用的 [dlib](http://dlib.net/) 访问一个经过训练的模型。它确实做了我们需要它做的事情——当我们传入某人的面部图像时，它输出一串数字(面部编码);比较来自不同图像的人脸编码将告诉我们某人的脸是否与我们有图像的任何人相匹配。以下是我们将采取的步骤:

1.  **检测**/识别图像中的人脸(使用人脸检测模型)——为了简单起见，本教程将只使用包含一个人脸/人的图像，而不是更多/更少

2.  预测面部姿态/ **界标**(针对步骤 1 中识别的面部)

3.  使用来自步骤 2 的数据和实际图像，计算面部**编码**(描述面部的数字)

4.  **比较**已知人脸的人脸编码和测试图像中的人脸编码，判断照片中的人是谁

希望你对这将如何工作有一个基本的概念(当然上面的描述是非常简化的)。现在是时候开始建设了！

## 准备图像

首先，创建一个项目文件夹(只是一个保存代码和图片的文件夹)。对我来说，这叫做人脸识别，但你可以随便叫它什么。在该文件夹中，创建另一个名为 images 的文件夹。这是一个文件夹，将保存您要对其运行人脸识别的不同人的图像。从脸书下载一些朋友的图片(每人一张)，将图片重命名为朋友的名字(如 taus.jpg 或 john.jpg)，然后将它们全部存储在您刚刚创建的 images 文件夹中。需要记住的一件重要事情是:请确保所有这些图片中只有一张脸(即它们不能是组图)，并且它们都是 JPEG 格式，文件名以. jpg 结尾。

接下来，在您的项目文件夹(对我来说是 face_recognition 文件夹)中创建另一个文件夹，并将其命名为 test。该文件夹将包含同一个人的**幅不同的**幅图像，这些人的照片存储在 images 文件夹中。再次，确保每张照片只有一个人在里面。在测试文件夹中，您可以随意命名图像文件，并且您可以拥有每个人的多张照片(因为我们将对测试文件夹中的所有照片运行人脸识别)。

## 安装依赖项

这个项目最重要的依赖项是 Python 2.7 和 pip。您可以使用 Anaconda 2(这只是一个预打包了 pip 的 Python 发行版)通过点击[这个链接](https://docs.continuum.io/anaconda/install)来安装这两个版本(如果您还没有安装的话)。*注意:请**确保将 Anaconda 2** 添加到您的**路径**中，并注册为您的系统 Python 2.7(在安装过程中会有提示；只需按“是”或勾选复选框)。*

如果您已经完成了 Anaconda 2 的设置，或者您已经预先在机器上安装了 Python 2.7 和 pip，那么您可以继续安装 [dlib](http://dlib.net/) (我们将使用的机器学习库)和其他依赖项。为此，请在终端(Mac OS 或 Linux)或命令提示符(Windows)中键入以下命令:

```py
pip install --user numpy scipy dlib 
```

如果您是 Mac 或 Linux 用户，并且在使用上述命令时遇到问题，请尝试以下命令:

```py
sudo pip install --user numpy scipy dlib 
```

如果上述过程对您不起作用，您可能需要手动下载、编译和安装 dlib 及其 Python API。为此，你必须读一些关于 http://dlib.net/的书。不幸的是，这超出了这篇博文的范围，因此我不会在这里讨论。

你需要做的最后一件事是下载预先训练好的人脸识别模型。你需要两种型号。一个模型预测一张脸的形状/姿势(基本上给你形状在图像中如何定位的数字)。另一个模型，获取人脸并给你人脸编码(基本上是描述那个特定人的脸的数字)。以下是关于如何下载、提取和准备它们的说明:

1.  从[这个链接](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)下载 dlib _ face _ recognition _ resnet _ model _ v1 . dat . bz2，从[这个链接](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)下载 shape _ predictor _ 68 _ face _ landmarks . dat . bz2

2.  一旦下载了这两个文件，就需要解压它们(它们以 bz2 格式压缩)。在 Windows 上，你可以使用 [Easy 7-zip](http://www.e7z.org/) 来实现。在 Mac 或 Linux 上，您应该能够双击文件并提取它们。如果这不起作用，只需在您的终端中为这两个文件键入以下内容:bzip2 **{PATH_TO_FILE}** -解压缩(将{PATH_TO_FILE}替换为您试图提取的文件的实际路径；对我来说，命令应该是 bzip2 ~/Downloads/dlib _ face _ recognition _ resnet _ model _ v1 . dat . bz2-decompress 和 bzip2 ~/Downloads/shape _ predictor _ 68 _ face _ landmarks . dat . bz2-decompress。

3.  提取之后，您应该有两个名为 dlib _ face _ recognition _ resnet _ model _ v1 . dat 和 shape _ predictor _ 68 _ face _ landmarks . dat 的文件。将这两个文件复制到您的项目文件夹中(对我来说，这是我为此项目创建的 *face_recognition* 文件夹)。

## 代码！

现在，你已经设置好了一切，在文本编辑器(最好是 [Atom](http://atom.io) 或 [Sublime Text](https://www.sublimetext.com/) )中打开你的项目文件夹(对我来说叫做 _ recognitionfor)。在名为 recognize.py 的文件夹中创建一个新文件。我们将在这里添加代码以匹配您朋友的面孔。注意，这个过程有两个主要部分:首先，在 images 文件夹中加载已知人脸的人脸编码；完成后，从存储在测试文件夹中的人脸/图像中获取人脸编码，并将它们与我们所有已知的人脸一一匹配。我们将逐步完成这一部分。如果您想看到它运行，您可以将该部分中的所有代码一个接一个地复制粘贴到您的文件中(即，按照下面列出的相同顺序合并所有单独的代码部分)。仔细阅读每个代码块中的注释，了解它的作用。

**第 1 部分:初始化和设置**在这里，我们导入所需的库，并设置人脸识别所需的对象/参数。

```py
import dlib
import scipy.misc
import numpy as np
import os

# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()

# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.6 
```

**第 2 部分:从图像中获取人脸编码**这里我们编写一个函数，它将获取一个图像文件名，并为我们提供该图像的人脸编码。

```py
# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    # Load image using scipy
    image = scipy.misc.imread(path_to_image)

    # Detect faces using the face detector
    detected_faces = face_detector(image, 1)

    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]

    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces] 
```

**第 3a 部分:比较人脸**我们在这里编写一个函数，它将给定的人脸编码与一系列已知的人脸编码进行比较。它将返回一个布尔值(真/假)数组，指示是否存在匹配。

```py
# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE) 
```

**第 3b 部分:查找匹配**我们在这里编写一个函数，它将接受一个已知人脸编码列表、一个人名列表(对应于已知人脸编码列表)和一个要查找匹配的人脸。它将调用 3a 中的函数，并返回与给定人脸匹配的人的姓名。

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

至此，我们拥有了运行程序所需的函数。是时候编写应用程序的最后一部分了(我将把它分成两个独立的部分)。

**部分 4a:获取*图像*文件夹**中所有人脸的人脸编码

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

**部分 4b:将*测试*文件夹中的每张图像与已知人脸进行匹配(逐个)**

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

就是这样！一旦您将第 1 部分到第 4b 部分的所有代码复制粘贴到 recognize.py 文件中(一个接一个——按照我编写它们的顺序),您应该能够使用您的终端(Mac OS 或 Linux)或命令提示符(Windows)通过键入这些命令来运行它(用您的项目文件夹的完整路径替换 **{PROJECT_FOLDER_PATH}** ;对我来说是**/用户/陶斯/人脸识别**):

```py
cd **{PROJECT_FOLDER_PATH}
**python recognize.py 
```

这将为您提供类似如下的输出:

```py
('test/1.jpg', 'Motasim')
('test/2.jpg', 'Not Found')
('test/3.jpg', 'Taus')
('test/4.jpg', 'Sania')
('test/5.jpg', 'Mubin') 
```

文件名旁边的名称显示与给定人脸匹配的人的姓名。请注意，这可能并不适用于所有图像。为了获得此代码的最佳性能，请尝试使用人脸清晰可见的图像。当然，还有其他方法可以使它准确(比如通过实际修改我们的代码来检查多个图像或者使用抖动，等等)。)但这只是为了让你对人脸识别的工作原理有一个基本的了解。

这篇文章的灵感来自于亚当·盖特基(Adam Geitgey)的博客文章(T3)和(T4)关于人脸识别的 Github 报告(T5)。此外，我们正在使用 dlib 网站上提供的一些预先训练好的模型——所以向他们致敬，让他们可以公开访问。我的主要目标是介绍和解释人脸识别的基本深度学习解决方案。当然，有更简单的方法来做同样的事情，但我认为我应该使用 [dlib](http://dlib.net/) 来一部分一部分地(详细地)做这件事，这样你就能真正理解不同的运动部件。还有其他运行人脸识别的方法(非深度学习)，请随意研究它们。这种方法很酷的一点是，你可以用每个人/对象的一两张图像来运行它(假设该模型在实际区分两张脸方面做得很好)。

不管怎样，我希望你喜欢这篇文章。请随意发表评论。