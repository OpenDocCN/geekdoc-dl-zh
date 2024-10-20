# 使用 Swift、Core ML 和 TuriCreate 创建一个人脸识别应用程序

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/create-a-face-recognition-application>

面部识别技术已经出现了一段时间，涉及到越来越多的应用，肯定会彻底改变我们的生活。依赖这些技术的应用程序向最终客户保证了数据隐私和安全性的高度可靠性。尽管最近的一些伦理争议，如 [Clearview AI](https://web.archive.org/web/20220928195528/https://www.technologyreview.com/2021/04/09/1022240/clearview-ai-nypd-emails/) 在很大程度上呼应了公共面部识别的可能威胁，但人们一直渴望学习和理解这项技术的工作原理。

如今，谷歌、脸书或苹果等领先的科技行业提供第三方软件来帮助开发者快速构建和迭代使用这些技术扰乱市场并帮助塑造未来时代的产品。后者的一个明显例子是苹果。最近几个月发布了其视觉 API 的主要更新，这是他们所有与计算机视觉相关的事情的主要框架。

Vision API 包括以下功能:

*   本地人脸检测 API
*   使用 ARkit 进行人脸跟踪
*   文本和条形码识别
*   光栅重合
*   Vision 允许为各种图像任务使用定制的核心 ML 模型

在本文中，我们将通过查看以下内容，尝试对这些技术有更多的了解:

***注*** *:您可以在我的* [*Github repo*](https://web.archive.org/web/20220928195528/https://github.com/aymanehachcham/FaceRecogntion-CoreML) 中找到整个项目的代码

## 苹果核心 ML 框架之旅

**Core ML** 是苹果的机器学习框架，通过充分利用所有模型的统一表示，开发人员可以在设备上部署强大的 ML 模型。

更具体地说， *Core ML* 旨在为设备体验提供优化的性能，允许开发人员从各种 ML 模型中进行选择，他们可以在已经配备了专用神经引擎和 ML 加速器的 Apple 硬件上部署这些模型。

### 如何在设备上进行 ML 部署

在展示 Core ML 3.0 中的新功能之前，我想解释一下将一个训练好的模型从 Pytorch 或 Tensorflow 导出到 Core ML 并最终部署到 IOS 应用程序中的不同步骤。

核心 ML 文档推荐使用一个 python 包来简化从第三方培训库(如 [TensorFlow](https://web.archive.org/web/20220928195528/https://www.tensorflow.org/lite/performance/coreml_delegate) 和 [PyTorch](https://web.archive.org/web/20220928195528/https://pytorch.org/mobile/ios/) )到核心 ML 格式的迁移。

使用 coremltools 软件包，您可以:

*   轻松转换来自第三方库的训练模型的权重和结构
*   优化和超调核心 ML 模型
*   利用 Catalyst 和核心 ML 验证 macOS 转换

很明显，并不是所有的模型都被支持，但是在每一次更新中，他们试图增加对更多神经结构、线性模型、集成算法等的支持。你可以在他们的[官方文档网站](https://web.archive.org/web/20220928195528/https://coremltools.readme.io/docs/what-are-coreml-tools)找到当前支持的库和框架如下:

模型类别

支持的软件包

Supported packages:

Tensorflow 1
Tensorflow 2
Pytorch (1.4.0+)
Keras (2.0.4+)
ONNX (1.6.0)
Caffe

Supported packages:

XGBoost
Sci-kit 学习

广义线性模型

Supported packages:

Sci-kit 学习

Supported packages:

LIBSVM

数据管道(后处理和预处理)

Supported packages:

Sci-kit 学习

使用 Pytorch 的转换示例

### 为了说明如何轻松利用 *coremltools* 并将一个经过训练的 Pytorch 模型转换为 Core ML 格式，我将给出一个简单的实际操作示例，说明如何使用 *TorchScript* 和 [*torch* 转换来自](https://web.archive.org/web/20220928195528/https://pytorch.org/docs/stable/generated/torch.jit.trace.html) [torchvision 库](https://web.archive.org/web/20220928195528/https://pytorch.org/vision/stable/index.html)的 ***MobileNetV2*** 模型。 *jit.trace* 对模型权重进行量化和压缩。

***注意*** *:这个例子的代码可以在 coremltools* [*官方文档*](https://web.archive.org/web/20220928195528/https://coremltools.readme.io/docs/pytorch-conversion) *页面*中找到

模型转换的步骤:

加载 MobileNetV2 的预训练版本，并将其设置为评估模式

1.  使用 torch.jit.trace 模块生成 Torchscript 对象
2.  使用 coremltools 将 TorchScript 对象转换为 Core ML
3.  首先，您需要安装 coremltools python 包:

按照官方文档的建议使用 Anaconda

创建 conda 虚拟环境:

*   激活您的康达虚拟环境:

```py
conda create --name coreml-env python=3.6
```

*   安装 conda-forge 的 coremltools

```py
conda activate coreml-env
```

*   或者使用 pip 和 virtualenv 软件包:

```py
conda install -c conda-forge coremltools 
```

安装 virtualenv 软件包:

*   sudo pip 安装虚拟

激活虚拟环境并安装 coremltools:

```py
virtualenv coreml-env
```

*   加载预训练版本的 MobileNetV2

```py
source coreml-env/bin/activate
pip install -u coremltools
```

#### 使用 torchvision 库导入在 ImageNet 上训练的 MobileNetV2 版本。

将模型设置为评估模式:

```py
import torch
Import torchvision

mobile_net  = torchvision.models.mobilenet_v2(pretrained=True)
```

使用 torch.jit.trace 生成 Torchscript 对象

```py
mobile_net.eval()
```

#### Torch jit 跟踪模块采用与模型通常采用的输入张量维数完全相同的输入示例。跟踪只正确记录那些不依赖于数据的函数和模块(例如，张量中的数据没有条件)，并且没有未跟踪的外部依赖项(例如，执行 I/O 或访问全局变量)。

给出随机数据的轨迹:

从单独的文件下载分类标签:

```py
import torch

input = torch.randn(1, 3, 224, 224)
mobile_net_traced = torch.jit.trace(mobile_net, input)
```

使用 coremltools 将 TorchScript 对象转换为 Core ML 格式

```py
import urllib
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()

class_labels = class_labels[1:] 
assert len(class_labels) == 1000
```

#### 多亏了**统一转换 API** ，转换成核心 ML 格式才成为可能。

***MLModel*** 扩展封装了核心 ML 模型的预测方法、配置和模型描述。正如您所看到的，coremltools 包帮助您将来自各种训练工具的训练模型转换成核心 ML 模型。

```py
import coremltools as ct

model = ct.convert(
    mobile_net_traced,
    inputs=[ct.ImageType(name="traced_input", shape=input.shape)]
    classifier_config = ct.ClassifierConfig(class_labels) 
)

model.save("MobileNetV2.mlmodel")
```

核心 ML 内部 ML 工具

### 从我们到目前为止所讨论的内容中，我们了解了 Core ML 是如何工作的，以及将模型从第三方库转换成 Core ML 格式是多么容易。现在，让我们来看看如何使用苹果内部的人工智能生态系统来构建、训练和部署一个 ML 模型。

苹果在他们的整个 ML 框架中集成的两个主要工具是:

Turi Create

#### 如果您希望快速迭代模型实现，以完成系统推荐、对象检测、图像分割、图像相似性或活动分类等任务，这应该是您的目标。

Turi Create 非常有用的一点是，它已经为每个任务定义了预训练模型，您可以使用自定义数据集对其进行微调。Turi-Create 使您能够使用 python 构建和训练您的模型，然后将其导出到 Core ML，以便在 IOS、macOS、watchOS 和 tvOS 应用程序中使用。

What is incredibly useful with Turi Create is that it already defines pretrained models for each task that you can fine-tune with your custom datasets. Turi-Create enables you to build and train your model using python and then just export it to Core ML for use in IOS, macOS, watchOS, and tvOS apps.

机器学习任务

描述

个性化和定制用户选择

Description:

对图像进行标记和分类

Description:

识别图画和手势

Description:

识别和分类声音

Description:

分类和检测场景中的对象

Description:

风格化的图像和视频

Description:

使用传感器对活动进行检测和分类

Description:

寻找图像之间的相似之处

Description:

预测标签

Description:

预测数值

Description:

以无人监督的方式对相似的数据点进行分组

Description:

分析情感分析

Description:

创建 ML

与 Turi create 不同，Create ML 使用户能够构建和训练他们的 ML 模型，而无需编写太多代码。macOS 上可用的 Create ML 提供了一个图形界面，您可以在其中拖放您的训练数据，并选择您想要训练的模型类型(语音识别、图像分类、对象检测等)。)

#### 核心 ML 3.0 中的新特性

在 2019 年 WWDC 发布会上，苹果发布了几个关于 Core ML 和板上新功能的有趣公告。我会给你一个关于新增强的快速总结，以防你错过。

### 到目前为止，Core ML 3.0 中引入的最令人兴奋的特性是可以直接在设备上训练部署的模型。在此之前，我们只有设备上的推理，这基本上意味着我们在其他机器上训练模型，然后利用训练好的模型在设备上进行预测。

通过设备上的培训，您可以执行**转移学习**或**在线学习**，在那里您可以调整现有的模型，以随着时间的推移提高性能和可持续性。

他们包括新型的神经网络层

他们主要关注中间操作层，如屏蔽、张量操作、控制流和布尔逻辑。

2.  如果你对所有更新都感兴趣，请随时观看 [WWDC 2019](https://web.archive.org/web/20220928195528/https://developer.apple.com/videos/play/wwdc2019/704/) 视频。

人脸识别和 Apple Vision API

苹果的视觉框架旨在提供一个高级 API，包含随时可用的复杂计算机视觉模型。他们在 2019 年发布的最新版本包括令人兴奋的功能和改进，再次展示了设备上的机器学习模型是他们移动武器库中的一个巨大部分，他们肯定非常重视它。

## 用苹果自己的话说:

***视觉*** *是一个新的功能强大且易于使用的框架，通过一致的接口为计算机视觉挑战提供解决方案。了解如何使用视觉来检测面部、计算面部标志、跟踪物体等*。

vision API 可分为三个主要部分:

**1。请求**:当你请求框架分析实际场景，它返回给你任何被发现的物体。它被称为请求**对*进行分析。*** 不同种类的请求由多个 API 类处理:

**vndetectfacerectangles 请求**:人脸检测

**VNDetectBarcodesRequest:** 条形码检测

*   **VNDetectTextRectanglesRequest**:图像内的可见文本区域
*   **VNCoreMLRequest** :请求使用核心 ML 功能进行图像分析
*   **VNClassifyImageRequest** :图像分类请求
*   **VNDetectFaceLandmarksRequest:**请求分析人脸并检测特定的拓扑区域，如鼻子、嘴、嘴唇等。基于用包含计算的面部标志的数据训练的模型
*   **VNTrackObjectRequest:** 视频场景内的实时对象跟踪。
*   **2。请求处理器**:分析并执行您触发的请求。它处理从发送请求到执行请求之间发生的所有相关的中间事务。
*   **VNImageRequestHandler** :处理图像分析的请求

**VNSequenceRequestHandler** :处理实时对象跟踪的请求，例如，他们专注于跟踪制作视频时生成的各种图像序列或帧。

*   **3。观察:**请求返回的结果被包装到观察类中，每个观察类引用相应的请求类型。
*   **VNClassificationObservation:**图像分析产生的分类信息

**VNFaceObservation** :专门针对人脸检测。

*   **VNDetectedObjectObservation**:用于物体检测。
*   **VNCoreMLFeatureValueObservation**:用核心 ML 模型预测图像分析得到的键值信息的集合。
*   **vnhorizonto observation**:确定场景中物体的角度和地平线。
*   **VNImageAlignmentObservation**:检测对齐两幅图像内容所需的变换。
*   **VNPixelBufferObservation** :嵌入式核心 ML 模型处理后的输出图像。
*   用 Turicreate 训练人脸识别模型
*   我们将训练一个图像分类器，利用 **resnet-50** 的 **Turi-reate** 预训练版本来检测和识别我们的正确面部。这个想法是用精选的人类面部数据集执行一些迁移学习，然后将模型导出到 Core ML，用于**设备上部署**。

## 设置

要继续学习，您需要在系统中安装 Python 3.6 和 anaconda。

### 然后会创建一个康达虚拟环境，安装 turicreate 5.0。

创建 conda 虚拟环境:

激活您的 conda 环境:

*   收集和分离培训数据

```py
conda create --name face_recog python=3.6
```

*   为了训练我们的分类器，我们需要一些人脸样本和其他与人脸不对应的事物样本，如动物图像、实物等。最终，我们将需要创建两个数据文件夹，包含我们的脸和其余图像的图像。

```py
conda activate face_recog
```

```py
pip install turicreate==5.03
```

### 为了收集我们面部的图像，我们可以使用手机的前置摄像头给自己拍照。我们可以从 ImageNet 或任何其他类似的提供商那里获得其他图像。

数据扩充

*数据扩充*很有帮助，因为扫描过程中从前置摄像头拍摄的照片可能有不同的光照、曝光、方向、裁剪等，我们希望考虑所有的情况。

![image-collection](img/ac9b46b7f6112bc2e017e0aa2ba7fa53.png)

*Source: Author *

### 为了扩充我们的数据，我们将依赖一个非常有用的 python 包 Augmentor，它完全可以在 Github 上获得。

使用增强器，我们可以应用广泛的随机数据增强，如**旋转**、**缩放**、**剪切**或**裁剪**。我们将创建一个数据处理函数，负责所有的转换。

Augmentor 将生成 1500 个额外的面部数据样本。

模特培训

```py
import Augmentor as augment

def data_processing(root_dir: str):
    data = augment.Pipeline(root_dir)
    data.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    data.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    data.skew(probability=0.5, magnitude=0.5)
    data.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    data.crop_random(probability=0.5, percentage_area=0.9, randomise_percentage_area=True)
    data.sample(1500)
```

我们将在我们的虚拟环境中创建一个简单的 python 脚本，其中我们调用 **turicreate resnet-50** 预训练模型，并用我们收集的相应数据对其进行训练。

### 从培训文件夹加载图像

从文件夹名称创建目标标签:艾曼-面/非艾曼-面

1.  用新数据微调模型
2.  将训练好的模型导出到核心 ML 格式。
3.  模型将开始训练，并在整个过程中显示历元结果。
4.  构建 IOS 应用程序

```py
import turicreate as tc
import os

data = tc.image_analysis.load_images('Training Data', with_path=True)

data['label'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))

model = tc.image_classifier.create(data, target='label', model='resnet-50', max_iterations=100)

model.export_coreml('face_recognition.mlmodel')
```

我们将建立一个小的 IOS 应用程序，检测和识别我的脸在前置摄像头流。该应用程序将触发我的 iphone 的前置摄像头，并使用我们之前训练的 turicreate 模型进行实时人脸识别。

![](img/931a3a1dd0792d936a351ccb6cd66ed9.png)

*Displaying training stats on the terminal*

## 打开 XCode 并创建一个单视图应用程序。应用程序的一般 UX 相当简单，有两个 ViewControllers:

入口点 ViewController 定义了一个极简布局，带有一个自定义按钮来激活前置摄像头

一个 CameraViewController，管理相机流并执行实时推理来识别我的脸。

*   设置布局
*   让我们去掉主要的故事板文件，因为我总是喜欢以编程方式编写所有的应用程序，而完全不依赖于任何 XML。

### 删除主故事板文件，更改 info.plist 文件以删除故事板名称，并编辑 SceneDelegate 文件:

设计入口点 LayoutViewController 的布局，将应用程序的徽标图像放在最上面部分的中心，并将导航到 CameraViewController 的按钮设置在其稍下方。

*   ViewController 布局:

```py
var window: UIWindow?
func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
    guard let windowScene = (scene as? UIWindowScene) else { return }
    window = UIWindow(frame: windowScene.coordinateSpace.bounds)
    window?.windowScene = windowScene

    window?.rootViewController = LayoutViewController()
    window?.makeKeyAndVisible()
}
```

处理人脸识别方法:

![](img/3b886338192e9fcbc4bfad8711e746fb.png)

*Application Mockup, Source: Author*

```py
let logo: UIImageView = {
    let image = UIImageView(image: 
    image.translatesAutoresizingMaskIntoConstraints = false
   return image
}()
```

```py
let faceRecognitionButton: CustomButton = {
        let button = CustomButton()
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: 
        button.setTitle("Object detection", for: .normal)
        let icon = UIImage(systemName: "crop")?.resized(newSize: CGSize(width: 50, height: 50))
        button.addRightImage(image: icon!, offset: 30)
        button.backgroundColor = .systemPurple
        button.layer.borderColor = UIColor.systemPurple.cgColor
        button.layer.shadowOpacity = 0.3
        button.layer.shadowColor = UIColor.systemPurple.cgColor

        return button       
    }()
```

*   人脸识别视图控制器

```py
override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        addButtonsToSubview()
    }

fileprivate func addButtonsToSubview() {
    view.addSubview(logo)
    view.addSubview(faceRecognitionButton)
}

fileprivate func setupView() {    
    logo.centerXAnchor.constraint(equalTo:  self.view.centerXAnchor).isActive = true
    logo.topAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.topAnchor, constant: 20).isActive = true

    faceRecognitionButton.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
    faceRecognitionButton.widthAnchor.constraint(equalToConstant: view.frame.width - 40).isActive = true
    faceRecognitionButton.heightAnchor.constraint(equalToConstant: 60).isActive = true
    faceRecognitionButton.bottomAnchor.constraint(equalTo: openToUploadBtn.topAnchor, constant: -40).isActive = true
}
```

*   该 ViewController 采用实时摄像机预览，并触发模型对摄像机流产生的每一帧进行实时推理。在操作每个视频帧时，我们应该格外小心，因为我们可能会由于实时推断而使可用资源迅速超载，并使应用程序崩溃，从而导致内存泄漏。

```py
@objc func handleFaceRecognition() {

       let controller = FaceRecognitionViewController()

       let navController = UINavigationController(rootViewController: controller)

       self.present(navController, animated: true, completion: nil)
    }
```

### 为了在相机设置过程中保持稳定的每秒帧数，建议将分辨率和视频质量降低到:30 FPS 和 640×480。

实例化模型

```py
var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 1)

    let videoPreview: UIView = {
       let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.videoCapture.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }

    // MARK: - SetUp Camera preview
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .vga640x480) { success in

            if success {
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                self.videoCapture.start()
            }
        }
    }
```

我们需要实例化之前获得的核心 ML 模型( *face_recognition.mlmodel* )并开始进行预测。这个想法是通过输入帧来触发模型。该模型应返回封装边界框的多数组对象。最后的步骤将是预测，解析对象，并在脸部周围画一个方框。

### 实现***VideoCaptureDelegate***启动模型推理。

定义对每一帧执行推理的预测函数。

```py
func initModel() {
    if let faceRecognitionModel = try? VNCoreMLModel(for: face_recognition().model) {
        self.visionModel = visionModel
        request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
        request?.imageCropAndScaleOption = .scaleFill
    } else {
        fatalError("fail to create the model")
    }
}
```

*   最后，在后处理阶段，在每个预测上画一个方框。

```py
extension FaceRecognitionViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // the captured image from camera is contained on pixelBuffer
        if !self.isInferencing, let pixelBuffer = pixelBuffer {
            self.isInferencing = true
            // make predictions
            self.predictFaces(pixelBuffer: pixelBuffer)
        }
    }
}
```

*   扩展 facecognitionviewcontroller { func visionrequestdiddomplete(请求:VNRequest，错误:error？){ if let predictions = request . results as？[VNRecognizedObjectObservation]{ dispatch queue . main . async { self。bounding box view . predicted objects = predictions self . is referencing = false } } else { self . is referencing = false } self . semaphore . signal()} }

```py
extension FaceRecognitionViewController {
    func predictFaces(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }

        self.semaphore.wait()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
```

*   最终输出

结论

## 苹果的 vision API 为希望将 ML 模型集成到他们的应用程序中的移动开发者开辟了新的可能性。整个图书馆的设计非常直观，易于理解。没有必要携带机器学习的重要背景来享受核心 ML 的乐趣，开箱即用的各种工具和功能非常令人鼓舞。

## 苹果通过增加对新架构的支持来不断改进他们的 ML 库，并确保与他们的硬件无缝集成。

您可以随时通过改进数据集或创建自己的网络并使用 coremltools 进行转换来改进这些模型。

参考

艾曼·哈克姆

### Spotbills 的数据科学家|机器学习爱好者。

### **阅读下一篇**

ML 实验跟踪:它是什么，为什么重要，以及如何实施

* * *

10 分钟阅读|作者 Jakub Czakon |年 7 月 14 日更新

## ML Experiment Tracking: What It Is, Why It Matters, and How to Implement It

我来分享一个听了太多次的故事。

*“…我们和我的团队正在开发一个 ML 模型，我们进行了大量的实验，并获得了有希望的结果…*

*…不幸的是，我们无法确切地说出哪种性能最好，因为我们忘记了保存一些模型参数和数据集版本…*

> *…几周后，我们甚至不确定我们实际尝试了什么，我们需要重新运行几乎所有的东西"*
> 
> 不幸的 ML 研究员。
> 
> 事实是，当你开发 ML 模型时，你会进行大量的实验。
> 
> 这些实验可能:

使用不同的模型和模型超参数

使用不同的培训或评估数据，

*   运行不同的代码(包括您想要快速测试的这个小变化)
*   在不同的环境中运行相同的代码(不知道安装的是 PyTorch 还是 Tensorflow 版本)
*   因此，它们可以产生完全不同的评估指标。
*   跟踪所有这些信息会很快变得非常困难。特别是如果你想组织和比较这些实验，并且确信你知道哪个设置产生了最好的结果。

这就是 ML 实验跟踪的用武之地。

Keeping track of all that information can very quickly become really hard. Especially if you want to organize and compare those experiments and feel confident that you know which setup produced the best result.  

This is where ML experiment tracking comes in. 

[Continue reading ->](/web/20220928195528/https://neptune.ai/blog/ml-experiment-tracking)

* * *