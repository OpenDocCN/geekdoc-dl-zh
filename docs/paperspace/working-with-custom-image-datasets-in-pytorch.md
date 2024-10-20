# 在 PyTorch 中使用自定义影像数据集

> 原文：<https://blog.paperspace.com/working-with-custom-image-datasets-in-pytorch/>

许多初学者在尝试使用 PyTorch 定制、管理的数据集时可能会遇到一些困难。之前已经探讨了如何[管理定制图像数据集(通过网络抓取)](https://blog.paperspace.com/building-simple-web-scrapers-for-image-data-collection/)，本文将作为如何加载和标记定制数据集以与 PyTorch 一起使用的指南。

### 创建自定义数据集

本节借用了关于管理数据集的文章中的代码。这里的目标是为一个模型策划一个自定义数据集，该数据集将区分男士运动鞋/运动鞋和男士靴子。

为了简洁起见，我不会详细介绍代码做了什么，而是提供一个快速的总结，因为我相信您一定已经阅读了前一篇文章。如果你没有，不要担心:它再次链接到这里的。您也可以简单地运行代码块，为下一部分做好准备。

```py
#  article dependencies
import cv2
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
import time
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from tqdm import tqdm
```

#### WebScraper 类

下面的类包含的方法将帮助我们通过使用 beautifulsoup 库解析 html，使用感兴趣的标签和属性提取图像 src 链接，最后从网页下载/抓取感兴趣的图像来管理自定义数据集。这些方法被相应地命名。

```py
class WebScraper():
    def __init__(self, headers, tag: str, attribute: dict,
                src_attribute: str, filepath: str, count=0):
      self.headers = headers
      self.tag = tag
      self.attribute = attribute
      self.src_attribute = src_attribute
      self.filepath = filepath
      self.count = count
      self.bs = []
      self.interest = []

    def __str__(self):
      display = f"""      CLASS ATTRIBUTES
      headers: headers used so as to mimic requests coming from web browsers.
      tag: html tags intended for scraping.
      attribute: attributes of the html tags of interest.
      filepath: path ending with filenames to use when scraping images.
      count: numerical suffix to differentiate files in the same folder.
      bs: a list of each page's beautifulsoup elements.
      interest: a list of each page's image links."""
      return display

    def __repr__(self):
      display = f"""      CLASS ATTRIBUTES
      headers: {self.headers}
      tag: {self.tag}
      attribute: {self.attribute}
      filepath: {self.filepath}
      count: {self.count}
      bs: {self.bs}
      interest: {self.interest}"""
      return display

    def parse_html(self, url):
      """
      This method requests the webpage from the server and
      returns a beautifulsoup element
      """
      try:
        request = Request(url, headers=self.headers)
        html = urlopen(request)
        bs = BeautifulSoup(html.read(), 'html.parser')
        self.bs.append(bs)
      except Exception as e:
        print(f'problem with webpage\n{e}')
      pass

    def extract_src(self):
      """
      This method extracts tags of interest from the webpage's
      html
      """
      #  extracting tag of interest
      interest = self.bs[-1].find_all(self.tag, attrs=self.attribute)
      interest = [listing[self.src_attribute] for listing in interest]
      self.interest.append(interest)
      pass

    def scrape_images(self):
      """
      This method grabs images located in the src links and
      saves them as required
      """
      for link in tqdm(self.interest[-1]):
        try:
          with open(f'{self.filepath}_{self.count}.jpg', 'wb') as f:
            response = requests.get(link)
            image = response.content
            f.write(image)
            self.count+=1
            time.sleep(0.4)
        except Exception as e:
          print(f'problem with image\n{e}')
          time.sleep(0.4)
      pass
```

#### 刮擦功能

为了使用我们的 web scraper 遍历多个页面，我们需要将它封装在一个允许它这样做的函数中。下面的函数就是为了达到这个效果而编写的，因为它包含了格式化为 f 字符串的感兴趣的 url，这将允许迭代 url 中包含的页面引用。

```py
def my_scraper(scraper, page_range: list):
    """
    This function wraps around the web scraper class allowing it to scrape
    multiple pages. The argument page_range takes both a list of two elements
    to define a range of pages or a list of one element to define a single page.
    """
    if len(page_range) > 1:
      for i in range(page_range[0], page_range[1] + 1):
        scraper.parse_html(url=f'https://www.jumia.com.ng/mlp-fashion-deals/mens-athletic-shoes/?page={i}#catalog-listing')
        scraper.extract_src()
        scraper.scrape_images()
        print(f'\npage {i} done.')
      print('All Done!')
    else:
      scraper.parse_html(url=f'https://www.jumia.com.ng/mlp-fashion-deals/mens-athletic-shoes/?page={page_range[0]}#catalog-listing')
      scraper.extract_src()
      scraper.scrape_images()
      print('\nAll Done!')
    pass
```

#### 创建目录

由于目标是管理男鞋数据集，我们需要为此创建目录。为了整洁，我们在根目录下创建一个名为 shoes 的父目录，这个目录包含两个名为 athletic 和 boots 的子目录，它们将保存相应的图像。

```py
#  create directories to hold images
os.mkdir('shoes')
os.mkdir('shoes/athletic')
os.mkdir('shoes/boots')
```

#### 抓取图像

首先，我们需要为 web scraper 定义一个合适的标题。标题有助于屏蔽 scraper，因为它模拟了来自实际 web 浏览器的请求。然后，我们可以使用我们定义的头、我们想要从中提取图像的标签(img)、感兴趣的标签的属性(class: img)、保存图像链接的属性(data-src)、以文件名结尾的感兴趣的文件路径以及要包含在文件名中的计数前缀的起点来实例化运动鞋图像的 scraper。然后我们可以将运动刮刀传递给 my_scraper 函数，因为它已经包含了与运动鞋相关的 URL。

```py
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
          'Accept-Encoding': 'none',
          'Accept-Language': 'en-US,en;q=0.8',
          'Connection': 'keep-alive'}
```

```py
#  scrape athletic shoe images
athletic_scraper = WebScraper(headers=headers, tag='img', attribute = {'class':'img'},
                              src_attribute='data-src', filepath='shoes/athletic/atl', count=0)

my_scraper(scraper=athletic_scraper, page_range=[1, 3])
```

为了抓取靴子的图像，复制下面评论中的两个 URL，替换 my_scraper 函数中的当前 URL。从那以后，引导刮擦器以与运动刮擦器相同的方式被实例化，并被提供给 my_scraper 函数，以便刮擦引导映像。

```py
#  replace the urls in the my scraper function with the urls below
#  first url:
#  f'https://www.jumia.com.ng/mlp-fashion-deals/mens-boots/?page={i}#catalog-listing'
#  second url:
#  f'https://www.jumia.com.ng/mlp-fashion-deals/mens-boots/?page={page_range[0]}#catalog-listing'
#  rerun my_scraper function code cell

#  scrape boot images
boot_scraper = WebScraper(headers=headers, tag='img', attribute = {'class':'img'},
                          src_attribute='data-src', filepath='shoes/boots/boot', count=0)

my_scraper(scraper=boot_scraper, page_range=[1, 3])
```

当所有这些代码单元按顺序运行时，应该在当前工作目录中创建一个名为“shoes”的父目录。这个父目录应该包含两个名为“运动”和“靴子”的子目录，它们将保存属于这两个类别的图像。

### 加载和标记图像

现在我们已经有了自定义数据集，我们需要生成其组成图像的数组表示(加载)，标记数组，然后将它们转换为张量，以便在 PyTorch 中使用。存档这将需要我们定义一个类来完成所有这些过程。下面定义的类执行前两个步骤，它将图像读取为灰度，将它们的大小调整为 100 x 100 像素，然后根据需要对它们进行标记(运动鞋= `[1, 0]`，靴子= `[0, 1]`)。*注意:从我的角度来看，我的工作目录是根目录，所以我在下面的 Python 类中相应地定义了文件路径，你应该基于你自己的工作目录定义文件路径。*

```py
#  defining class to load and label data
class LoadShoeData():
    """
    This class loads in data from each directory in numpy array format then saves
    loaded dataset
    """
    def __init__(self):
        self.athletic = 'shoes/athletic'
        self.boots = 'shoes/boots'
        self.labels = {self.athletic: np.eye(2, 2)[0], self.boots: np.eye(2, 2)[1]}
        self.img_size = 100
        self.dataset = []
        self.athletic_count = 0
        self.boots_count = 0

    def create_dataset(self):
        """
        This method reads images as grayscale from directories,
        resizes them and labels them as required.
        """

        #  reading from directory
        for key in self.labels:
          print(key)

          #  looping through all files in the directory
          for img_file in tqdm(os.listdir(key)):
            try:
              #  deriving image path
              path = os.path.join(key, img_file)

              #  reading image
              image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
              image = cv2.resize(image, (self.img_size, self.img_size))

              #  appending image and class label to list
              self.dataset.append([image, self.labels[key]])

              #  incrementing counter
              if key == self.athletic:
                self.athletic_count+=1
              elif key == self.boots:
                self.boots_count+=1

            except Exception as e:
              pass

        #  shuffling array of images
        np.random.shuffle(self.dataset)

        #  printing to screen
        print(f'\nathletic shoe images: {self.athletic_count}')
        print(f'boot images: {self.boots_count}')
        print(f'total: {self.athletic_count + self.boots_count}')
        print('All done!')
        return np.array(self.dataset, dtype='object')
```

```py
#  load data
data = LoadShoeData()

dataset = data.create_dataset()
```

运行上面的代码单元格应该会返回一个包含自定义数据集中所有图像的 NumPy 数组。该数组的每个元素都是一个自己的数组，保存一个图像及其标签。

### 创建 PyTorch 数据集

生成了自定义数据集中所有图像和标签的数组表示之后，就该创建 PyTorch 数据集了。为此，我们需要定义一个从 PyTorch datasets 类继承的类，如下所示。

```py
#  extending Dataset class
class ShoeDataset(Dataset):
    def __init__(self, custom_dataset, transforms=None):
        self.custom_dataset = custom_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.custom_dataset)

    def __getitem__(self, idx):
        #  extracting image from index and scaling
        image = self.custom_dataset[idx][0]
        #  extracting label from index
        label = torch.tensor(self.custom_dataset[idx][1])
        #  applying transforms if transforms are supplied
        if self.transforms:
          image = self.transforms(image)
        return (image, label)
```

基本上，定义了两个重要的方法`__len__()`和`__getitem__()`。`__len__()`方法返回自定义数据集的长度，而`__getitem__()`方法通过索引从自定义数据集中获取图像及其标签，应用转换(如果有)并返回一个元组，然后 PyTorch 可以使用该元组。

```py
#  creating an instance of the dataset class
dataset = ShoeDataset(dataset, transforms=transforms.ToTensor())
```

当上面的代码单元运行时，dataset 对象成为 PyTorch 数据集，现在可以用于构建深度学习模型。

### 结束语

在本文中，我们了解了如何在 PyTorch 中使用自定义数据集。我们通过网络抓取整理了一个自定义数据集，加载并标记了它，并从中创建了一个 PyTorch 数据集。

在本文中，Python 类的知识将被揭示出来。大多数被定义为类的过程也可以用常规函数来完成(除了 PyTorch dataset 类),但是作为一个个人偏好，我选择这样做。在您的编程之旅的这个阶段，通过做最适合您的事情，您可以随意尝试复制这些代码。