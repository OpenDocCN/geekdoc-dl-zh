# PyTorch 101，第 5 部分:理解钩子

> 原文：<https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/>

读者们好。欢迎来到我们的 PyTorch 调试和可视化教程。至少现在，这是我们 PyTorch 系列的最后一部分，从对图形的基本理解开始，一直到本教程。

在本教程中，我们将介绍 PyTorch 钩子，以及如何使用它们来调试我们的反向传递，可视化激活和修改渐变。

在我们开始之前，让我提醒您这是我们 PyTorch 系列的第 5 部分。

1.  [理解图形，自动微分和亲笔签名](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)
2.  [建立你的第一个神经网络](https://blog.paperspace.com/pytorch-101-building-neural-networks/)
3.  [深入 PyTorch](blog.paperspace.com/pytorch-101-advanced/)
4.  [内存管理和使用多个 GPU](blog.paperspace.com/pytorch-memory-multi-gpu-debugging/)
5.  [理解挂钩](blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/)

你可以在 Github repo [这里](https://github.com/Paperspace/PyTorch-101-Tutorial-Series)获得这篇文章(以及其他文章)中的所有代码。

## 了解 PyTorch 挂钩

PyTorch 中的钩子为桌面带来的功能性严重不足。就像超级英雄的医生命运一样。没听说过他？没错。这才是重点。

我如此喜欢钩子的一个原因是它们让你在反向传播过程中做一些事情。钩子就像是许多英雄留在恶棍巢穴中获取所有信息的工具之一。

你可以在`Tensor`或`nn.Module`上*登记*一个钩子。钩子基本上是一个函数，当`forward`或`backward`被调用时执行。

当我说`forward`的时候，我不是指一个`nn.Module`的`forward`。`forward`函数在这里指的是张量的`grad_fn`对象`torch.Autograd.Function`的`forward`函数。你觉得最后一行是胡言乱语吗？我推荐你在 PyTorch 中查看我们关于计算图的文章。如果你只是在偷懒，那么要明白每个张量都有一个`grad_fn`，它是创建张量的`torch.Autograd.Function`对象。例如，如果一个张量是由`tens = tens1 + tens2`创建的，那么它的`grad_fn`就是`AddBackward`。还是说不通？你一定要回去看看这篇文章。

注意，像`nn.Linear`一样的`nn.Module`有多个`forward`调用。它的输出由两个操作创建，(Y = W * X + B)，加法和乘法，因此将有两个`forward`调用。这可能会把事情弄糟，并可能导致多个输出。我们将在本文后面更详细地讨论这一点。

PyTorch 提供了两种类型的挂钩。

1.  向前的钩拳
2.  向后的钩子

前向钩子是在向前传递的过程中执行的，而后向钩子是在调用`backward`函数时执行的。再次提醒你，这些是一个`Autograd.Function`对象的`forward`和`backward`函数。

### 张量挂钩

一个钩子基本上是一个函数，有一个非常具体的签名。当我们说一个钩子被执行时，实际上，我们说的是这个函数被执行。

对于张量来说，后弯的特征是，

```py
hook(grad) -> Tensor or None 
```

张量没有`forward`钩子。

`grad`基本上就是调用 `backward`后张量**的`grad`属性中包含的值。函数不应该修改它的参数。它必须返回`None`或一个张量，该张量将代替`grad`用于进一步的梯度计算。下面我们提供一个例子。**

```py
import torch 
a = torch.ones(5)
a.requires_grad = True

b = 2*a

b.retain_grad()   # Since b is non-leaf and it's grad will be destroyed otherwise.

c = b.mean()

c.backward()

print(a.grad, b.grad)

# Redo the experiment but with a hook that multiplies b's grad by 2\. 
a = torch.ones(5)

a.requires_grad = True

b = 2*a

b.retain_grad()

b.register_hook(lambda x: print(x))  

b.mean().backward() 

print(a.grad, b.grad)
```

如上所述，功能有多种用途。

1.  您可以打印梯度的值用于调试。您也可以记录它们。这对于梯度被释放的非叶变量尤其有用，除非您对它们调用`retain_grad`。做后者可以增加记忆保持力。钩子提供了更简洁的方式来聚集这些值。
2.  您可以在反向过程中修改梯度**。这一点非常重要。虽然您仍然可以访问网络中张量的`grad`变量，但您只能在完成整个**向后传递后才能访问它。例如，让我们考虑一下我们在上面做了什么。我们将`b`的梯度乘以 2，现在后续的梯度计算，如`a`(或任何依赖于`b`梯度的张量)的梯度计算，使用 2 * grad(b)代替 grad(b)。相比之下，如果我们在`backward`的**之后单独更新参数**，我们必须将`b.grad`和`a.grad`(或者实际上，所有依赖于`b`梯度的张量)乘以 2。****

```py
a = torch.ones(5)

a.requires_grad = True
b = 2*a

b.retain_grad()

b.mean().backward() 

print(a.grad, b.grad)

b.grad *= 2

print(a.grad, b.grad)       # a's gradient needs to updated manually 
```

### **钩子为 nn。模块对象**

对于`nn.Module`对象，钩子函数的签名，

```py
hook(module, grad_input, grad_output) -> Tensor or None 
```

对于后弯钩，以及

```py
hook(module, input, output) -> None 
```

向前勾拳。

在我们开始之前，让我澄清一下，我不喜欢在`nn.Module`对象上使用钩子。首先，因为它们迫使我们打破抽象。`nn.Module`应该是一个表示层的模块化对象。然而，一个`hook`被赋予一个`forward`和一个`backward`，其中在一个`nn.Module`对象中可以有任意数量。这就需要我知道模块化对象的内部结构。

例如，一个`nn.Linear`在其执行期间包含两个`forward`调用。乘法和加法(y = w ***** x **+** b)。这就是为什么钩子函数的`input`可以是一个包含两个不同的`forward`调用的输入和`output`前向调用的输出的元组。

`grad_input`是`nn.Module`对象 w.r.t 的输入对损耗的梯度(dL / dx，dL / dw，dL / b)。`grad_output`是`nn.Module`对象的输出相对于梯度的梯度。由于在一个`nn.Module`对象中的多次调用，这些可能会非常不明确。

考虑下面的代码。

```py
import torch 
import torch.nn as nn

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)

  def forward(self, x):
    x = self.relu(self.conv(x))
    return self.fc1(self.flatten(x))

net = myNet()

def hook_fn(m, i, o):
  print(m)
  print("------------Input Grad------------")

  for grad in i:
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in o:  
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")
  print("\n")
net.conv.register_backward_hook(hook_fn)
net.fc1.register_backward_hook(hook_fn)
inp = torch.randn(1,3,8,8)
out = net(inp)

(1 - out.mean()).backward()
```

产生的输出是。

```py
Linear(in_features=160, out_features=5, bias=True)
------------Input Grad------------
torch.Size([5])
torch.Size([5])
------------Output Grad------------
torch.Size([5])

Conv2d(3, 10, kernel_size=(2, 2), stride=(2, 2))
------------Input Grad------------
None found for Gradient
torch.Size([10, 3, 2, 2])
torch.Size([10])
------------Output Grad------------
torch.Size([1, 10, 4, 4])
```

在上面的代码中，我使用了一个钩子来打印`grad_input`和`grad_output`的形状。现在我对这方面的知识可能是有限的，如果你有其他选择，请做评论，但出于对平克·弗洛伊德的爱，我不知道什么`grad_input`应该代表什么？

在`conv2d`中你可以通过形状来猜测。尺寸`[10, 3, 3, 2]`的`grad_input`是重量等级。那个`[10]`可能是`bias`。但是输入特征图的梯度呢？`None`？除此之外，`Conv2d`使用`im2col`或它的同类来展平图像，这样整个图像上的卷积可以通过矩阵计算来完成，而不是循环。那里有电话吗？所以为了得到 x 的梯度，我必须调用它后面的层的`grad_output`？

`linear`令人费解。两个`grad_inputs`都是大小`[5]`但是线性层的权重矩阵不应该是`160 x 5`吗？

对于这样的混乱，我不喜欢用钩子来处理`nn.Modules`。你可以做像 ReLU 这样简单的事情，但是对于复杂的事情呢？不是我喜欢的。

### **正确使用钩子的方法:一个观点**

所以，我完全赞成在张量上使用钩子。通过使用`named_parameters`函数，我已经成功地用 PyTorch 完成了我所有的渐变修改/裁剪需求。`named_parameters`允许我们更多地控制要修补的渐变。这么说吧，我想做两件事。

1.  反向传播时将线性偏差的梯度变为零。
2.  确保没有梯度去 conv 层小于 0。

```py
import torch 
import torch.nn as nn

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)

  def forward(self, x):
    x = self.relu(self.conv(x))
    x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated 
                                                                  #conv outside less than 0

    # print whether there is any negative grad
    x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any())))  
    return self.fc1(self.flatten(x))

net = myNet()

for name, param in net.named_parameters():
  # if the param is from a linear and is a bias
  if "fc" in name and "bias" in name:
    param.register_hook(lambda grad: torch.zeros(grad.shape))

out = net(torch.randn(1,3,8,8)) 

(1 - out).mean().backward()

print("The biases are", net.fc1.bias.grad)     #bias grads are zero 
```

产生的输出是:

```py
Gradients less than zero: False
The biases are tensor([0., 0., 0., 0., 0.])
```

## **用于可视化激活的向前挂钩**

如果你注意到了，`Tensor`没有前向钩子，而`nn.Module`有，当调用`forward`时执行。尽管我已经强调了将钩子附加到 PyTorch 的问题，但我已经看到许多人使用前向钩子通过将特征映射保存到钩子函数外部的 python 变量来保存中间特征映射。类似这样的。

```py
visualisation = {}

inp = torch.randn(1,3,8,8)

def hook_fn(m, i, o):
  visualisation[m] = o 

net = myNet()

for name, layer in net._modules.items():
  layer.register_forward_hook(hook_fn)

out = net(inp) 
```

一般来说，一个`nn.Module`的`output`是最后一个`forward`的输出。但是，不使用钩子也可以安全地复制上述功能。只需将`nn.Module`对象的`forward`函数中的中间输出添加到一个列表中。不过，打印`nn.Sequential`内部模块的中间激活可能会有点问题。为了解决这个问题，我们需要将一个钩子注册到 Sequential 的子模块，而不是注册到`Sequential`本身。

```py
import torch 
import torch.nn as nn

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)
    self.seq = nn.Sequential(nn.Linear(5,3), nn.Linear(3,2))

  def forward(self, x):
    x = self.relu(self.conv(x))
    x = self.fc1(self.flatten(x))
    x = self.seq(x)

net = myNet()
visualisation = {}

def hook_fn(m, i, o):
  visualisation[m] = o 

def get_all_layers(net):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)

get_all_layers(net)

out = net(torch.randn(1,3,8,8))

# Just to check whether we got all layers
visualisation.keys()      #output includes sequential layers
```

最后，您可以将这个张量转换成 numpy 数组并绘制激活图。

## 结论

这就结束了我们对 PyTorch 的讨论，py torch 是可视化和调试回传的一个非常有效的工具。希望这篇文章能帮助你更快地解决你的问题。