# 用遗传算法和 PyGAD 训练 PyTorch 模型

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad>

[PyGAD](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/) 是一个用于求解优化问题的遗传算法 Python 3 库。其中一个问题是训练机器学习算法。

PyGAD 有一个模块叫做 [pygad.kerasga](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/KerasGA) 。它使用遗传算法训练 Keras 模型。2021 年 1 月 3 日，新发布的 [PyGAD 2.10.0](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/) 带来了一个名为 [pygad.torchga](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/TorchGA) 的新模块来训练 PyTorch 模型。它非常容易使用，但有几个棘手的步骤。

因此，在本教程中，我们将探索如何使用 PyGAD 来训练 PyTorch 模型。

让我们开始吧。

## 安装 PyGAD

PyGAD 是一个 Python 3 库，可以在 [PyPI (Python 包索引)](https://web.archive.org/web/20230131180157/https://pypi.org/project/pygad)获得。因此，您可以简单地使用这个 pip 命令来安装它:

```py
pip install pygad>=2.10.0
```

确保你的版本至少是 2.10.0，早期版本不支持 pygad.torchga 模块。

你也可以从[这个链接](https://web.archive.org/web/20230131180157/https://files.pythonhosted.org/packages/3b/28/06a37e94ac31a9fe0945f39e7e05ed2390225e45582ff144125433c2f598/pygad-2.10.0-py3-none-any.whl)下载 PyGAD 2.10.0 的轮子分发文件，用下面的命令安装(确保当前目录设置为带有。whl 文件)。

```py
pip install pygad-2.10.0-py3-none-any.whl
```

安装完 [PyGAD](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/) 之后，就该开始使用 pygad.torchga 模块了。

要了解更多关于 PyGAD 的信息，请阅读它的文档。你也可以通过[这个链接](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html)直接访问 [pygad.torchga 模块](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html)的文档。

## pygad.torchga 模块

PyGAD 2.10.0 允许我们使用遗传算法(GA)训练 PyTorch 模型。训练 PyTorch 模型的问题被公式化为 g a 的优化问题，其中模型中的所有参数(例如，权重和偏差)被表示为单个向量(即，染色体)。

pygad.torchga 模块( **torchga** 是 **Torch 遗传算法**的缩写)帮助我们以 pygad 期望的方式制定 PyTorch 模型训练问题。该模块有 1 个类别和 2 个功能:

1.  TorchGA:为 PyTorch 模型创建解决方案群体(即染色体)的类。每个解/染色体保存一组模型的所有参数。
2.  model_weights_as_vector():一个函数，它接受表示 PyTorch 模型的名为 model 的参数，并将其参数作为向量(即染色体)返回。
3.  model_weights_as_dict():一个接受两个参数的函数。第一个被称为模型，它接受 PyTorch 模型。第二个参数称为 weights_vector，它是代表所有模型参数的向量。该函数返回 PyTorch 模型参数的字典，该字典可以传递给名为 load_state_dict()的 PyTorch 方法来设置模型权重。

pygad.torchga 模块的源代码可以在 [ahmedfgad/TorchGA](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/TorchGA) GitHub 项目中获得。

TorchGA 类的构造函数接受以下两个参数:

1.  型号:PyTorch 型号。
2.  num_solutions:群体中解的数量。每个解决方案都有一组不同的 PyTorch 模型参数。

在 pygad.torchga.TorchGA 类的实例中，每个参数都用作一个属性。这意味着您可以通过使用模型属性来访问模型，如下所示:

```py
torchga = TorchGA(model=---, num_solutions=---)
torchga.model
```

第三个属性称为 population_weights，它是人口中所有解决方案的 2D 列表。请记住，每个解决方案都是包含模型参数的 1D 列表。

下面是一个创建 TorchGA 类实例的例子。模型参数可以分配给任何 PyTorch 模型。传递给 num_solutions 参数的值是 10，这意味着群体中有 10 个解决方案。

```py
import pygad.torchga

torch_ga = pygad.torchga.TorchGA(model=...,
                                 num_solutions=10)

initial_population = torch_ga.population_weights

```

TorchGA 类的构造函数调用一个名为 create_population()的方法，该方法创建并向 PyTorch 模型返回一组解决方案。首先，调用 model_weights_as_vector()函数以向量形式返回模型参数。

该向量用于在群体中创建解。为了使解决方案有所不同，随机值被添加到向量中。

假设模型有 30 个参数，那么 population_weights 数组的形状是 10×30。

现在，让我们回顾一下使用 PyGAD 训练 PyTorch 模型所需的步骤。

## 使用 PyGAD 训练 PyTorch 模型

要使用 PyGAD 训练 PyTorch 模型，我们需要完成以下步骤:

*   分类还是回归？
*   创建 PyTorch 模型
*   创建 pygad.torchga.TorchGA 类的实例
*   准备培训数据
*   决定损失函数
*   建立适应度函数
*   生成回调函数(可选)
*   创建 pygad 的一个实例。GA 级
*   运行遗传算法

我们将详细讨论每个步骤。

### 分类还是回归？

决定 PyTorch 模型所解决的问题类型是分类还是回归是很重要的。这将帮助我们准备:

1.  模型的损失函数(用于构建适应度函数)，
2.  模型输出层中的激活函数，
3.  训练数据。

对于 PyTorch 提供的损失函数，检查[此链接](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/nn.html#loss-functions)。回归问题的损失函数的例子包括平均绝对误差( [nn。L1Loss](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) 和均方误差( [nn)。ms loss](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss))。

对于分类问题，一些例子是二元交叉熵( [nn。BCELoss](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss) )进行二元分类和交叉熵( [nn。多类问题的 CrossEntropyLoss](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) )。

基于问题是分类还是回归，我们可以决定输出层中的激活函数。例如， **softmax** 用于分类，**线性**用于回归。

训练数据也取决于问题类型。如果问题是分类，那么输出来自一组有限的离散值。如果问题是回归，那么输出来自一组无限连续的值。

### 创建 PyTorch 模型

我们将使用 torch.nn 模块来构建 PyTorch 模型，以解决一个简单的回归问题。该模型有 3 层:

1.  一个[线性](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)层作为具有 3 个输入和 2 个输出的输入层，
2.  一个 [ReLU](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) 激活层，
3.  另一个[线性](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)层作为输出层，有 2 个输入和 1 个输出。

如果问题是分类，我们必须添加一个合适的输出层，像 [SoftMax](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 。

最后，该模型被创建为 [torch.nn.Sequential](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 类的一个实例，它接受所有先前按顺序创建的层。

```py
import torch.nn

input_layer = torch.nn.Linear(3, 2)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(2, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

```

关于如何构建 PyTorch 模型，我们就不深入探讨了。更多细节，可以查看 [PyTorch 文档](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/index.html)。

现在，我们将使用 pygad.torchga.TorchGA 类创建 PyTorch 模型参数的初始填充。

### 创建 pygad.torchga.TorchGA 类的实例

使用 TorchGA 类，PyGAD 提供了一个简单的接口来创建 PyTorch 模型的初始解决方案群体。只需创建 pygad.torchga.TorchGA 类的一个实例，就会自动创建一个初始群体。

下面是一个将之前创建的模型传递给 TorchGA 类的构造函数的示例。

```py
import pygad.torchga

torch_ga = pygad.torchga.TorchGA(model=model,
                                num_solutions=10)
```

现在，让我们创建随机训练数据来训练模型。

### **准备培训数据**

基于问题是分类还是回归，我们相应地准备训练数据。

这里有 5 个随机样本，每个样本有 3 个输入和 1 个输出。

```py
import numpy

data_inputs = numpy.array([[0.02, 0.1, 0.15],
                           [0.7, 0.6, 0.8],
                           [1.5, 1.2, 1.7],
                           [3.2, 2.9, 3.1]])

data_outputs = numpy.array([[0.1],
                            [0.6],
                            [1.3],
                            [2.5]])

```

如果我们正在解决像 XOR 这样的二进制分类问题，那么它的数据如下所示，其中有 4 个样本，有 2 个输入和 1 个输出。

```py
import numpy

data_inputs = numpy.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

data_outputs = numpy.array([[1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]])

```

回归和分类问题损失函数的时间。

### 决定损失函数

#### 回归

对于回归问题，损失函数包括:

#### 分类

对于分类问题，损失函数包括:

查看[本页](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/nn.html#loss-functions)了解 PyTorch 中损失函数的更多信息。

下面是一个使用 torch.nn.BCELoss 类计算二进制交叉熵的例子。调用 [detach()](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach) 方法从图中分离张量，以返回其值。查看[这个链接](https://web.archive.org/web/20230131180157/http://www.bnikolic.co.uk/blog/pytorch-detach.html)以获得更多关于 [detach()](https://web.archive.org/web/20230131180157/https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach) 方法的信息。

```py
loss_function = torch.nn.BCELoss()

loss = loss_function(predictions, data_outputs).detach().numpy()
```

然后基于所计算的损失来计算适应度函数。

### 建立适应度函数

遗传算法期望适应度函数是最大化的，其输出越高，结果越好。然而，计算机器学习模型的损失是基于最小化损失函数。损失越低，效果越好。

如果适应度设置为等于损失，那么遗传算法将在使适应度增加的方向上搜索。因此，它将在相反的方向上增加损失。这就是为什么适应度是根据下一条线作为损失的倒数来计算的。

当 loss=0.0 时，添加小值 0.00000001 是为了避免被零除。

```py
fitness_value = (1.0 / (loss + 0.00000001))

```

当使用 PyGAD 训练 PyTorch 模型时，有多个解，并且每个解都是保存模型所有参数的向量。

要构建适应度函数，请遵循以下步骤:

1.  从 1D 向量恢复模型参数。
2.  设置模型参数。
3.  做预测。
4.  计算损失值。
5.  计算适应值。
6.  回归健身值。

接下来，我们将为回归和二元分类问题构建适应度函数。

#### 回归的适应度函数

PyGAD 中的 fitness 函数是作为常规 Python 函数构建的，但是它必须接受两个参数，分别表示:

1.  计算其适应值的解决方案，
2.  总体中解的指数。

传递给适应度函数的解是 1D 向量。这个向量不能直接用于 PyTorch 模型的参数，因为模型需要字典形式的参数。因此，在计算损失之前，我们需要将向量转换为字典。我们可以在 pygad.torchga 模块中使用 model_weights_as_dict()函数，如下所示:

```py
model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                   weights_vector=solution)
```

一旦创建了参数字典，就调用 load_state_dict()方法来使用这个字典中的参数作为模型的当前参数。

```py
model.load_state_dict(model_weights_dict)

```

根据当前参数，模型对训练数据进行预测。

```py
predictions = model(data_inputs)
```

模型的预测被传递给损失函数，以计算解决方案的损失。平均绝对误差被用作损失函数。

```py
loss_function = torch.nn.L1Loss()

solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

```

最后，返回适应值。

```py
loss_function = torch.nn.L1Loss()

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                         weights_vector=solution)

    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

```

#### 二元分类的适合度

这是二元分类问题的适应度函数。使用的损失函数是二元交叉熵。

```py
loss_function = torch.nn.BCELoss()

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                         weights_vector=solution)

    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

```

创建的适应度函数应该分配给 pygad 中的 fitness_func 参数。GA 类的构造函数。

接下来，我们将构建一个在每一代结束时执行的回调函数。

### 生成回调函数(可选)

根据下图所示的 PyGAD 生命周期，有一个回调函数，每生成一次就调用一次。这个函数可以被实现并用来打印一些调试信息，比如每代中的最佳适应值，以及完成的代数。请注意，这一步是可选的，仅用于调试目的。

您需要做的就是实现回调函数，然后在 pygad 的构造函数中将它赋给 on_generation 参数。GA 级。下面是一个回调函数，它接受一个表示 pygad 实例的参数。GA 级。

使用这个实例，返回属性 generations_completed，它保存已完成的代的数量。best_solution()方法也被调用，它返回关于当前代中最佳解决方案的信息。

```py
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

```

下一步是创建 pygad 的实例。GA 类，负责运行遗传算法来训练 PyTorch 模型。

### 创建 pygad 的一个实例。GA 级

pygad 的建造者。GA 类接受许多参数，这些参数可以在[文档](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#init)中找到。下一段代码只使用了其中的一些参数，创建了 pygad 的一个实例。GA 类，并将其保存在 ga_instance 变量中:

*   num_generations:代的数量。
*   num _ parents _ mating:要交配的亲本数量。
*   initial _ population:py torch 模型参数的初始总体。
*   fitness_func:适应函数。
*   on_generation:生成回调函数。

```py
num_generations = 250
num_parents_mating = 5
initial_population = torch_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

```

请注意，在 TorchGA 类的构造函数中，群体中的解的数量先前设置为 10。因此，要交配的亲本数量必须少于 10 个。

在下一节中，我们调用 run()方法来运行遗传算法并训练 PyTorch 模型。

### 运行遗传算法

pygad 的 ga _ 实例。GA 现在可以调用 run()方法来启动遗传算法。

```py
ga_instance.run()

```

在这种方法完成后，我们可以使用遗传算法在最后一代中找到的最佳解决方案进行预测。

pygad 中有一个很有用的方法叫做 plot_result()。GA 类中，它显示了一个将适应值与代数相关联的图形。在 run()方法完成后，这很有用。

```py
ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness")
```

## 有关已定型模型的统计信息

皮加德人。GA 类有一个名为 best_solution()的方法，它返回 3 个输出:

1.  找到最佳解决方案，
2.  最佳解决方案的适应值，
3.  群体中最佳解决方案的索引。

下一段代码调用 best_solution()方法，并输出最佳解决方案的信息。

```py
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

```

最佳解决方案的参数可以转换成字典，输入 PyTorch 模型进行预测。

```py
best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)
model.load_state_dict(best_solution_weights)
predictions = model(data_inputs)
print("Predictions : n", predictions.detach().numpy())
```

接下来的代码计算模型定型后的损失。

```py
abs_error = loss_function(predictions, data_outputs)
print("Absolute Error : ", abs_error.detach().numpy())

```

在介绍了使用 PyGAD 构建和训练 PyTorch 模型的所有步骤之后，接下来我们将查看两个带有完整代码的示例。

## 例子

### 回归

对于使用平均绝对误差作为损失函数的回归问题，这里是完整的代码。

```py
import torch
import torchga
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)
    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

input_layer = torch.nn.Linear(3, 2)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(2, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.L1Loss()

data_inputs = torch.tensor([[0.02, 0.1, 0.15],
                            [0.7, 0.6, 0.8],
                            [1.5, 1.2, 1.7],
                            [3.2, 2.9, 3.1]])

data_outputs = torch.tensor([[0.1],
                             [0.6],
                             [1.3],
                             [2.5]])

num_generations = 250 
num_parents_mating = 5 
initial_population = torch_ga.population_weights 
parent_selection_type = "sss" 
crossover_type = "single_point" 
mutation_type = "random" 
mutation_percent_genes = 10 
keep_parents = -1 

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)
model.load_state_dict(best_solution_weights)
predictions = model(data_inputs)
print("Predictions : n", predictions.detach().numpy())

abs_error = loss_function(predictions, data_outputs)
print("Absolute Error : ", abs_error.detach().numpy())

```

**下图是调用** **plot_result()** **方法的结果。**显示适应值逐代变化。

下面是代码中打印语句的输出。平均汇率为 0.0069。

```py
Fitness value of the best solution = 145.42425295191546
Index of the best solution : 0
Predictions :
Predictions :
[[0.08401088]
 [0.60939324]
 [1.3010881 ]
 [2.5010352 ]]
Absolute Error :  0.006876422

```

### **使用 CNN 分类**

接下来的代码使用 PyTorch 构建了一个卷积神经网络(CNN ),用于对 80 幅图像的数据集进行分类，其中每幅图像的大小为 100x100x3。在这个例子中使用了交叉熵损失，因为有两个以上的类。

可以从以下链接下载培训数据:

1.  [dataset_inputs.npy](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy)
2.  [dataset_outputs.npy](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy)

```py
import torch
import torchga
import pygad
import numpy

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

input_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=7)
relu_layer1 = torch.nn.ReLU()
max_pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)

conv_layer2 = torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3)
relu_layer2 = torch.nn.ReLU()

flatten_layer1 = torch.nn.Flatten()

dense_layer1 = torch.nn.Linear(in_features=768, out_features=15)
relu_layer3 = torch.nn.ReLU()

dense_layer2 = torch.nn.Linear(in_features=15, out_features=4)
output_layer = torch.nn.Softmax(1)

model = torch.nn.Sequential(input_layer,
                            relu_layer1,
                            max_pool1,
                            conv_layer2,
                            relu_layer2,
                            flatten_layer1,
                            dense_layer1,
                            relu_layer3,
                            dense_layer2,
                            output_layer)

torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.CrossEntropyLoss()

data_inputs = torch.from_numpy(numpy.load("dataset_inputs.npy")).float()
data_inputs = data_inputs.reshape((data_inputs.shape[0], data_inputs.shape[3], data_inputs.shape[1], data_inputs.shape[2]))

data_outputs = torch.from_numpy(numpy.load("dataset_outputs.npy")).long()

num_generations = 200 
num_parents_mating = 5 
initial_population = torch_ga.population_weights 
parent_selection_type = "sss" 
crossover_type = "single_point" 
mutation_type = "random" 
mutation_percent_genes = 10 
keep_parents = -1 

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)
model.load_state_dict(best_solution_weights)
predictions = model(data_inputs)

print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

accuracy = torch.sum(torch.max(predictions, axis=1).indices == data_outputs) / len(data_outputs)
print("Accuracy : ", accuracy.detach().numpy())

```

**下图是调用** **plot_result()** **方法的结果。**显示适应值逐代变化。

以下是一些关于已训练模型的信息。

```py
Fitness value of the best solution = 1.3009520689219258
Index of the best solution : 0
Crossentropy :  0.7686678
Accuracy :  0.975

```

## 结论

我们探索了如何使用名为 [PyGAD](https://web.archive.org/web/20230131180157/https://pygad.readthedocs.io/) 的 Python 3 库通过遗传算法训练 PyTorch 模型。

PyGAD 有一个模块 [torchga](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/TorchGA) ，它帮助将训练 PyTorch 模型的问题公式化为遗传算法的优化问题。 [torchga](https://web.archive.org/web/20230131180157/https://github.com/ahmedfgad/TorchGA) 模块创建 PyTorch 模型参数的初始群体，其中每个解决方案为模型保存一组不同的参数。使用 PyGAD，进化群体中的解。

这是一个研究遗传算法的好方法。尝试一下，试验一下，看看会出现什么！