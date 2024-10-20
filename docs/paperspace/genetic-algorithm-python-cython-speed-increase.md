# Cythonizing 遗传算法:快 18 倍

> 原文：<https://blog.paperspace.com/genetic-algorithm-python-cython-speed-increase/>

在前两个教程中，我们看到了对 Cython 的[介绍，这是一种主要为 Python 中使用的变量定义静态数据类型的语言。这提高了 Python 脚本的性能，从而显著提高了速度。例如，当应用于](https://blog.paperspace.com/boosting-python-scripts-cython/) [NumPy 数组](https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/)时，Cython 完成 10 亿个数的求和比 Python 快 1250 倍。

本教程建立在我们之前讨论的基础上，以加速用 Python 实现遗传算法(GA)的项目的执行。基础项目可以在 [GitHub](https://github.com/ahmedfgad/GeneticAlgorithmPython) 上获得。我们将检查代码，并按照前两个教程中讨论的说明进行尽可能多的更改以提高性能，并且与 Python 相比，运行各代所需的时间要少得多。

我们将从下载 GitHub 项目开始。然后，我们将着眼于使遗传算法的每一个部分都变得和谐；适应度函数、交配池、交叉和变异。我们还将看到如何用 C-speed 实现不同的 NumPy 函数，并以完整代码的最终实现和与 Python 的比较来结束这篇文章。

注意，你不需要知道遗传算法来完成本教程；我们将仔细检查它的每一部分，你所需要做的就是将 Python 代码具体化，不管它是遗传算法还是其他什么。如果你想知道更多关于遗传算法如何工作的细节，请看我在 LinkedIn 上的其他帖子(在 [GitHub](https://github.com/ahmedfgad/GeneticAlgorithmPython) 上实现):

1.  [遗传算法优化简介](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)
2.  [遗传算法在 Python 中的实现](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)

让我们开始吧。

# 下载和使用 GitHub 项目

遗传算法的 Python 实现可在 GitHub 页面获得。该项目有两个文件。第一个是 *ga.py* 文件，它实现了遗传算法操作，包括:

*   使用`cal_pop_fitness()`功能计算适应度函数
*   使用`select_mating_pool()`功能的交配池
*   使用`crossover()`功能进行交叉(实现单点交叉)
*   使用`mutation()`函数进行突变(只有一个基因的值被更新)

第二个文件被命名为*Example _ genetic algorithm . py .*我们来看一个优化以下等式的基本示例，其中 **x** 是一个具有 6 个元素的随机输入向量:

```py
y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + 6w*x6
```

*Example _ genetic algorithm . py*脚本准备初始群体并遍历各代。在每一代中，上面列出的 *ga.py* 中的函数都会被调用。

在本教程中，我们将检查 *ga.py* 和*Example _ genetic algorithm . py*脚本的实现，看看我们可以做些什么来减少计算时间。通过运行该项目并删除所有打印语句(这非常耗时)，Python 代码需要大约 1.46 秒来完成 10，000 代(在 Core i7-6500U CPU @ 2.5 GHz 上运行，具有 16 GB DDR3 RAM)。

让我们从 *ga.py* 文件开始。

# ga.py 内部的 Cythonizing 功能

在 *ga.py* 文件里面，第一个函数是`cal_pop_fitness()`。这将计算群体中每个个体的适应值。这是遗传算法的第一步。

## 适应度函数

`cal_pop_fitness()`函数接受两个参数:一个有 6 个值的向量(上式中的 **x1** 到 **x6** )，以及将计算适合度值的人口。种群由个体组成，每个个体的长度为 6(因为有 6 个权重， **w1** 到 **w6** ，用于 6 个输入 **x1** 到 **x6** )。例如，如果有 8 个人，那么保存人口的数组的大小是 8 x 6。换句话说，它是一个 2D 阵列(或矩阵)。

该函数通过对每个人的 6 个权重中的每一个与 6 个方程输入之间的乘积求和来计算每个人的适合度值。然后，该函数将所有个体的适应值作为向量返回。

```py
def cal_pop_fitness(equation_inputs, pop):
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness
```

我们怎样才能使这一点变得清晰呢？根据上一篇教程中关于使用 [Cython 和 NumPy](https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/) 的四个技巧，第一步是在函数中处理 NumPy 数组——这已经是事实了。定义函数后，我们需要做的就是定义参数的数据类型、返回数据类型、函数中定义的局部变量的数据类型(可选地，我们也可以禁用不必要的功能，如边界检查)。以下是进行这些更改后的新函数:

```py
import numpy
cimport numpy
import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=1] 
cal_pop_fitness(numpy.ndarray[numpy.double_t, ndim=1] equation_inputs, numpy.ndarray[numpy.double_t, ndim=2] pop):
    cdef numpy.ndarray[numpy.double_t, ndim=1] fitness

    fitness = numpy.sum(pop*equation_inputs, axis=1)

    return fitness
```

在函数外部，Cython 用于调用几个 decoratorss，这些 decorator 禁用三个特性:回绕(因为我们不再使用负索引)、检查 None 值和边界检查。注意，我们只禁用了边界检查，因为我们确信没有索引会超出边界。

通常，我们可以用三种方式在 Cython 中定义函数:

1.  `def`:定义一个以 Python 速度工作的函数，因此有点慢。`def` 关键字可用于定义 Python 或 Cython 脚本中的函数。同样，使用`def`定义的函数可以在 Cython/Python 脚本内部或外部调用。
2.  `cdef`:这只能在 Cython 脚本中定义，不能从 Python 脚本中调用。它比使用`def`定义的函数运行得更快。
3.  `cpdef`:这给出了`def`和`cdef`的优点。该函数只能在 Cython 脚本中定义，但可以从 Cython 或 Python 脚本中调用。`cpdef`和`cdef`一样快。

因为我们可能会使用 Python 脚本中 Cython 脚本内部定义的所有函数，所以我们将使用`cpdef`关键字来定义所有函数。

正好在 cpdef 之后，函数的返回数据类型被设置为`numpy.ndarray[numpy.double_t, ndim=1]`。这意味着该函数将返回一个类型为`numpy.ndarray`的变量。数组中元素的类型也使用`numpy.double_t`设置为 double。最后，使用 ndim 参数将维数设置为 1，因为会返回一个 1D 数组(向量)。请注意，如果返回类型中定义的维数与实际返回的数据不匹配，将会引发异常。

接下来，定义两个参数的数据类型。它们都是`numpy.ndarray`，元素类型是`double`。第一个参数是一维的，而第二个参数是二维的。

现在函数头已经完全定义好了。在函数内部有一个局部变量，即*适应度*向量。它的定义与第一个函数自变量相同。最后，返回一维数组。

在这一点上,`cal_pop_fitness()`被同步化；它不如 Python 可读，但现在速度更快了。

## 交配池

下一个函数`select_mating_pool()`在 Python 中实现如下:

```py
def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents
```

下面是 Cython 版本。您可以很容易理解 Cython 函数，因为它与 Python 版本没有太大区别。这个函数返回由多个个体组成的交配池。结果，返回的数组是 2D，因此 ndim 在返回数据类型中被设置为 2。函数中有 6 个局部变量，每个变量都是使用 cdef 关键字定义的。请注意，NumPy 数组的切片和索引与 Python 中的一样。遍历数组也使用索引，这是更快的方法。

```py
import numpy
cimport numpy
import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness

    min_val = -999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = min_val
    return parents
```

## 交叉

下一个函数是`crossover()`，下面用 Python 定义。

```py
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
```

Cython 版本如下。注意，`wraparound()` 装饰器被设置为 True，因为这里需要负索引。另请注意，_ size 参数的类型是 tuple，因此您必须同样提供该参数。任何不匹配都会导致错误。

因为`crossover_point`局部变量被定义为一个整数变量，所以我们使用`numpy.uint8()`来加强这一点并防止任何错误。该函数的其余部分与 Python 中的完全相同。请注意，稍后仍有一些更改要做，我们将把一些耗时的操作替换为耗时较少的操作。

```py
import numpy
cimport numpy
import cython

@cython.wraparound(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] crossover(numpy.ndarray[numpy.double_t, ndim=2] parents, tuple offspring_size):
    cdef numpy.ndarray[numpy.double_t, ndim=2] offspring
    offspring = numpy.empty(offspring_size)
    cdef int k, parent1_idx, parent2_idx
    cdef numpy.int_t crossover_point
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
```

## 变化

*ga.py* 文件中的最后一个函数是`mutation()`，用 Python 显示如下:

```py
def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover
```

下面是 cythonized 版本。它遵循我们之前看到的步骤:禁用未使用的特性，使用`cpdef`而不是`def`，声明参数、返回值和局部变量的数据类型。因为不需要负索引，所以该功能禁用负索引。

```py
import numpy
cimport numpy
import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] mutation(numpy.ndarray[numpy.double_t, ndim=2] offspring_crossover, int num_mutations=1):
    cdef int idx, mutation_num, gene_idx
    cdef double random_value
    cdef Py_ssize_t mutations_counter
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover
```

我们已经完成了 cytonization*ga . py*！下面列出了新的完整代码。只需将这段代码保存到一个名为*ga . pyx***的文件中，我们将在*构建中构建它。使用 *setup.py* 文件的 pyx 文件*部分。**

```py
import numpy
cimport numpy
import time
import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=1] cal_pop_fitness(numpy.ndarray[numpy.double_t, ndim=1] equation_inputs, numpy.ndarray[numpy.double_t, ndim=2] pop):
    cdef numpy.ndarray[numpy.double_t, ndim=1] fitness
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness, a

    min_val = -99999999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = min_val
    return parents

@cython.wraparound(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] crossover(numpy.ndarray[numpy.double_t, ndim=2] parents, tuple offspring_size):
    cdef numpy.ndarray[numpy.double_t, ndim=2] offspring
    offspring = numpy.empty(offspring_size)
    cdef int k, parent1_idx, parent2_idx
    cdef numpy.int_t crossover_point
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] mutation(numpy.ndarray[numpy.double_t, ndim=2] offspring_crossover, int num_mutations=1):
    cdef int idx, mutation_num, gene_idx
    cdef double random_value
    cdef Py_ssize_t mutations_counter
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover
```

第二个文件*Example _ genetic algorithm . py*，调用在 *ga.py* 文件中定义的函数。让我们在运行 GA 之前完成第二个文件。

# Cythonizing 示例 _GeneticAlgorithm.py

*Example _ genetic algorithm . py***文件的 Python 实现如下。导入了时间模块，因此我们可以比较 Python 和 Cython 的性能。**

```py
`import numpy
import ga
import time

equation_inputs = [4,-2,3.5,5,-11,-4.7]

num_weights = len(equation_inputs)

sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop,num_weights)
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

best_outputs = []
num_generations = 10000
t1 = time.time()
for generation in range(num_generations):
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)

    best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

    parents = ga.select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
t2 = time.time()
t = t2-t1
print("Total Time %.20f" % t)`
```

**下面列出了 cythonized 代码。 *ga* 模块作为常规 Python 模块导入。你所要做的就是声明所有变量的数据类型。只需注意将传递的变量与之前编辑的函数所接受的类型相匹配。**

```py
`import ga
import numpy
cimport numpy
import time

cdef numpy.ndarray equation_inputs, parents, new_population, fitness, offspring_crossover, offspring_mutation
cdef int num_weights, sol_per_pop, num_parents_mating, num_generations
cdef tuple pop_size
cdef double t1, t2, t

equation_inputs = numpy.array([4,-2,3.5,5,-11,-4.7])
num_weights = equation_inputs.shape[0]

num_weights = equation_inputs.shape[0]
num_parents_mating = 4

sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop, num_weights)
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

num_generations = 10000

t1 = time.time()
for generation in range(num_generations):
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)

    parents = ga.select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
t2 = time.time()
t = t2-t1
print("Total Time %.20f" % t)`
```

**我们只能将`numpy.ndarray`数据类型赋给 NumPy 变量，仅此而已。我们无法指定维度的数量或元素的数据类型，因为 Cython 尚不支持这些功能。然而，如果代码被打包成一个函数，那么我们就可以定义一切并加快处理速度。我们将在以后做这方面的工作。**

**现在，只需将 Cython 代码保存到名为*Example _ genetic algorithm . pyx*的文件中，该文件将与 *ga.pyx* 文件一起构建。**

# **构建。pyx 文件**

**下一步是建造*。pyx* 文件生成*。pyd* **/** *。所以* 文件要导入到项目中。下面列出了用于此目的的 *setup.py* 文件。因为有两个*。pyx* 文件，函数`cythonize()` 没有给出明确的名称，而是要求用*构建所有文件。pyx* 扩展。**

```py
`import distutils.core
import Cython.Build
import numpy

distutils.core.setup(
    ext_modules = Cython.Build.cythonize("*.pyx"),
    include_dirs=[numpy.get_include()]
)`
```

**为了构建文件，从命令行发出下面的命令。**

```py
`python setup.py build_ext --inplace`
```

**命令成功完成后，我们可以使用下面的命令导入*Example _ genetic algorithm . pyx*文件。这将自动运行代码。**

```py
`import Example_GeneticAlgorithm`
```

**Cython 代码完成需要 **0.945** 秒。与 Python 代码的 **1.46** 秒相比；Cython 比 T4 快 1.55 倍(注意，代码是在一台配备酷睿 i7-6500U CPU @ 2.5 GHz 和 16 GB DDR3 RAM 的机器上运行的)。**

**为了进一步减少时间，我们可以做一个简单的编辑:使用一个函数包装*Example _ genetic algorithm . pyx***文件的内容。****

# ****在函数和脚本体中进化代****

****让我们在*Example _ genetic algorithm . pyx*中创建一个名为`optimize()`的函数，并将该文件的内容放入我们的新函数中:****

```py
**`import ga
import numpy
cimport numpy
import time
import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef optimize():
    cdef numpy.ndarray equation_inputs, parents, new_population, fitness, offspring_crossover, offspring_mutation
    cdef int num_weights, sol_per_pop, num_parents_mating, num_generations
    cdef list pop_size
    cdef double t1, t2, t

    equation_inputs = numpy.array([4,-2,3.5,5,-11,-4.7])
    num_weights = equation_inputs.shape[0]

    sol_per_pop = 8
    num_weights = equation_inputs.shape[0]
    num_parents_mating = 4

    pop_size = [sol_per_pop,num_weights]
    #Creating the initial population.
    new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

    num_generations = 1000000
    t1 = time.time()
    for generation in range(num_generations):
        fitness = cal_pop_fitness(equation_inputs, new_population)

        parents = select_mating_pool(new_population, fitness,
                                          num_parents_mating)

        offspring_crossover = crossover(parents,
                                           offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        offspring_mutation = mutation(offspring_crossover, num_mutations=2)

        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    t2 = time.time()
    t = t2-t1
    print("Total Time %.20f" % t)
    print(cal_pop_fitness(equation_inputs, new_population))`**
```

****为了调用`optimize()`函数，只需重新构建 Cython *即可。pyx* 文件，并从命令行发出以下 Python 命令:****

```py
**`import Example_GeneticAlgorithm
Example_GeneticAlgorithm.optimize()`**
```

****现在只需要 0.944 秒(T2)而不是 0.945 秒(T4)；几乎没有任何变化。一个原因是由于调用外部模块 *ga* 用于每个所需的功能。相反，我们将通过在 *ga.pyx* 文件中复制并粘贴`optimize()`函数来保存函数调用。因为这些函数是同一个文件的一部分，所以调用它们的开销较少。****

****因为`optimize()`函数现在是 *ga.pyx* 文件的一部分，我们不再需要*Example _ genetic algorithm . pyx*文件。您可以编辑 *setup.py* 文件，指定只构建 *ga.pyx* 文件。****

****以下命令用于调用**优化()**函数。时间现在是 0.9 秒，而不是 T2 的 0.944 秒，因此 Cython 代码现在比 Python 快了 T4 的 1.62 倍。****

```py
**`import ga
ga.optimize()`**
```

****现在所有的代码都已经被同步化了，但是还可以做更多的工作来提高速度。让我们看看如何使用 C 函数，而不是 Python 函数——这将带来迄今为止最大的速度提升。****

# ****用 C 语言实现 Python 特性****

****Python 使得许多事情对程序员来说更容易，这是它的好处之一。但这在某些情况下会增加时间。在这一节中，我们将检查 Python 中可用但速度较慢的一些函数，并了解如何实现它们以 C 语言速度运行。****

## ****用 C 语言实现 NumPy sum()****

****在 **cal_pop_fitness()** 函数中，使用 **numpy.sum()** 函数计算每个个体与方程输入之间的乘积之和。我们可以根据下面的代码使用 2 for 循环手动实现这个函数。注意，循环以 C 速度运行。由于这个原因，变量**适应度**被声明为 **numpy.ndarray** 类型，并使用 **numpy.zeros()** 初始化为零数组。计算适应值的结果保存在该变量中。****

```py
**`@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef cal_pop_fitness(numpy.ndarray[numpy.double_t, ndim=1] equation_inputs, numpy.ndarray[numpy.double_t, ndim=2] pop):
    cdef numpy.ndarray[numpy.double_t, ndim=1] fitness
    fitness = numpy.zeros(pop.shape[0])
    # fitness = numpy.sum(pop*equation_inputs, axis=1) # slower than looping.
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            fitness[i] += pop[i, j]*equation_inputs[j]
    return fitness`**
```

****编辑完成后，我们可以构建**。pyx** 文件，看看新代码有多快。使用上述功能后的新代码只需要 **0.8** 秒。因此，使用循环实现 **numpy.sum()** 函数节省了 **0.1** 秒( **100** 毫秒)。让我们考虑一下其他需要优化的地方。****

****在**select _ matting _ pool()**函数中，健康数组中最大元素的索引是使用以下代码行返回的。****

```py
**`max_fitness_idx = numpy.where(fitness == numpy.max(fitness))[0][0]`**
```

****我们可以使用下面的循环来编辑函数，以 C 速度实现这一行。通过这样做，执行时间现在是 0.44 秒，而不是 0.8 秒。与 Python 相比，Cython 现在快了 3.32 倍。****

```py
**`@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness, a

    min_val = -99999999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = 0
        # numpy.where(fitness == numpy.max(fitness))
        for a in range(1, fitness.shape[0]):
            if fitness[a] > fitness[max_fitness_idx]:
                max_fitness_idx = a
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = min_val
    return parents`**
```

## ****C 速度下的 NumPy 数组切片****

****切片只是将数组的一部分返回到另一个数组中。我们可以在 Cython 中为下面列出的新函数中的`parents`和`pop`实现这一点。通过这样做，Cython 只需要 0.427 秒，而不是 0.44 秒。****

```py
**`@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness, a

    min_val = -99999999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = 0
        # numpy.where(fitness == numpy.max(fitness))
        for a in range(1, fitness.shape[0]):
            if fitness[a] > fitness[max_fitness_idx]:
                max_fitness_idx = a

        # parents[parent_num, :] = pop[max_fitness_idx, :] # slower han looping by 20 ms
        for a in range(parents.shape[1]):
            parents[parent_num, a] = pop[max_fitness_idx, a]
        fitness[max_fitness_idx] = min_val
    return parents`**
```

****因为切片也在`crossover()` 函数中使用，所以我们可以编辑它，使用以 C 速度运行的循环来实现数组切片。新函数如下，耗时 0.344 秒而不是 0.427 秒。这些变化可能看起来很小，但是当您运行数百或数千行代码时，它们会产生巨大的影响。此时，这个函数的运行速度是 Python 的 4.24 倍。****

****分配给`crossover_point` 变量的值之前已使用`numpy.uint8()`进行了转换。现在，它被转换成 C 风格使用`(int)`。****

```py
**`@cython.wraparound(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] crossover(numpy.ndarray[numpy.double_t, ndim=2] parents, tuple offspring_size):
    cdef numpy.ndarray[numpy.double_t, ndim=2] offspring
    offspring = numpy.empty(offspring_size)
    cdef int k, parent1_idx, parent2_idx
    cdef numpy.int_t crossover_point
    crossover_point = (int) (offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]

        for m in range(crossover_point):
            offspring[k, m] = parents[parent1_idx, m]
        for m in range(crossover_point-1, -1):
            offspring[k, m] = parents[parent2_idx, m]

        # The next 2 lines are slower than using the above loops because they run with C speed.
        # offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring`**
```

## ****在 C 中生成随机值****

****`mutation()`函数使用`numpy.random.uniform()` 函数返回添加到基因中的随机双精度值:****

```py
**`random_value = numpy.random.uniform(-1.0, 1.0, 1)`**
```

****我们可以避免使用这个函数，而是使用 c 语言的`stdlib` 库中的`rand()`函数来生成随机数。`mutation()` 函数的实现就变成了:****

```py
**`from libc.stdlib cimport rand, RAND_MAX
cdef double DOUBLE_RAND_MAX = RAND_MAX # a double variable holding the maximum random integer in C

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] mutation(numpy.ndarray[numpy.double_t, ndim=2] offspring_crossover, int num_mutations=1):
    cdef int idx, mutation_num, gene_idx
    cdef double random_value
    cdef Py_ssize_t mutations_counter
    mutations_counter = (int) (offspring_crossover.shape[1] / num_mutations) # using numpy.uint8() is slower than using (int)
    cdef double rand_num
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # random_value = numpy.random.uniform(-1.0, 1.0, 1)
            rand_double = rand()/DOUBLE_RAND_MAX
            random_value = rand_double * (1.0 - (-1.0)) + (-1.0)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover`**
```

****首先，`rand()`函数是从`stdlib`导入的，这样我们就可以在 c 中访问这个函数，`rand()`返回一个 0 到 RAND_MAX 范围内的整数值，它是一个常数(它的值至少是 32767)。因为我们希望随机数在 0 到 1 的范围内，所以我们需要将返回的随机值除以最大可能的随机整数。我们通过将 RAND_MAX 复制到一个名为 double_RAND_MAX 的 DOUBLE 变量中，并将随机数除以这个值来实现这一点。缩放后的随机值现在在`rand_double` 变量中可用。然后进行缩放，使其落在-1 到 1 的范围内，并保存在`random_value` 变量中。****

****通过使用 C `rand()` 函数生成随机值，Cython 现在只需要 0.08 秒(80 毫秒)就可以运行。与之前的 0.344 秒相比。这是迄今为止最大的不同。现在代码运行速度比 Python 快 18.25 倍。****

****现在我们已经完成了所有的编辑，完整的 *ga.pyx* 文件如下所示:****

```py
**`import numpy
cimport numpy
import time
import cython

from libc.stdlib cimport rand, RAND_MAX

cdef double DOUBLE_RAND_MAX = RAND_MAX # a double variable holding the maximum random integer in C

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef cal_pop_fitness(numpy.ndarray[numpy.double_t, ndim=1] equation_inputs, numpy.ndarray[numpy.double_t, ndim=2] pop):
    cdef numpy.ndarray[numpy.double_t, ndim=1] fitness
    fitness = numpy.zeros(pop.shape[0])
    # fitness = numpy.sum(pop*equation_inputs, axis=1) # slower than looping.
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            fitness[i] += pop[i, j]*equation_inputs[j]
    return fitness

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness, a

    min_val = -99999999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = 0
        # numpy.where(fitness == numpy.max(fitness)) # slower than looping by 250 ms.
        for a in range(1, fitness.shape[0]):
            if fitness[a] > fitness[max_fitness_idx]:
                max_fitness_idx = a
        # parents[parent_num, :] = pop[max_fitness_idx, :]
        for a in range(parents.shape[1]):
            parents[parent_num, a] = pop[max_fitness_idx, a]
        fitness[max_fitness_idx] = min_val
    return parents

@cython.wraparound(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] crossover(numpy.ndarray[numpy.double_t, ndim=2] parents, tuple offspring_size):
    cdef numpy.ndarray[numpy.double_t, ndim=2] offspring
    offspring = numpy.empty(offspring_size)
    cdef int k, parent1_idx, parent2_idx
    cdef numpy.int_t crossover_point
    crossover_point = (int) (offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]

        for m in range(crossover_point):
            offspring[k, m] = parents[parent1_idx, m]
        for m in range(crossover_point-1, -1):
            offspring[k, m] = parents[parent2_idx, m]

        # The next 2 lines are slower than using the above loops because they run with C speed.
        # offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] mutation(numpy.ndarray[numpy.double_t, ndim=2] offspring_crossover, int num_mutations=1):
    cdef int idx, mutation_num, gene_idx
    cdef double random_value
    cdef Py_ssize_t mutations_counter
    mutations_counter = (int) (offspring_crossover.shape[1] / num_mutations) # using numpy.uint8() is slower than using (int)
    cdef double rand_num
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # random_value = numpy.random.uniform(-1.0, 1.0, 1)
            rand_double = rand()/DOUBLE_RAND_MAX
            random_value = rand_double * (1.0 - (-1.0)) + (-1.0)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef optimize():
    cdef numpy.ndarray equation_inputs, parents, new_population, fitness, offspring_crossover, offspring_mutation
    cdef int num_weights, sol_per_pop, num_parents_mating, num_generations
    cdef list pop_size
    cdef double t1, t2, t

    equation_inputs = numpy.array([4,-2,3.5,5,-11,-4.7])
    num_weights = equation_inputs.shape[0]

    sol_per_pop = 8
    num_weights = equation_inputs.shape[0]
    num_parents_mating = 4

    pop_size = [sol_per_pop,num_weights]
    #Creating the initial population.
    new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

    num_generations = 10000
    t1 = time.time()
    for generation in range(num_generations):
        fitness = cal_pop_fitness(equation_inputs, new_population)

        parents = select_mating_pool(new_population, fitness,
                                          num_parents_mating)

        offspring_crossover = crossover(parents,
                                           offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        offspring_mutation = mutation(offspring_crossover, num_mutations=2)

        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    t2 = time.time()
    t = t2-t1
    print("Total Time %.20f" % t)
    print(cal_pop_fitness(equation_inputs, new_population))`**
```

# ****结论****

****本教程使用 Cython 来减少使用 NumPy 的遗传算法 Python 实现的执行时间。我们将计算时间从 1.46 秒缩短到仅仅 0.08 秒，速度提高了 18 倍。因此，使用 Cython，我们可以在不到 10 秒的时间内完成 100 万次生成，而 Python 需要 180 秒。****

****同样的方法可以用于任何用 Python 编写的代码；一行一行地检查它，找出瓶颈，并通过实现我们在这里看到的技巧来减少计算时间。您不一定需要了解 C，但是了解 C 显然会帮助您实现更快的解决方法。即使没有对 C 语言的深刻理解，在运行长代码或计算量大的代码时，像定义变量类型这样的简单技巧也能产生很大的影响。****