# 神经结构搜索第 2 部分:搜索空间、结构设计和一次性训练

> 原文：<https://blog.paperspace.com/neural-architecture-search-one-shot-training/>

在系列的第一部分[中，我们看了看神经结构问题被处理的所有不同角度。](https://blog.paperspace.com/overview-of-neural-architecture-search/)

有了基础，我们现在将看到如何实现我们在本系列第一篇文章中看到的一些重要概念。具体来说，我们将着眼于设计一个多层感知器的神经架构搜索方法。我们的实施将包括三个特殊功能:

*   一次性建筑培训
*   控制器中的精度预测器
*   加强梯度训练控制器

在这一部分中，我们将着眼于 MLPs 的搜索空间设计，从序列中创建模型架构，以及如何着手实现一次性架构。

完整的代码可以在这里找到[。](https://github.com/codeaway23/MLPNAS)

## 介绍

多层感知器是最容易实现的深度学习架构。几个线性层堆叠在彼此之上；每一个都从前一层获取一个输入，乘以它的权重，加上一个偏差向量，然后将这个向量通过一个选择的激活函数来获得该层的输出。这个前馈过程一直持续到我们最终从最后一层得到我们的分类或回归输出。将此最终输出与地面真实分类或回归值进行比较，使用适当的损失函数计算损失，并使用梯度下降逐一更新所有图层的权重。

## MLPNAS 的游戏计划

当试图自动化神经架构创建时，有一些事情要考虑。在我们深入研究之前，让我们看一下多层感知器神经架构搜索(MLPNAS)管道的简化视图:

```py
def search(self):
	# for the number of controller epochs 
	for controller_epoch in range(controller_sampling_epochs):

		# sample a set number of architecture sequences
		sequences = sample_architecture_sequences(controller_model, samples_per_controller_epoch)

		# predict their accuracies using a hybrid controller
		pred_accuracies = get_predicted_accuracies(controller_model, sequences)

		# for each of these sequences
		for i, sequence in enumerate(sequences):

			# create and compile the model corresponding to the sequence
			model = create_architecture(sequence)

			# train said model
			history = train_architecture(model)

			# log the training metrics
			append_model_metrics(sequence, history, pred_accuracies[i])

		# use this data to train the controller
		xc, yc, val_acc_target = prepare_controller_data(sequences)
		train_controller(controller_model, xc, yc, val_acc_target) 
```

您可能已经注意到，所有外环的功能都属于控制器，所有内环的功能都属于 MLP 发生器。在这个系列的这一部分，我们将看看内部循环。但在我们能够做到这一点之前，我们首先必须对控制器如何生成架构有一些了解。

我们使用的控制器是一个 LSTM 架构，可以生成数字序列。这些数字被解码以创建架构参数，这些参数随后被用于生成架构。我们将在接下来的文章中探讨控制器如何顺序创建有效的架构。现在，我们需要理解每个可能的层配置需要被编码成一个数字，并且我们需要一种机制来将所述数字解码成层中相应数量的神经元和激活。

让我们更详细地看一下。

## 搜索空间

第一个问题是设计搜索空间。我们知道，理论上有无限的可能性存在多少配置，即使我们正在处理非常少的隐藏层。

每个隐藏层中神经元的数量可以是任意正整数。也有很多激活函数，如上所述，它们服务于不同的目的(例如，除非用于分类层，否则你很少使用 *softmax* ，或者如果是二进制分类问题，你将只在分类层使用 *sigmoid* )。

为了照顾到所有这些，我们设计了搜索空间，它大致类似于人类对 MLP 架构的看法，并为我们提供了一种对所述配置进行数字编码或解码的方法。

每个隐层可以用两个参数表示:节点和激活函数。所以我们为序列生成器的*词汇*创建了一个字典。我们考虑一个离散的搜索空间，其中节点的数量可以取特定的值——*8，16，32，64，128，256* 和 *512* 。激活功能也是如此-*sigmoid、tanh、relu* 和 *elu。*我们用一个元组`(number of nodes, activation)`来表示每个这样的可能的层组合。在字典中，关键字是数字，值是层超参数的所述元组。我们从`1`开始编码，因为我们稍后需要填充序列，这样我们可以训练我们的控制器，并且不希望`0`造成混乱。

在为上述节点和激活的每个组合分配一个数字代码后，我们添加了另一个退出选项。最后，根据目标类，我们还将添加最后一层。我们在这个项目中保持 dropout 参数不变，以防止事情过于复杂。

如果有两个目标类，那么我们选择一个单节点 sigmoid 层；否则，我们选择一个 softmax 层，其节点数与类数一样多。

还有一些函数将给定的元组编码成它的数字对应物，反之亦然。

```py
class MLPSearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
    	# define the allowed nodes and activation functions
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']

        # initialize lists for keys and values of the vocabulary
        layer_params = []
        layer_id = []

        # for all activation functions for each node
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):

                # create an id and a configuration tuple (node, activation)
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)

        # zip the id and configurations into a dictionary
        vocab = dict(zip(layer_id, layer_params))

        # add dropout in the volcabulary
        vocab[len(vocab) + 1] = (('dropout'))

        # add the final softmax/sigmoid layer in the vocabulary
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab

	# function to encode a sequence of configuration tuples
    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

	# function to decode a sequence back to configuration tuples
    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence 
```

既然我们已经定义了我们的搜索空间和编码架构的方法，让我们看看如何在给定一个表示有效架构的序列的情况下生成神经网络架构。我们已经在搜索空间中添加了我们可能需要的所有不同的层配置，但我们还没有编写哪些配置有效，哪些无效的规则。我们将在编写控制器时这样做，这将在以后的文章中讨论。

## 模型生成器

人类如何着手设计 MLP？如果你有深度学习的经验，你知道这项工作不需要超过几分钟。

需要考虑的事项有:

1.  **每个隐层有多少个神经元**:有无数个选项，我们需要找到哪种配置会给我们最好的精度。
2.  每个隐藏层使用哪种激活函数:有几种激活函数，要找出哪种函数最适合特定的数据集，需要我们进行自动化的实验。
3.  **添加一个脱落层**:它是有助于我的架构的性能，还是有害？
4.  **最后一层像什么**:是多类问题还是二分类问题？这决定了我们最终层中的节点数量，以及我们在训练这些架构时最终使用的损失函数。
5.  **数据的维度**:如果它需要二维输入，我们可能希望在添加线性层之前将其展平。
6.  **多少个隐藏层** : [Panchal 等人(2011)](http://ijcte.org/papers/328-L318.pdf) 提出，在 MLP 中，我们很少需要两个以上的隐藏层来获得最佳性能。

当我们编写控制器时，会考虑到这些问题。

### 生成 mips

现在，我们将假设我们的控制器工作良好，并且正在生成有效的序列。

我们需要编写一个生成器，它可以获取这些序列，并将它们转换成可以训练和评估的模型。模型生成器将包括以下功能:

*   将序列转换为 Keras 模型
*   编译这些模型

我们将在“一次性架构”小节中讨论减重，并在之后进行训练。这将包括:

*   为 Keras 模型设置权重
*   在训练每个模型后保存训练的权重

准确性的记录将在以后的文章中讨论。

我们的`MLPGenerator`类将继承上面定义的`MLPSearchSpace`类。我们还将在另一个名为`CONSTANTS.py`的文件中保存几个常量。我们导入了如下常量:

*   目标类别
*   使用的优化器
*   学习率
*   衰退
*   动力
*   辍学率
*   损失函数

和其他文件，使用:

```py
from CONSTANTS import *
```

这些常数在`MLPGenerator`类中初始化，如下所示。

```py
class MLPGenerator(MLPSearchSpace):

    def __init__(self):

        self.target_classes = TARGET_CLASSES
        self.mlp_optimizer = MLP_OPTIMIZER
        self.mlp_lr = MLP_LEARNING_RATE
        self.mlp_decay = MLP_DECAY
        self.mlp_momentum = MLP_MOMENTUM
        self.mlp_dropout = MLP_DROPOUT
        self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.mlp_one_shot = MLP_ONE_SHOT
        self.metrics = ['accuracy']

        super().__init__(TARGET_CLASSES)
```

`MLP_ONE_SHOT`常量是一个布尔值，告诉算法是否使用单次训练。

下面是在给定一个对架构和输入形状进行编码的有效序列的情况下创建模型的函数。我们对序列进行解码，创建一个序列模型，并逐个添加序列中的每一层。我们还考虑了二维以上的输入，这将要求我们展平输入。我们也增加了退学的条件。

```py
 # function to create a keras model given a sequence and input data shape
    def create_model(self, sequence, mlp_input_shape):

            # decode sequence to get nodes and activations of each layer
            layer_configs = self.decode_sequence(sequence)

            # create a sequential model
            model = Sequential()

            # add a flatten layer if the input is 3 or higher dimensional
            if len(mlp_input_shape) > 1:
                model.add(Flatten(name='flatten', input_shape=mlp_input_shape))

                # for each element in the decoded sequence
                for i, layer_conf in enumerate(layer_configs):

                    # add a model layer (Dense or Dropout)
                    if layer_conf is 'dropout':
                        model.add(Dropout(self.mlp_dropout, name='dropout'))
                    else:
                        model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))

            else:
                # for 2D inputs
                for i, layer_conf in enumerate(layer_configs):

                    # add the first layer (requires the input shape parameter)
                    if i == 0:
                        model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))

                    # add subsequent layers (Dense or Dropout)
                    elif layer_conf is 'dropout':
                        model.add(Dropout(self.mlp_dropout, name='dropout'))
                    else:
                        model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))

            # return the keras model
            return model
```

请记住，命名`flatten` 和`dropout` 层是很重要的，因为这些名称对于我们的一次性权重设置和更新会很有用。

现在，我们定义另一个函数来编译我们的模型，它将使用我们在`init`函数中定义的常量来获得一个优化器和损失函数，并使用`model.compile`方法返回一个编译后的模型。

```py
 # function to compile the model with the appropriate optimizer and loss function
    def compile_model(self, model):

            # get optimizer
            if self.mlp_optimizer == 'sgd':
                optim = optimizers.SGD(lr=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum)
            else:
                optim = getattr(optimizers, self.mlp_optimizer)(lr=self.mlp_lr, decay=self.mlp_decay)

            # compile model 
            model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)

            # return the compiled keras model
            return model
```

### 一次性建筑

除此之外，我们将处理的另一个有趣的概念是一次性学习或参数共享。参数共享的概念由 [Pham et al. (2018)](https://arxiv.org/pdf/1802.03268.pdf) 引入并推广，其中控制器通过在大型计算图中搜索最佳子图来发现神经网络架构。用策略梯度训练控制器，以选择使验证集上的期望回报最大化的子图。

这意味着整个搜索空间被构建到一个大的计算图中，每个新的架构只是这个超级架构的一个子图。本质上，层的所有可能组合之间的所有权重都可以相互转移。例如，如果生成的第一个神经网络具有以下体系结构:

```py
[(16, 'relu'), (32, 'relu'), (10, 'softmax')]
```

它的重量是这样的:

```py
16 X 32
32 X 10
```

然后，如果第二个网络具有以下体系结构:

```py
[(64, 'relu'), (32, 'relu'), (10, 'softmax')]
```

它的重量是这样的:

```py
64 X 32
32 X 10
```

单次架构方法或参数共享将希望在给定数据上训练第二架构之前，将第二层和最终层之间的训练权重从第一架构转移到第二架构。

因此，我们的算法要求我们始终保持不同层对及其相应权重矩阵的映射。在我们训练任何新的架构之前，我们需要查看特定的层组合在过去是否出现过。如果是，则转移权重。如果没有，权重被初始化，模型被训练，并且新的层组合与权重一起被记录到我们的映射中。

这里要做的第一件事是初始化一个熊猫数据帧来存储我们所有的体重。可以选择将 NumPy 数组直接存储在不同的*中。npz* 文件，或者任何其他你觉得方便的格式。

为此，我们将这段代码添加到`init` 函数中。

```py
if self.mlp_one_shot:

    # path to shared weights file 
    self.weights_file = 'LOGS/shared_weights.pkl'

    # open an empty dataframe with columns for bigrams IDs and weights
    self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})

    # pickle the dataframe
	if not os.path.exists(self.weights_file):
        print("Initializing shared weights dictionary...")
        self.shared_weights.to_pickle(self.weights_file)
```

一次性学习需要我们完成两项任务:

*   在我们开始训练之前，设定架构的权重
*   用新训练的重量更新我们的数据框架

我们为这些写了两个函数。这些函数采用一个模型，逐层提取配置，并将其转换为二元模型–一个 32 节点层后跟一个 64 节点层，因此大小为 *(32 x 64)* ，而一个 16 节点层后跟另一个 16 节点层意味着大小为 *(16 x 16)* 。我们从配置中删除了辍学，因为辍学不影响重量大小。

一旦我们有了这些，在设置重量时，我们查看所有可用的存储重量，看看我们是否已经有了满足重量转移标准的重量。如果是这样，我们转移这些重量；如果没有，我们让 Keras 自动初始化权重。

```py
def set_model_weights(self, model):

    # get nodes and activations for each layer    
    layer_configs = ['input']
    for layer in model.layers:

        # add flatten since it affects the size of the weights
        if 'flatten' in layer.name:
            layer_configs.append(('flatten'))

        # don't add dropout since it doesn't affect weight sizes or activations
        elif 'dropout' not in layer.name:
            layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))

    # get bigrams of relevant layers for weights transfer
    config_ids = []
    for i in range(1, len(layer_configs)):
        config_ids.append((layer_configs[i - 1], layer_configs[i]))

    # for all layers
    j = 0
    for i, layer in enumerate(model.layers):
        if 'dropout' not in layer.name:
            warnings.simplefilter(action='ignore', category=FutureWarning)

            # get all bigram values we already have weights for
            bigram_ids = self.shared_weights['bigram_id'].values

            # check if a bigram already exists in the dataframe
            search_index = []
            for i in range(len(bigram_ids)):
                if config_ids[j] == bigram_ids[i]:
                    search_index.append(i)

            # set layer weights if there is a bigram match in the dataframe 
            if len(search_index) > 0:
                print("Transferring weights for layer:", config_ids[j])
                layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
            j += 1
```

在更新权重时，我们再次查看熊猫数据帧中所有存储的权重，看看我们是否已经有了与训练后的模型中相同大小和激活的权重。如果是，我们用新的重量替换数据框中的重量。否则，我们在数据帧的新行中添加新的形状二元模型以及权重。

```py
def update_weights(self, model):

    # get nodes and activations for each layer
    layer_configs = ['input']
    for layer in model.layers:

        # add flatten since it affects the size of the weights
        if 'flatten' in layer.name:
            layer_configs.append(('flatten'))

        # don't add dropout since it doesn't affect weight sizes or activations
        elif 'dropout' not in layer.name:
            layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))

    # get bigrams of relevant layers for weights transfer
    config_ids = []
    for i in range(1, len(layer_configs)):
        config_ids.append((layer_configs[i - 1], layer_configs[i]))

    # for all layers
    j = 0
    for i, layer in enumerate(model.layers):
        if 'dropout' not in layer.name:
            warnings.simplefilter(action='ignore', category=FutureWarning)

            #get all bigram values we already have weights for
            bigram_ids = self.shared_weights['bigram_id'].values

            # check if a bigram already exists in the dataframe
            search_index = []
            for i in range(len(bigram_ids)):
                if config_ids[j] == bigram_ids[i]:
                    search_index.append(i)

            # add weights to df in a new row if weights aren't already available
            if len(search_index) == 0:
                self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                  'weights': layer.get_weights()},
                                                                 ignore_index=True)
            # else update weights 
            else:
                self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
            j += 1
    self.shared_weights.to_pickle(self.weights_file)
```

一旦我们准备好了权重传递函数，我们就可以编写一个函数来训练我们的模型了。

### 培训生成的架构

如果启用了一次性学习，训练功能将设置模型权重、训练和更新模型权重。否则，它将简单地训练模型并跟踪指标。它将输入数据、Keras 模型、训练模型的时期数、训练测试分割和回调作为输入，并相应地训练模型。

在这个实现中，我们没有添加一旦搜索阶段完成就自动为更多的时期训练最佳模型的功能，但是允许回调作为函数中的变量可以允许我们容易地包括，例如，在我们的最终训练中提前停止。

```py
def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
            self.update_weights(model)
        else:
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
        return history
```

我们的 MLP 发电机准备好了。

## 要记住的事情

有几件事要记住，而处理一杆训练。一次性训练法有几个不容易回答的问题。例如:

*   因为没有预先训练好的权重被转移，所以在获得排名时，更快得到训练的模型处于固有的劣势吗？
*   权重的转移有没有可能损害某个特定架构的性能，而不是提高它？
*   一次性架构方法论如何改变管制员的训练？

除了考虑上面提到的问题之外，还有一个特定于实现的细节需要注意。

单次训练权重必须以有效存储、搜索和检索的方式保存。我意识到，将它们存储在 Pandas 数据帧中会使 NAS 后期的权重转移花费更长的时间，因为数据帧中已经填充了许多权重，搜索它们以进行正确的转移需要更长的时间。如果您有另一种策略来更快地存储和检索权重，那么您应该尝试在您自己的实现中测试它。当你浏览的搜索空间变得巨大，你想要更深或更复杂的架构(想想 CNN 或 ResNet)等时，这变得更加重要。

## 结论

在神经结构搜索系列的第二部分中，我们研究了编码序列到 Keras 结构的自动转换。我们为我们的问题建立了一个搜索空间，对描述层配置的元组进行编码的函数，以及将编码值转换为层配置元组的解码函数。

我们研究了将权重分别传递给每一层，并存储它们的权重以备将来使用。我们看到了如何编译和训练这些模型，并使用 Pandas 数据帧根据它们创建的二元模型存储层权重。我们使用相同的二元模型来检查在新的架构中是否有可以转移的权重。

最后，我们将这些编译后的模型与损失函数、优化器、时期数等信息一起使用。编写用于训练模型的函数。

在下一部分中，我们将设计一个控制器，它可以创建数字序列，这些序列可以通过`MLPGenerator` *转换成有效的架构。*我们还将研究控制器本身是如何被训练的，以及我们是否可以通过调整控制器架构来获得更好的结果。

我希望你喜欢这篇文章。