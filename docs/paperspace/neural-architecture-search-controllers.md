# 神经结构研究第 3 部分:控制器和精度预测器

> 原文：<https://blog.paperspace.com/neural-architecture-search-controllers/>

在本系列的第一部分中，我们看到了关于神经架构搜索的概述，包括文献综述。在第 2 部分中，我们看到了如何[将我们的编码序列转换成 MLP 模型](https://blog.paperspace.com/neural-architecture-search-one-shot-training/)。我们还研究了训练这些模型，为一次性学习逐层转移权重，以及保存这些权重。

为了得到这些编码序列，我们需要另一种机制，以对应于 MLP 的有效架构的方式生成序列。我们还想确保我们不会两次训练同一个架构。这就是我们将在这里讨论的内容。

具体来说，我们将在本文中讨论以下主题:

*   控制器在 NAS 中的角色
*   创建控制器
*   控制器架构
*   准确性预测
*   培训管制员
*   采样架构
*   获得预测精度
*   结论

## 控制器在 NAS 中的角色

获得这些编码序列的方法是使用一个循环网络，它会不断地为我们生成序列。每个序列都是我们以定向方式导航的搜索空间的一部分。我们寻找最佳架构的方向取决于我们如何训练控制器本身。

我们将把代码分成几个部分，但是完整的代码也可以在这里找到。

在[第 2 部分](https://blog.paperspace.com/neural-architecture-search-one-shot-training/)中，我们看到我们的 NAS 项目的总体流程如下所示:

```py
def search(self):
	# for number of controller epochs 
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

搜索的内环主要是模型生成器的任务，而外环是控制器的任务。内循环和外循环中都有一些函数，涉及准备数据和存储模型度量，这是我们的生成器和控制器顺利工作所必需的。这些是主`MLPNAS`类的一部分，我们将在本系列的下一部分(也是最后一部分)研究它们。

现在，我们将特别关注控制器:

*   它是如何设计的，以及控制器设计的不同替代方案
*   如何生成可以传递给 MLP 生成器以创建和训练架构的有效序列
*   如何训练管制员本身

控制器架构可以设计为包含精度预测器。这也可以通过多种方式来实现，例如，通过共享控制器和精度预测器的 LSTM 权重。使用准确度预测器有一些缺点。我们也会谈到这些。

## 创建控制器

控制器是一个递归系统，它根据我们在搜索空间中设计的映射生成编码序列(在系列的第[第二部分](https://blog.paperspace.com/neural-architecture-search-one-shot-training/))。这个控制器将是一个模型，可以根据它生成的序列进行迭代训练。这从一个控制器开始，该控制器在不知道性能良好的架构看起来像什么的情况下生成序列。我们创建一些序列，训练这些序列，评估它们，并从这些序列中创建一个数据集来训练我们的控制器。**本质上，在每个控制器时期，都会创建一个新的数据集供控制器学习。**

为了做到这些，我们需要在控制器类中初始化一些参数——我们需要的常量。控制器将继承我们在本系列第二部分中创建的`MLPSearchSpace`类。

这些常数包括:

*   控制器 LSTM 隐藏层数
*   用于培训的优化器
*   用于培训的学习率
*   衰减到用于训练
*   培训的动力(在 SGD 优化器的情况下)
*   是否使用精度预测器
*   建筑的最大长度

```py
class Controller(MLPSearchSpace):

    def __init__(self):

        # defining training and sequence creation related parameters
        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR

        # file path of controller weights to be stored at
        self.controller_weights = 'LOGS/controller_weights.h5'

        # initializing a list for all the sequences created
        self.seq_data = []

        # inheriting from the search space
        super().__init__(TARGET_CLASSES)

        # number of classes for the controller (+ 1 for padding)
        self.controller_classes = len(self.vocab) + 1
```

我们还初始化一个空列表来存储我们的控制器已经创建并测试的所有编码架构。这将防止我们的控制器一次又一次地采样相同的序列。我们需要在开始时初始化这个序列(而不是在函数中),因为这个序列数据需要在多个控制器时期中保持不变，而用于采样的函数要被调用几次。

如果您不仅要设计 MLP，还要设计基于深度 CNN 的架构(类似于 Inception 或 ResNets)，您可能要考虑不要将这些架构存储在列表中，而是将其保存在某个临时文件中以释放内存。

## 控制器架构

控制器可以以多种方式设计，并且可以做的实验数量没有真正的限制。从根本上说，我们需要一个可以从控制器中提取并解码成实际 MLP 架构的顺序输出。RNNs 和 LSTMs 听起来是很好的选择。

为控制器的学习尝试不同的优化技术，大多需要我们处理不同的优化器或者构建定制的损失函数。

下面可以看到一个简单的 LSTM 控制器。

```py
 def control_model(self, controller_input_shape, controller_batch_size):
        main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output])
        return model
```

上面显示的架构非常简单。有:

*   一个输入层，其大小取决于输入形状和批处理形状
*   具有用户指定尺寸的 LSTM 图层
*   密集层，节点取决于词汇表的大小

这是一个顺序架构，可以使用我们选择的优化器和损失函数进行训练。

但是，除了利用上面提到的架构，还有其他方法来设计这些控制器。我们可以:

*   改变我们建筑中 LSTM 层的 LSTM 维度
*   添加更多的 LSTM 层，并改变其尺寸
*   添加密集层，并改变节点和激活函数的数量

其他方法包括建立可以同时输出两种东西的模型。

## 准确性预测

这种简单的 LSTM 体系结构可以通过不仅考虑基于损失函数的优化，而且考虑使用准确度预测器的并行模型，而变成对抗模型。精度预测器将与序列发生器的 LSTM 层共享权重，帮助我们创建更好的通用架构。

```py
def hybrid_control_model(self, controller_input_shape, controller_batch_size):
    # input layer initialized with input shape and batch size
    main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')

    # LSTM layer
    x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)

    # two layers take the same LSTM layer as the input, 
    # the accuracy predictor as well as the sequence generation classification layer
    predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
    main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)

    # finally the Keras Model class is used to create a multi-output model
    model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
    return model
```

预测器输出将是具有 sigmoid 激活函数的单神经元密集层。这一层的输出将作为架构验证准确性的代理。

我们还可以从主输出 LSTM 中分离出精度预测器 LSTM，如下所示。

```py
def hybrid_control_model(self, controller_input_shape, controller_batch_size):
    # input layer initialized with input shape and batch size
    main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')

    # LSTM layer
    x1 = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
    # output for the sequence generator network
    main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x1)

    # LSTM layer
    x2 = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
    # single neuron sigmoid layer for accuracy prediction
    predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x2)

    # finally the Keras Model class is used to create a multi-output model
    model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
    return model
```

这样，我们不会因为精度预测器而影响序列预测，但我们仍然有一个网络学习来预测架构的精度，而无需训练它们。

在下一个也是最后一个部分，我们将看到当通过应用增强梯度来训练模型时，我们的损失函数实际上已经考虑了每个架构的验证准确性。准确性预测器的干预实际上导致控制器创建的架构不能给我们提供与仅使用一次性学习生成的架构一样高的准确性。为了彻底起见，我们在这里提到精度预测器。

## 培训管制员

一旦我们准备好了模型，我们就编写一个函数来训练它。作为输入，该函数将采用损失函数、数据、批量大小和历元数。这样，我们可以使用自定义损失函数来训练我们的控制器。

下面的训练代码是针对上面提到的简单控制器的。它不包括精度预测值。

```py
def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
    # get the optimizer required for training
    if self.controller_optimizer == 'sgd':
        optim = optimizers.SGD(lr=self.controller_lr,
                               decay=self.controller_decay,
                               momentum=self.controller_momentum)
    else:
        optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr, 
                                                   decay=self.controller_decay)

    # compile model depending on loss function and optimizer provided
    model.compile(optimizer=optim, loss={'main_output': loss_func})

    # load controller weights
    if os.path.exists(self.controller_weights):
        model.load_weights(self.controller_weights)

    # train the controller
    print("TRAINING CONTROLLER...")
    model.fit({'main_input': x_data},
              {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
              epochs=nb_epochs,
              batch_size=controller_batch_size,
              verbose=0)

    # save controller weights
    model.save_weights(self.controller_weights)
```

为了训练具有准确度预测器的模型，在两个地方修改上述函数:编译模型阶段和训练阶段。它需要包括用于两个不同输出的两个损失，以及用于每个损失的权重。类似地，训练命令需要在输出字典中包含第二个输出。对于预测器，我们使用均方误差作为损失函数。

```py
def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
    # get the optimizer required for training
    if self.controller_optimizer == 'sgd':
        optim = optimizers.SGD(lr=self.controller_lr,
                               decay=self.controller_decay,
                               momentum=self.controller_momentum)
    else:
        optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr, 
                                                   decay=self.controller_decay)

    # compile model depending on loss function and optimizer provided
    model.compile(optimizer=optim,
                  loss={'main_output': loss_func, 'predictor_output': 'mse'},
                  loss_weights={'main_output': 1, 'predictor_output': 1})

    # load controller weights
    if os.path.exists(self.controller_weights):
        model.load_weights(self.controller_weights)

    # train the controller
    print("TRAINING CONTROLLER...")
    model.fit({'main_input': x_data},
              {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
              'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
              epochs=nb_epochs,
              batch_size=controller_batch_size,
              verbose=0)

    # save controller weights
    model.save_weights(self.controller_weights)
```

## 采样架构

一旦模型架构和训练功能完成，我们需要最终使用这些模型来预测架构序列。如果使用精度预测器，还需要一个函数来获取预测的精度。

取样过程要求我们对 MLP 架构的设计规则进行编码，以避免无效的架构。我们也不希望一次又一次地创建相同的架构。其他需要考虑的事项包括:

*   脱落层出现的时间和位置
*   建筑的最大长度
*   建筑的最小长度
*   每个控制器时期要采样多少架构

这些问题在下面的采样结构序列的函数中被考虑。我们运行一个嵌套循环；外部循环继续进行，直到我们获得所需数量的采样序列。内部循环使用控制器模型来预测每个架构序列中的下一个元素，从空序列开始，以具有一个或多个隐藏层的架构结束。其他约束条件是，辍学不能在第一层，最后一层不能重复。

在生成序列中的下一个元素时，我们根据所有可能元素的概率分布对其进行随机采样。这允许我们利用从控制器获得的 softmax 分布来导航搜索空间。概率采样有助于搜索空间的有效探索，同时不会偏离控制器模型所指示的太多。

```py
 def sample_architecture_sequences(self, model, number_of_samples):
        # define values needed for sampling 
        final_layer_id = len(self.vocab)
        dropout_id = final_layer_id - 1
        vocab_idx = [0] + list(self.vocab.keys())

        # initialize list for architecture samples
        samples = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')

        # while number of architectures sampled is less than required
        while len(samples) < number_of_samples:

            # initialise the empty list for architecture sequence
            seed = []

            # while len of generated sequence is less than maximum architecture length
            while len(seed) < self.max_len:

                # pad sequence for correctly shaped input for controller
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)

                # given the previous elements, get softmax distribution for the next element
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                probab = probab[0][0]

                # sample the next element randomly given the probability of next elements (the softmax distribution)
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]

                # first layer isn't dropout
                if next == dropout_id and len(seed) == 0:
                    continue
                # first layer is not final layer
                if next == final_layer_id and len(seed) == 0:
                    continue
                # if final layer, break out of inner loop
                if next == final_layer_id:
                    seed.append(next)
                    break
                # if sequence length is 1 less than maximum, add final
                # layer and break out of inner loop
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                # ignore padding
                if not next == 0:
                    seed.append(next)

            # check if the generated sequence has been generated before.
            # if not, add it to the sequence data. 
            if seed not in self.seq_data:
                samples.append(seed)
                self.seq_data.append(seed)
        return samples
```

## 获得预测精度

获得预测的另一部分是获得每个模型的预测精度。该函数的输入将是模型和生成的序列。输出将是一个介于 0 和 1 之间的数字，即模型的预测验证精度。

```py
 def get_predicted_accuracies_hybrid_model(self, model, seqs):
        pred_accuracies = []        
        for seq in seqs:
            # pad each sequence
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            # get predicted accuracies
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies
```

我们已经完成了控制器的构建。

## 结论

之前我们看到了如何编写我们的模型生成器，它将实现一次性学习作为一个可选的特性。只有当我们已经拥有架构的编码序列，以及映射这些编码的搜索空间时，模型生成器才是有用的。因此，我们在本系列的第二部分定义了搜索空间和生成器。

在这一部分中，我们学习了如何使用基于 LSTM 的架构对架构的编码序列进行采样。我们研究了设计控制器的不同方法，以及如何利用精度预测器。准确度预测器不一定能创建出优秀的架构，但它确实有助于创建更好地概括的模型。我们还讨论了在精度预测器和序列生成器之间共享权重。

之后，我们学习了如何训练这些控制器模型，这取决于它们是单输出还是多输出模型。我们查看了架构编码生成器本身，它考虑了不同的约束来创建在所用层的顺序、这些架构的最大长度等方面有效的架构。我们用一个很小的函数完成了我们的控制器，该函数用于在给定模型和我们需要预测的序列的情况下获得预测精度。

本系列的下一部分也是最后一部分将总结 MLPNAS 的完整工作流程。最后，我们将把第二部分的模型生成器与第三部分的控制器集成起来，以实现神经结构搜索过程的自动化。我们也将学习加强梯度，作为我们控制器的优化方法。

敬请关注。