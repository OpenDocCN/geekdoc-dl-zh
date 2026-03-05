# 如何在没有数据或数据很少的情况下启动数据科学项目

> 原文：<https://winder.ai/how-to-start-a-data-science-project-with-no-or-little-data/>

数据是现代企业的重要资产。它通过展现对客户的独特见解来赋予公司权力，并创造可操作的产品。您拥有的数据越多，您就越能满足并超越客户的期望。

然而，最初对机器学习项目的热情往往会因为缺乏可用数据而降温。当云德。当 AI 被要求帮助公司开发一个基于数据的项目时，最常见的问题是:“数据的可用性如何影响我的项目？”

这篇文章介绍了七种常见情况及其解决方法。如果您的问题没有列出，请[联系](https://winder.ai/about/contact/)我们，我们将很乐意帮助您。

## Q1。我没有数据。我该如何开始？关于数据收集的最佳实践是什么？

任何数据科学项目的第一步都是获取数据。您需要:

1.  找出问题所在

你需要完全理解这个问题。目标是什么？它们是如何衡量的？实体是什么或谁？确定目标和指标将提供关于收集什么和你需要多少的见解。

2.  确定时间框架

接下来，您应该计划如何获得所需的数据。数据是立即可用还是需要收集？那需要多长时间？要花多少钱？这方面可能会影响你的交货时间表。

3.  选择合适的收集方法

您的数据来源将取决于您的业务目标和项目领域。例如，您可以从以下位置收集数据:

*   采访或调查
*   产品指标
*   文件或记录
*   交易或程序数据
*   顾客行为

4.  收集

您现在可以开始收集数据了。创建一个计划来监控您的数据收集过程。定期检查进度会让你适应环境的变化。你可能会发现你需要比你最初意识到的更多的数据。如果是，跳回步骤 1 并重复。在早期阶段，总是偏向更多的数据；忽视比发现更多要容易得多。

## Q2。我有一些正面的例子。我尝试过训练一个分类模型，但是效果不是很好。我如何利用这个来帮助我找到更多的数据？

一个常见的问题是数据集中类的不平衡分布。例如，在欺诈检测数据集中，大多数交易不是欺诈，只有少数是。以下是一些缓解技术:

*   **欠采样**是从多数类中随机删除一些观察值，以使数字与少数类匹配的过程。

**Note**: The full code for this article is [available on WinderResearch’s Gitlab](https://gitlab.com/WinderAI/snippets/blog-no-data/snippets).

```py
# Separate input features and target
X = df.drop('targetClass', axis=1)
y = df.Class

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
majorityClass = X[X.Class==0]
minorityClass = X[X.Class==1]

majorityClass_downsampled = resample(majorityClass,
									replace = False, # sample without replacement
									n_samples = len(minorityClass), # match minority n
									random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([majorityClass_downsampled, minorityClass])

# checking counts
downsampled.targetClass.value_counts() 
```

*   **过采样**是产生合成数据的过程，该过程从少数类的观察值中随机取样。一种常用的技术被称为合成少数过采样技术(SMOTE ),其中通过在特征空间中的点之间画线来产生新的观察值。

```py
#Still using the same majorityClass and minorityClass from above

#upsample minority
minorityClass_upsampled = resample(minorityClass,
								   replace=True, # sample with replacement
								   n_samples=len(majorityClass), # match number in majority class
								   random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([majorityClass, minorityClass_upsampled])

# check new class counts
upsampled.targetClass.value_counts() 
```

```py
#Upsamling minority using SMOTE
smoteTechnique = SMOTE(sampling_strategy='minority', random_state=27)
X_train, y_train = smoteTechnique.fit_sample(X_train, y_train) 
```

请注意，在应用这些技术之前，您应该始终将您的数据集划分为训练测试集。这将提高您的模型性能，并避免过度拟合。

## Q3。我有少量的数据，因为获取新数据很有挑战性。但是我想创建一个分类模型。我如何才能稳健地做到这一点？

你缺乏数据是至关重要的，因为数据是任何人工智能项目的核心。生产模型性能与训练数据的大小相关。但是多少才够好呢？

作为一个粗略的指导，您需要的观察值大约是您的模型中参数数量的十倍。例如，如果您正在构建具有两个特征的线性回归模型，那么您至少需要 30 个观测值(两个权重，一个截距)。少于此，您的模型可能会过拟合。然而，根据用例，可以使用的数据更少。

以下因素会影响您需要多少数据:

*   模型中的参数数量
*   预期模型性能
*   你的模型的输出。

对于任何数量的数据，请遵循以下建议，使您的模型更加稳健:

1.  选择简单的模型:较小的模型需要较少的数据。

*   如果你在训练一个分类器，从逻辑回归开始。
*   对于基于树的模型，限制最大深度。
*   如果您要预测类别，请从具有有限数量特征的简单线性模型开始。
*   应用正则化方法使模型更加保守。

```py
#Implementing a simple classifier with regularization method
logReg = LogisticRegression(solver='liblinear',
							penalty='l1',
							C=0.1,
							class_weight='balanced'
)

#fit model
logReg.fit(X_train, y_train)

#get predictions
logReg.predict(X_test) 
```

2.  从数据中移除异常值。在小数据集上进行训练时，离群值会显著影响您的模型。您可以删除它们或使用更稳健的技术，如分位数回归。

```py
#Detect Outliers and remove them
isoforest = IsolationForest(n_jobs=-1, random_state=1)
isoforest.fit(X_train)
outliersPred = isoforest.predict(X_train)
X_train = X_train[np.where(outliersPred == 1, True, False)]
y_train = y_train[np.where(outliersPred == 1, True, False)] 
```

3.  选择相关特征:这可以使用几种技术来完成，例如包括递归消除、分析与目标变量的相关性以及重要性分析。特性选择需要熟悉主题领域，因此咨询领域专家将是有益的。

```py
#Select Relevant features with recursive elimination based on initial model and removing them
from sklearn.feature_selection import RFE
rfe = RFE(logReg)
rfe.fit(X_train, y_train)
X_train.drop(X_train.columns[np.where(rfe.support_ == False)[0]], axis=1, inplace=True) 
```

4.  集合几个模型。组合来自许多模型的结果可以提供共识，并使解决方案更加稳健。

```py
#Ensembling initial model with XGBooost
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier

preds = pd.DataFrame() #predictions Dataframe
stackedModel = StackingClassifier(classifiers=[
									logReg,
									XGBClassifier(max_depth=2)
								],
								meta_classifier= logReg
)
stackedModel.fit(X_train, y_train)
preds['stack_pred'] = stackedModel.predict(X_test) 
```

## Q4。我根据少量数据创建了一个模型。但是当我把它投入生产时，性能下降了。如何提高模型的稳健性？

有几种策略可以用来提高模型性能:

*   如果你的模型不合适，你可以尝试**增加输入特征的数量**或者一个**更复杂的模型**。如果你的模型过度拟合，做相反的事情。

*   交叉验证:是防止过度拟合的重要预防措施。您使用初始训练数据来创建多个小型训练测试拆分，然后使用它们来调整您的模型。在 k 倍交叉验证中，数据被分成 k 个子集，称为折叠。然后，在 k-1 个折叠上迭代训练该算法，同时使用剩余的折叠作为测试集(称为“保持折叠”)。

*   正则化有助于使你的模型更简单。这取决于您使用的模型类型。例如:在神经网络上使用 dropout，修剪决策树等。

*   集成将来自不同模型的预测结合起来。集合有多种方法，但最常用的两种方法是:

    *   Bagging 试图通过并行训练大量“强”学习者来减少复杂模型过度拟合的机会，然后将他们结合起来以“平滑”他们的预测。

    *   Boosting 试图通过顺序训练大量“弱”学习者来增加简单模型的预测灵活性，然后将它们组合成单个强学习者。

你也可以通过简单多数投票将不同的模型组合在一起。

## Q5。我已经基于少量数据创建了一个模型，并且我正在使用一个复杂的模型(例如深度学习)。但是在现实生活中效果并不好？

深度神经网络需要大型数据集来实现高性能。获取的数据越多，模型的表现就越好。对于更小的数据集，回归、随机森林和 SVM 等更简单的机器学习模型往往优于深度网络。考虑应用经典模型或获取更多数据。线性算法获得了良好的性能，每个类有数百个示例。但是，对于像人工神经网络这样的非线性算法，每个类可能需要数千个样本。

## Q6。我有很多数据，但我不知道用哪一个。我该怎么办？

阻碍数据科学项目的最常见问题是没有清楚地理解业务问题。确定项目目标首先要问很多问题，这些问题要具体、相关、明确。

当提出正确的问题时，数据开始提供全面的观点和相关的预测。

以下是不同类型的数据如何帮助您解决您的业务计划:

*   交易数据支持更精细、更详细的决策(本地化、季节性、多维度)。
*   非结构化数据支持完整且更准确的决策(新的指标、维度和维度属性)
*   数据速度支持更频繁、更及时的决策(每小时比每周；按需分析模型更新)。
*   预测分析支持更具可操作性和预测性的决策(优化、推荐、预测、评分、预测)。

## Q7。我有数据，但很敏感。我如何使用敏感数据？

敏感数据是有用的信息，只能在混淆时使用。这包括:

*   **个人数据**:姓名、身份证号、来自手机或 GPS 的位置数据、身体特征、经济特征、&mldr；
*   **机密数据**:财务信息、密码、国家安全
*   **业务关键数据**:如果泄露，可能会对业务造成损害(例如商业秘密)
*   **道德**:有时没有法律要求，但从道德上讲，数据应该匿名。

### 存储和共享敏感数据

有几种存储敏感数据的安全策略:

*   **匿名化**:可识别数据的不可逆销毁。匿名化的个人数据不再能够识别个人身份，也不再被视为个人数据。

*   **假名化**:用可逆的、一致的值替代可识别数据的方法。与匿名化不同，可能允许回溯的个人相关数据不会被清除。尽管假名数据在法律上仍被视为敏感数据，但它被认为是一种安全的方法。

*   **加密**:将明文转换为密文的过程。加密获取可读数据并对其进行修改，使其看起来是随机的。好的加密策略使用可靠的加密和方便的密钥管理。

```py
#Import libraries
import cryptography
from cryptography.fernet import Fernet

#generate an encryption key.
key = Fernet.generate_key()

#Save encryption key
file = open('key.key', 'wb') #wb = write bytes
file.write(key)
file.close()

# Open the file to encrypt
with open(‘sensitiveData.csv’, 'rb') as f:
	data = f.read()

#Encrypt data using the key
fernet = Fernet(key)
encrypted = fernet.encrypt(data)

# Write the encrypted file
with open('sensitiveData.csv.encrypted', 'wb') as f:
	f.write(encrypted) 
```

### 模型的敏感数据最佳实践

您应该权衡数据的效用和风险水平:

*   以高置信度识别敏感数据:敏感数据的常见情况有:列中的敏感数据:可以是结构化数据集中的特定列，如用户的名字、姓氏或邮寄地址。基于文本的非结构化数据集中的敏感数据:通常可以使用基于文本的非结构化数据集中的已知模式进行检测。自由形式的非结构化数据中的敏感数据:可以是文本报告、录音、照片或扫描的收据。字段组合中的敏感数据非结构化内容中的敏感数据:非结构化内容中嵌入的上下文信息。

*   创建数据治理计划和最佳实践文档。这将有助于在无法屏蔽或删除敏感数据时做出适当的决策。这些是在建立治理策略框架时要考虑的常见概念:为治理文档建立一个安全的位置。在文档中省略加密密钥、散列函数或其他工具。记录敏感数据的所有来源、它们的存储位置，并精确说明当前数据的类型。包括为保护它而采取的补救措施。记录补救步骤复杂、不一致或不可能的位置。建立一个持续扫描和确定新的敏感数据来源的流程。描述临时或永久访问敏感数据的角色和(可能)个别员工的姓名，并描述他们要求访问权限的原因。确定员工可以在哪里访问敏感数据，是否、如何以及在哪里可以复制这些数据，以及与访问相关的任何其他限制。定期审查谁可以访问敏感数据，并确定是否仍然需要访问。传达、执行并定期审查您的政策。

*   在不显著影响项目的情况下保护敏感数据，您可以通过以下方式保护敏感数据:移除敏感数据:在构建您的机器学习模型之前，如果您的项目不需要特定于用户的信息，请将其从数据集中删除。然而，在某些情况下，这可能会显著降低数据集的价值。屏蔽敏感数据:当无法移除敏感信息时，您仍然可以使用屏蔽格式的数据来训练有效的模型。屏蔽技术包括:通过用散列值或加密值替换所有出现的纯文本标识符来应用替换密码。通过用不相关的虚拟值替换存储在每个敏感字段中的真实值来进行标记化。映射在一个单独的、更安全的数据库中被加密/散列。这种方法只有在相同的令牌值被重复用于相同的值时才有效。使用主成分分析(PCA)等降维技术来混合各种特征，并仅在生成的 PCA 向量上训练您的模型。粗化敏感数据:用于降低数据的精度或粒度，使识别数据集中的敏感数据变得更加困难，同时与用粗化前的数据训练模型相比，仍能给你带来类似的好处。

## 结论

启动一个数据科学项目不一定需要收集数十亿个样本。所需的数据量很大程度上取决于业务问题的类型和您使用的技术类型。也就是说，用少量数据启动您的数据科学之旅是可能的，但请确保您约束了问题并限制了您的模型选择。

**Note**: The full code for this article is [available on WinderResearch’s Gitlab](https://gitlab.com/WinderAI/snippets/blog-no-data/snippets).