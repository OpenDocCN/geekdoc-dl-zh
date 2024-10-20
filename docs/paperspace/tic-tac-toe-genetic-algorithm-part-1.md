# 用遗传算法开发井字游戏代理:遗传算法入门(第 1 部分)

> 原文：<https://blog.paperspace.com/tic-tac-toe-genetic-algorithm-part-1/>

![](img/9972e27f8b20fcacddbd104320c8171a.png)

## 
简介

‌Machine 学习是计算的一个不断发展的方面，它能够让系统从提供给它的数据中学习，这些系统可以分为‌‌

*   监督式学习:监督式学习只涉及从标记数据中学习，例如回归和分类。
*   无监督:无监督学习涉及一个从无标签数据中学习的系统或算法，它能够根据传递给它的数据中发现的相似性对数据进行分类，一个例子是 K 均值聚类算法
*   强化学习:这是一种解决机器学习问题的较新方法，其中给一个代理一个环境，并要求其与环境交互，每次交互都根据某些定义的规则进行评分和奖励或惩罚。‌‌

解决机器学习问题的方法有很多，其中一些可以分为监督学习和非监督学习，所以这并不总是一个是或否的分类。‌‌

### 什么是遗传算法

遗传算法是一种元启发式问题解决过程，其灵感来自于自然界中的自然选择过程。它用于在问题空间中寻找高度优化问题的解决方案，并通过遵循受自然过程启发的步骤来做到这一点，其中最强壮或最适合的个体被允许生存下来并将其遗传物质传递给下一代。这样做可以使下一代更好地适应他们所处的环境，或者更好地解决问题。‌‌

### 自然选择的过程

在自然界中，自然选择的概念支持适者生存的思想，即更适应环境的生物生存下来，并且最有可能将它们的特征/基因传递给下一代。它还涉及突变的概念，这意味着生物能够使其基因突变，以便能够生存或将适应其当前的环境。这些突变是随机的，在自然选择的过程中起着很大的可变作用。这种情况会在几代人或几个周期内发生，在人类身上也能看到:在某些个体的情况下，他们的祖先对疾病产生了免疫力，并幸存下来，能够将这种免疫力传递给他们的后代。[](https://www.nature.com/scitable/topicpage/evolutionary-adaptation-in-the-human-lineage-12397/)

‌A 类似的过程可以应用于计算和问题解决，我们试图模拟这一过程的所有步骤，以便能够在现实生活中重现。‌‌

让我们分别查看组件和步骤。‌‌

*   人口
*   基因
*   健康
*   基因库
*   选择
*   交叉
*   变化

### 人口

这些是我们为样本问题生成的个体。它们是我们特定问题的个别解决方案，而且从一开始就都是错的。随着时间的推移，它们被优化，以便能够在它们的问题空间中解决问题。每一种都有称为基因和染色体的组成部分。让我们来看一个问题，我们想解决找到一个简单的文本，让我们说，文本是一个 24 个字母的字符串“我是我的剑的骨头”，初始化一个群体来解决这将只是一个 N 群体的字符串组成的随机集的 24 个字符。‌‌

### 基因和染色体

在遗传算法问题中，群体中个体特征的最小单位是基因。染色体是由基因组成的；他们定义了解决问题的方法。它们可以通过与您的用例相匹配的任何方式来定义，但是，在我们上面提到的简单示例中，我们希望优化我们群体的成员，使之等于短语“我是我的剑的骨头”，我们群体的基因只是我们群体中每个个体的字母，而染色体可以是全文。‌

### 健身(健身功能)

这是一个计算个人解决问题集的能力的函数。它给群体中的每个个体一个健康分数。这被用来评估这些基因是否会传递给下一代。除了第一个种群之外的每个种群都是由先前的个体组成的，上一代的平均适应值总是大于上一代。‌‌

### 基因库

基因库被认为是可传递给下一代的每一组可用基因(染色体)的列表。它被分配以支持更多具有更有利基因的父母，并且具有更有利基因的个体被多次添加到池中，以增加其被选择产生后代或将其性状传递给下一代的概率。一个具有较少特征的个体不会被经常添加，但仍然会被添加，主要是因为它可能具有我们仍然想要的令人满意的特征。‌‌

### 选择

在自然界中，拥有令人满意的特征的个体或更适应环境的个体能够繁殖并把他们的遗传特征传递给下一代是非常普遍的。这同样适用于遗传算法。在这种情况下，我们优先选择适应性分数较高的个体，而不是适应性分数较低的个体，这是通过确保基因库中有更多合适的个体可供选择来实现的。‌
‌

一对个体被选择在一起，产生一个或更多的后代(取决于你的设计)。身体健康的人更有可能被选中。‌‌

### 交叉

这是结合两个父母个体创造一个或多个后代的过程。这涉及到父母双方染色体的交叉。要做到这一点，我们必须选择一个交叉点，或者它可以是随机的，或者是中间的一个固定点，这取决于你如何设计它。创建的新个体被添加到下一个群体中。‌

### 变化

不幸的是，这不会是我们在科幻电影中看到的那种突变，突变只是暗示染色体中的基因改变了它们的值。的可能性。突变发生率低且随机，但可以根据算法的需要进行调整。这是必要的，因为有一种可能性，即初始群体中没有一个成员具有特定的性状来帮助它达到全局终极。‌‌

### 最后一档

在我们注意到当前代没有比上一代更有效地解决问题之后，我们终止了我们的算法，因此我们剩下多个个体，这些个体都是我们的问题集的解决方案。‌

‌‌
‌**样本遗传算法问题的代码库**

您可以在本地环境中运行这个 GNN 类。为了更快地运行或处理更复杂的任务，可以考虑通过 Paperspace Core‌.使用高端 GPU 远程访问虚拟桌面来启动您的工作流程

```py
class GNN {
   constructor(number, goal) {
       this.population = []
       this.goal = goal || '';
       this.genePool = []
       this.numnber = number
       this.mostFit = {fitness:0}
       this.initializePopulation()
   }

   initializePopulation() {
       const length = this.goal.length;
       for (let i = 0; i <= this.numnber; i++) this.population.push({ chromosomes: this.makeIndividual(length), fitness: 0 })
   }

   solve() {
       for (var i = 0; i < 10000; i++) {
           this.calcFitnessAll()
           this.crossParents()
           this.mutate()
       }

       console.log( this.mostFit )
   }

   mutate() {
       this.population.forEach((object) => {
           const shouldIMutateChromosome = Math.random() < 0.1
           if (shouldIMutateChromosome) {

               for (var i = 0; i < object.chromosomes.length; i++) {
                   const shoulIMutateGene = Math.random() < 0.05
                   if (shoulIMutateGene) {
                       const splitVersion = object.chromosomes.split('')
                       splitVersion[i] = this.makeIndividual(1)
                       object.chromosomes = splitVersion.join('')
                   }
               }
           }
       })

   }

   calcFitnessAll() {
       let mostFit = {fitness:0}

       this.population.forEach((object) => {
           let str1 = this.goal
           let str2 = object.chromosomes
           let counter = 0;
           for (var i = 0; i < str1.length; i++) {
               if (str1.charAt(i) === str2.charAt(i)) {
                   counter++;
               }

           }
           object.fitness = counter > 0 ? counter : 1

           for (var i = 0; i < object.fitness; i++) {
               this.genePool.push(object.chromosomes)
           }
           object.fitness > mostFit.fitness ? mostFit = object : mostFit = mostFit

       })
       if (mostFit.fitness>this.mostFit.fitness) {
           this.mostFit = mostFit
       }

       console.log('Max fitnees so far is ' + this.mostFit.fitness)
   }

   crossParents() {
       const newPopulation = [];
       const pool = this.genePool
       this.population.forEach(() => {

           const parent1 = pool[Math.floor(Math.random() * pool.length)];

           const parent2 = pool[Math.floor(Math.random() * pool.length)];

           ////select crossSection
           const crossSection = Math.floor(Math.random() * this.goal.length - 1)

           let newKid = `${parent1.slice(0, crossSection)}${parent2.slice(crossSection, this.goal.length)}`;

           newPopulation.push({ chromosomes: newKid, fitness: 0 })
       })

       this.population = newPopulation
       this.genePool = []

   }

   makeIndividual(length) {
       var result = '';
       var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,';
       var charactersLength = characters.length;
       for (var i = 0; i < length; i++) {
           result += characters.charAt(Math.floor(Math.random() *
               charactersLength));
       }
       return result;
   }
}
const goal = `I am the bone of my sword. Steel is my body, and fire is my blood.`
const gnn = new GNN(1000, goal)
gnn.solve() 
```

## ‌
分解代码库

我们使用基于类的建模方法来管理我们的遗传算法的状态。‌‌

我们首先在 initializePopulation 函数中创建人口。‌

```py
initializePopulation() {
    const length = this.goal.length;
    for (let i = 0; i <= this.number; i++) this.population.push({ 			chromosomes: this.makeIndividual(length), fitness: 0 })
}
```

该函数使用`makeIndividual`函数创建一个个体，并将其添加到群体数组中。‌‌

下一步是用遗传算法来解决这个问题。‌

```py
solve() {
        for (var i = 0; i < 10000; i++) {
                this.calcFitnessAll()
                this.crossParents()
                this.mutate()
        }
        console.log( this.mostFit )
}
```

‌

solve 函数迭代 10000 次，一遍又一遍地执行优化步骤，每一步都创建一批对我们的问题更优化的个体。‌‌‌

```py
this.calcFitnessAll()
this.crossParents()
this.mutate()
```

这些是我们遵循的步骤，包括检查适应度，杂交亲本，然后变异它。‌‌

### 额外事实

香草遗传算法是一个很好的工具，但也存在其他变种，这也是值得 noting‌‌.

### ****精英主义****

这意味着我们把一定数量的表现最好的个体原封不动地传给下一代。这将确保解决方案的质量不会随着时间的推移而降低。‌‌

### ****并行遗传算法****

这涉及到何时使用多种遗传算法来解决一个特定的问题。这些网络都是彼此并行运行的，然后在它们完成后，选择其中最好的一个。‌‌

### ****最终想法****

遗传算法是机器学习中一个有趣的概念，用于寻找问题集的最优解。它并不总是解决某些问题的最理想的工具，但它对全局优化问题很有效。像变异和交叉这样的过程有助于将解从局部移到全局最大值。‌‌

### ****下一步****

这是我们系列的第一部分。我觉得解释项目的理论方面并提供一个基础代码库是很重要的，我们接下来将致力于构建我们的环境和代理来与 environment‌‌交互

### 来源:

1.  [https://www . nature . com/sci table/topic page/evolutionary-adaptation-in-the-human-lineage-12397/](https://www.nature.com/scitable/topicpage/evolutionary-adaptation-in-the-human-lineage-12397/)