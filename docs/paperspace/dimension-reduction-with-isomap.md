# 降维- IsoMap

> 原文：<https://blog.paperspace.com/dimension-reduction-with-isomap/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  【T0 向】T1
5.  t-SNE
6.  **[IsoMap](https://blog.paperspace.com/dimension-reduction-with-isomap)**
7.  [Autoencoders](https://blog.paperspace.com/dimension-reduction-with-autoencoders)

(带有数学和代码的 jupyter 笔记本(spark)可在 [github repo](https://github.com/asdspal/dimRed) 上获得)

Isomap 代表等距映射。Isomap 是一种基于谱理论的非线性降维方法，它试图在低维中保持测地线距离。Isomap 从创建一个邻居网络开始。之后，它使用图距离来近似所有点对之间的测地线距离。然后，通过测地距离矩阵的特征值分解，找到数据集的低维嵌入。在非线性流形中，当且仅当邻域结构可以近似为线性时，欧几里德距离度量才成立。如果邻域包含空洞，那么欧几里德距离可能会产生很大的误导。与此相反，如果我们通过遵循流形来测量两点之间的距离，我们将更好地近似两点之间的距离。让我们用一个极其简单的二维例子来理解这一点。假设我们的数据位于一个二维结构的圆形流形上，如下图所示。
![](img/fe1a5c33bbc95169deb34d752049a13d.png)
**为什么非线性流形中的测地线距离优于欧氏距离？**

我们将使用欧几里得距离和近似测地线距离将数据简化为一维。现在，如果我们看看基于欧几里德度量的 1-D 映射，我们看到对于相距很远的点(a & b)映射得很差。只有可以近似位于线性流形(c & d)上的点才能给出满意的结果。另一方面，看看测地线距离的映射，它很好地将近点近似为邻居，将远点近似为远点。
图像中两点间的测地线距离用两点间的图形距离来近似。因此，欧几里得距离不应用于近似非线性流形中两点之间的距离，而测地线距离可以使用。

Isomap 使用上述原理来创建特征值分解的相似矩阵。不像其他非线性降维方法，如 LLE 和 LPP 只使用局部信息，isomap 使用局部信息来创建全局相似性矩阵。isomap 算法使用欧几里得度量来准备邻域图。然后，它通过使用图距离测量两点之间的最短路径来近似两点之间的测地线距离。因此，在低维嵌入中，它近似数据集的全局和局部结构。

让我们对实现 Isomap 算法所需的一些概念有一个基本的了解。
**Pregel API**——Pregel 是 Google 开发的用于处理大规模图形的分布式编程模型。它是 Apache giraph 项目和 spark 的 GraphX 库背后的灵感。Pregel 基本上是一个消息传递接口，它基于一个顶点的状态应该依赖于它的邻居的想法。预凝胶计算将图形和一组顶点状态作为输入。在称为超级步骤的每次迭代中，它处理在顶点接收的消息并更新顶点状态。之后，它决定它的哪个邻居应该在下一个超步骤接收该消息，以及该消息应该是什么。因此，消息沿着边传递，计算只发生在顶点。该图不是仅通过网络传递的消息。计算在最大迭代次数或没有消息要传递时停止。我们用一个简单的例子来理解一下。假设，我们需要找到下图中每个顶点的度数。下图显示了预凝胶模型的单次迭代。
![](img/7bb9ce229bc4d359c354701f9f39f581.png)
在初始化时，每个顶点的度数都是 0。我们可以发送一个空消息作为初始消息来开始计算。在超级步骤 1 结束时，每个顶点通过其每条边发送消息 1。在下一个超步骤中，每个顶点对收到的消息求和并更新其度数。

**经典 MDS** - Isomap 与 Torgerson 和 Gower 提出的原始多维标度算法密切相关。实际上，它是经典多维标度的扩展。经典的多维算法给出了降维问题的闭式解。经典 MDS 使用欧几里得距离作为相似性度量，而 isomap 使用测地线距离。古典 MDS 的舞步是

1.  从给定的 *X* 创建相异度的平方δ²(X)的矩阵。
2.  通过对相异矩阵 B = 0.5 *(J *δ²* J)进行双重定心来获得矩阵 *B*
3.  计算矩阵 B 的特征值分解，B[δ]= QλQ’
4.  选择具有 K 个最高特征值的 K 个特征向量。

**IsoMap 的步骤** :
Isomap 仅在最初的几个步骤上不同于传统的 MDS。它使用图形距离，而不是使用欧几里得度量来表示不相似性。Isomap 算法的步骤是:

1.  **邻域图**:从数据集创建邻域图和邻接矩阵。

2.  **相异度矩阵**:邻域搜索之后，我们将使用 spark 的 graphX 库来计算点之间的测地线距离。在创建我们的邻居网络时，我们必须确保生成的图是单个连通的部分。如果没有，那么我们的相似性矩阵将保持不完整，结果将是不一致的。我们需要迭代不同的邻域选择参数值来获得完全连通图。到目前为止，spark 还没有加权图的最短路径函数。我们必须执行它。下面的代码展示了一个使用 pregel 的最短路径算法，比如 graphX 的 api。代码来自 graphX lib 的未加权图的最短路径函数。函数 **addMaps** 和 **sendMessage** 已经修改为支持加权图。

    ```py
    def ShortestPath(Verts: RDD[(VertexId, imMap[Long, Double])], 
              Edges: RDD[Edge[Double]], landmarks: Seq[Long] = Seq()): 
                                                     Graph[imMap[Long,Double],Double] = {

         val g = Graph(Verts, Edges)

         type SPMap = Map[VertexId, Double]

         def makeMap(x: (VertexId, Double)*) = Map(x: _*)

         def incrementMap(spmap1: SPMap, spmap2: SPMap, d: Double): SPMap = {
                 spmap1.map { case (k, v) => 
                     if (v + d < spmap2.getOrElse(k, Double.MaxValue)) k -> (v + d)
                     else -1L -> 0.0

                 }

             }

         def addMaps(spmap1: SPMap, spmap2: SPMap): SPMap = {
             (spmap1.keySet ++ spmap2.keySet).map {
                  k => k -> math.min(spmap1.
                   getOrElse(k, Double.MaxValue), 
                   spmap2.getOrElse(k, 
                   Double.MaxValue))
                  }(collection.breakOut)
              }

         var spGraph: Graph[imMap[Long,Double],Double]  = null

         if (landmarks.isEmpty){
             spGraph = g.mapVertices { (vid, attr) => makeMap(vid -> 0)}
         }
         else{
             spGraph = g.mapVertices { (vid, attr) => 
                      if (landmarks.contains(vid)) makeMap(vid -> 0) else makeMap()}
             }                                      

         val initialMessage = makeMap()

         def vertexProgram(id: VertexId, attr: SPMap, msg: SPMap): SPMap = {
                 addMaps(attr, msg)
         }

         def sendMessage(edge: EdgeTriplet[SPMap, Double]): 
                Iterator[(VertexId, SPMap)] = {

             val newAttr = incrementMap(edge.srcAttr, edge.dstAttr, edge.attr) - (-1)

             if (!newAttr.isEmpty) Iterator((edge.dstId, newAttr))
         else Iterator.empty

         }

         val h = Pregel(spGraph, initialMessage)(vertexProgram, sendMessage, addMaps)

         return(h)
     } 
    ```

**特征值分解**:在特征值分解之前，我们要对距离进行平方，并对平方后的相似度矩阵进行双居中。在特征值分解之后，选择具有 K 个最高特征值的前 K 个特征向量。

isomap 降维后 MNIST 数据集子集的图。
![](img/12fa350b48aeaaa28d7e9a02e249f903.png)

我们实现的是普通版本的 isomap。这需要大量的时间和计算能力。它有两个瓶颈:第一，相异矩阵的计算需要 O(N² 次运算，其中 **N** 是样本的数量；第二，成对图形距离的计算。如果 **N** 很大，这在大数据集的情况下通常是正确的，这就变得不切实际了。这个问题的解决方案是地标 Isomap。地标 isomap 基于地标 MDS。地标 MDS 选择一组称为地标的点，并在其上实现经典 MDS。基于从经典 MDS 获得的映射，使用基于距离的三角测量在低维嵌入中映射剩余的点。
地标经典缩放的步骤

1.  选择地标点 X[地标点]
2.  对标志点应用经典 MDS，得到低维插值 L[k]
3.  计算δ[u] 其中δ[ui] 是标志点相异矩阵的第 i[行]的平均值。
4.  给定向量 x[a] 计算δ[a] 其中δ[ai] 是点 x[a] 和标志点 *i* 之间的平方距离
5.  对 x[a] 的低维嵌入由 y[a]= 0.5 * L^(-1)-6】k(δ[a]-δ[u]给出其中 L ^(-1)[k] 是 L[k]
    的 penrose moore 逆选择标志点可以是随机的或者通过特定的方法。为了获得 K 维嵌入，至少需要 K+1 个标志点。出于与算法稳定性相关的原因，所选择的标志点的数量应该多于严格的最小值。地标 isomap 中等距映射的准确性不会由于算法中的近似而受到太大影响。

**Isomap 的缺点**:当流形没有被很好地采样并且包含孔洞时，Isomap 的性能很差。如前所述，邻域图的创建很复杂，稍有错误的参数就会产生不好的结果。

**结论**:在本文中，我们讨论了另一种流形学习算法 IsoMap(等距映射)。
在帖子的开头，我们谈到了什么是等距映射，以及它与其他降维算法有何不同。然后，我们简单讨论了一下 pregel API。后来，我们使用 spark 的 GraphX 库在 scala 中实现了一个 isomap 算法。对于希望深入研究分布式图形处理的人来说，GraphX 是一个很好的起点。
本系列的下一篇文章将是关于[自动编码器](https://blog.paperspace.com/p/dcc54161-3255-416f-983f-380515e8c859/)