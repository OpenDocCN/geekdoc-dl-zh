# 面向机器学习专家的 Plotly Python 教程

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/plotly-python-tutorial-for-machine-learning-specialists>

Plotly 是一个开源的 Python 图形库,非常适合构建漂亮的交互式可视化。在深入研究机器学习建模之前，它是发现数据集中模式的一个非常棒的工具。在本文中，我们将看看如何以示例驱动的方式使用它。

您可能会看到的一些可视化效果包括:

*   线形图，
*   散点图，
*   条形图，
*   误差线，
*   箱线图，
*   直方图，
*   热图，
*   支线剧情，
*   和气泡图。

### 为什么你会选择 Plotly

现在，事实是你仍然可以使用 Matplotlib、Seaborn 或者 T2 的 Bokeh 来获得这些可视化效果。有几个原因可以解释为什么你会选择 Plotly:

*   可视化是交互式的，不像 Seaborn 和 Matplotlib
*   使用 Plotly 的高级 Express API 生成复杂的视觉效果相当简单；
*   Plotly 还提供了一个名为 [Plotly Dash](https://web.archive.org/web/20221206044847/https://plotly.com/dash/) 的框架，你可以用它来托管你的可视化以及机器学习项目；
*   你可以为你的可视化生成 HTML 代码，如果你喜欢，你可以把它嵌入到你的网站上。

也就是说，生成可视化效果需要清理数据集。这是一个至关重要的部分，否则，你会有视觉传达错误的信息。在本文中，我们跳过清理和预处理部分，将重点放在可视化上。我们将在教程结束时提供整个笔记本。

创建可视化效果时，记住最佳实践也很重要，例如:

*   使用对眼睛友好的颜色
*   确保数字相加，例如在饼图中，百分比总和应为 100%
*   使用正确的色标，这样观众就能清楚地看到哪种颜色代表较高的数字，哪种颜色代表较低的数字
*   不要在同一个视图中放置太多的数据，例如，可以分组并绘制最上面的项目，而不是绘制数据集中的所有内容
*   保证剧情不要太忙
*   总是添加你的数据的来源，即使你是收集它的人。它建立信誉。

我们可以通过两种方式与 Plotly API 进行交互；

在这篇文章中，我们将交替使用它们。

Plotly 直方图

## 直方图是数字数据分布的表示，数据被分组到多个条块中。然后显示每个箱的计数。在 Plotly 中，可以使用 sum 或 average 等聚合函数来聚合数据。在绘图中，要入库的数据也可以是分类的。这里有一个例子:

Plotly 条形图

```py
import plotly.express as px
fig = px.histogram(views, x="views")
fig.show()
```

当您想要显示分类列和数字列时，条形图是一种很好的可视化工具。它显示了每个类别中某个数字列的数量。Plotly Express 使绘制一个非常容易。

## 你不仅仅局限于垂直条形图，你也可以使用水平条形图。这是通过定义“方向”来实现的。

Plotly 饼图

```py
fig = px.bar(views_top, x='event', y='views')
fig.show()
```

饼图是显示每个类别中项目数量的另一种可视化类型。这种类型使用户能够快速确定特定项或值在整个数据集中所占的份额。这次让我们展示一下如何使用 Plotly 的 Graph 对象来绘制。

```py
fig = px.bar(views_top, x='views', y='event',orientation='h')
fig.show()

```

Plotly 圆环图

## 您可以通过指定`hole`参数将上面的视觉效果更改为圆环图。这是您希望环形图上的孔的大小。

Plotly 散点图

```py
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Pie(labels=labels, values=values)
    ])
fig.show()
```

散点图对于确定两个数值变量之间是否存在关系或相关性非常有用。

## Plotly 线图

折线图主要用于显示某一数值如何随时间或某一区间变化。

```py
fig = go.Figure(
    data=[
        go.Pie(labels=labels, values=values, hole=0.2)
    ])
fig.show()

```

Plotly 注释

## 在 Plotly 中添加文本标签和注释非常简单。在散点图中，这可以通过指定`text`参数来实现。

Plotly 3D 散点图

```py
fig = px.scatter(df,x='comments',y='views')
fig.show()

```

在 Plotly 中，可以通过传递 x、y 和 z 参数来创建 3D 散点图。

## Plotly 写入 HTML

Plotly 还允许您将任何可视化保存到 HTML 文件中。这非常容易做到。

```py
fig = px.line(talks, x="published_year", y="number_of_events")
fig.show()
```

Plotly 3D 曲面

## 现在让我们看看如何在 Plotly 中绘制 3D 曲面。类似于 3D 散点图，我们必须传递 x、y 和 z 参数。

Plotly 气泡图

```py
fig = px.scatter(df,x='comments',y='views',color='duration',text="published_day")
fig.show()
```

绘制气泡图非常类似于散点图。事实上，它是根据散点图构建的。我们增加的唯一一项是气泡的大小。

## 绘图表

Plotly 还可用于将数据框可视化为表格。我们可以使用 Plotly Graph Objects `Table`来实现这一点。我们将标题和单元格传递给表格。我们还可以指定如下所示的样式:

```py
fig = px.scatter_3d(df,x='comments',y='views',z='duration',color='views')
fig.show()
```

打印热图

## 我们可以用密度热图来形象化一个聚集函数的 2D 分布。聚合函数应用于 z 轴上的变量。该函数可以是总和、平均值甚至是计数。

情节动画

```py
fig.write_html("3d.html")
```

Plotly 动画可用于制作特定值随时间变化的动画。为了实现这一点，我们必须定义`animation_frame`。在这种情况下，是年份。

## 箱形图

箱线图显示了数据通过其四分位数的表示。落在第四个四分位数之外的值表示数据集中的异常值。

```py
fig = go.Figure(data=[go.Surface(z=df[['duration','views','comments']].values)])

fig.update_layout(title='3D Surface', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()
```

Plotly 地图

## 为了在 Plotly 中使用地图，你需要前往[地图框](https://web.archive.org/web/20221206044847/https://www.mapbox.com/)并获取你的地图框 API 密钥。有了手头的，您可以在地图上直观地显示您的数据。这是在传递纬度和经度时使用`scatter_mapbox`完成的。

情节复杂的次要情节

```py
fig = px.scatter(df,x='comments',y='views',size='duration',color='num_speaker', log_x=True, size_max=60)
fig.show()

```

使用 Plotly，我们还可以在同一个图形上可视化多个图。这是用情节主线完成的。通过定义一个`facet_col`来创建图形。图表将被分解成尽可能多的来自`facet_col`列的唯一值。

## Plotly 误差线

误差线用于显示可视化数据的可变性。一般来说，它们有助于显示估计误差或某一测量的精确度。误差线的长度揭示了不确定性的水平。误差线越长，表明数据点越分散，因此不确定性越大。它们可以应用于图表，如折线图、条形图和散点图。

```py
fig = go.Figure(data=[go.Table(header=dict(values=views_top.columns,
                                           fill_color='yellow',
),
                 cells=dict(values=[views_top['event'],views_top['views']],
                            fill_color='paleturquoise',
                           ))
                     ])
fig.show()
```

最后的想法

## 希望这篇文章向您展示了如何在下一个机器学习工作流程中使用 Plotly。你甚至可以用它来可视化你的机器学习模型的性能指标。与其他工具不同，它的视觉效果既吸引眼球又具有互动性。

交互性使您能够放大和缩小图表中的特定部分。这样，你可以看得更深一点，更详细地分析你的图表。具体来说，我们已经看到了如何在 Plotly 中使用流行的图表，如直方图、条形图和散点图。我们还看到，我们可以在同一个图形上构建多个图，并在地图上可视化数据。

```py
fig = px.density_heatmap(df, x="published_year", y="views",z="comments")
fig.show()

```

使用的笔记本可以在这里找到[。](https://web.archive.org/web/20221206044847/https://nbviewer.jupyter.org/github/mwitiderrick/Neptune-Plotly/blob/1ac120b3c131c5bd371bfe2f9e1dfc39d5752a44/plotly.ipynb)

## 快乐策划——没有双关语！

Plotly Animations can be used to animate the changes in certain values over time. In order to achieve that, one has to define the `animation_frame`. In this case, it’s the year.

```py
px.scatter(df, x="duration", y="comments",animation_frame="published_year", size="duration", color="published_day")
```

## Plotly box plot

A box plot shows the representation of data through their quartiles. Values falling outside the fourth quartile represent the outliers in your dataset.

```py
fig = px.box(df, x="published_day", y="duration")
fig.show()

```

## Plotly maps

In order to work with maps in Plotly, you will need to head over to [Mapbox](https://web.archive.org/web/20221206044847/https://www.mapbox.com/) and grab your Mapbox API key. With the at hand, you can visualize your data on a map in Plotly. This is done using the `scatter_mapbox` while passing the latitude and the longitude. 

```py
px.set_mapbox_access_token('YOURTOKEN')
fig = px.scatter_mapbox(df, lat="lat", lon="lon",
                        color="region",
                        size="views",
                  color_continuous_scale=
                        px.colors.cyclical.IceFire, size_max=15)
fig.show()
```

## Plotly subplots

With Plotly, we can also visualize multiple plots on the same graph. This is done using Plotly Subplots. The plots are created by defining a `facet_col`. The graphs will be broken into as many unique values as available from the `facet_col` column. 

```py
px.scatter(df, x="duration", y="comments",
           animation_frame="published_month", animation_group="event",
           facet_col="published_day",width=1500, height=500,
           size="views", color="published_day",
          )
```

## Plotly error bars

Error bars are used to show the variability of data in a visualization. Generally, they help in showing the estimated error or the preciseness of a certain measure. The length of the error bar reveals the level of uncertainty. Longer error bars indicate that the data points are more spread out hence more uncertain. They can be applied to graphs such as line charts, bar graphs, and scatterplots.

```py
fig =  go.Figure(
    data=[
        go.Bar(
    x=views_top['event'], y=views_top['views'],
    error_y=dict(type='data', array=views_top['error'].values)
)
    ])
fig.show()
```

## Final thoughts

Hopefully, this piece has shown you how you can use Plotly in your next machine learning workflow. You can even use it to visualize the performance metrics of your machine learning models. Unlike other tools, its visuals are eye-catching as well as interactive. 

The interactivity enables you to zoom in and out of specific parts in the graph. In this way, you can look a little deeper to analyze your graph in more detail. Specifically, we have seen how you can use popular graphs such as histograms, bar charts, and scatter plots in Plotly. We have also seen that we can build multiple plots on the same graph as well as visualize data on the map. 

The Notebook used can be found [here](https://web.archive.org/web/20221206044847/https://nbviewer.jupyter.org/github/mwitiderrick/Neptune-Plotly/blob/1ac120b3c131c5bd371bfe2f9e1dfc39d5752a44/plotly.ipynb). 

Happy plotting – no pun intended!