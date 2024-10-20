# 使用 JSON 的 facets 数据模型

> 原文：<https://www.algolia.com/blog/engineering/facets-data-model-of-json-records/>

什么是[刻面搜索](https://www.algolia.com/doc/guides/managing-results/refine-results/faceting/)？一个重要的搜索 UI 功能，用户可以单击方面值来过滤他们的搜索结果。

这是描述刻面和刻面搜索的技术和数据方面的三篇系列博文的第二篇。在第 2 部分中，我们研究了应用于不同方面用例的各种数据结构。

像许多前端 UX 模式一样，*刻面搜索*需要精确且裁剪良好的刻面 JSON 数据集。幸运的是，在 JSON 记录中有许多表示方面的方法。大多数分面很简单，只需要一个单词过滤器来描述产品或服务——苹果是一种水果，苹果是一个品牌。我们可以用一个属性来表示“苹果”:

```
{
    "name": "apple",
    "type": "fruit",
    "seasonal_categories": ["fruits->late-summer", "fruits->autumn"]
} 
```

或者

```
{
    "name": "iPhone 8 Gold Protection Screen",
    "brand": "Apple",
    "category": ["smartphone->protective-screens"]
} 
```

方面“类型”和“品牌”是一维属性。但是方面可以变得更加复杂:您可以将它们设置为层次结构或嵌套属性。在本文中，我们将看到不同的方面数据模型如何实现不同的前端功能，所有这些都是方面搜索和导航的核心:

## [](#building-a-facets-data-model-into-json-records)构建一个 facets 数据模型到 JSON 记录中

我们将讨论定义方面和过滤器的四种常见方法。每一种都有其用途和优点:

*   **简单**(“类型”:“果”)
*   **嵌套**(actors->((actor 1->name，actor1- > character_name)，(actor2 …)))
*   **等级制度**(“类别:食物- >水果- >季节”)
*   **标签**(标签:“戏剧”、“和朋友一起看的乐趣”、“必须再看一遍”)

### [](#simple-facets)简单刻面

最简单的方面是最重要的。它们用最少的关键词传达了一个物体的本质。衬衫不是短袖就是长袖。一部电影只有一个平均评分。但是简单的方面也可以有多个值。t 恤可以只有一种颜色，也可以是红色和蓝色的混合。一部电影可以是“浪漫科幻喜剧恶搞”。简单而准确的标签有助于将最好的商品带回给用户，给他们他们想要的东西。在我们上一篇关于[刻面搜索](https://www.algolia.com/blog/engineering/facets-and-faceted-search-making-every-attribute-count/)的文章中，我们谈到了刻面不仅用于过滤，也用于搜索。

使用 JSON，简单的方面很容易表示:

```
{
    "name": "bold t-shirt",
    "desc": “Be bold, wear a t-shirt with only one color”,
    "color": “white”,
    "sleeves": "short"
},
{
    "name": "Hippie Vest with Fringes",
    "desc": “Be hip: get back to where you belong, peace + love”,
    "color": ["red", “orange”, “yellow”, “green”, "blue"],
    "sleeves": "long"
} 
```

### [](#nesting-facets)嵌套刻面

更高级的方面包含嵌套信息。嵌套就是组织你的数据。数据结构讲述了一个项目的故事。让我们比较一个非嵌套的例子和不同种类的嵌套。

**简单值:**

```
{
   "name": "Brad Pitt"
} 
```

**简单嵌套:**

```
{
   "first_name": "Brad",
   "last_name": "Pitt"
} 
```

**嵌套有点复杂:**

```
{
    "actors": [
     {
        "first_name": "Brad",
        "last_name": "Pitt"
     },
     {
        "first_name": "Scarlett",
        "last_name": "Johanson"
     }
   ]
} 
```

*在本文中，我们使用 actors 来搜索作为方面的*和*，以通过特定 actors 过滤结果。

#### 何时使用嵌套面进行多面搜索

您并不总是需要结构化或嵌套您的数据。有了搜索，你主要关心的是*找到*记录，这都是关于内容，而不是结构。找到“布拉德·皮特”对简单或复杂的结构同样有效。

> 虽然构造 JSON 可以给给定的记录增加人类的清晰度，但是引擎可以处理简单或复杂的结构。结构化数据总是需要更多的思考和更多的时间来维护，所以在过度思考之前考虑它的价值和目的是很重要的。

但是，简单的嵌套使您能够预测搜索模式。例如，如果人们依靠姓氏来查找作者，那么创建“姓氏”属性是有意义的。

```
{
    "last_name": "King",
    "first_name": "Stephen"
},
{
    "last_name": "Shakespeare",
    "first_name": "William"
} 
```

更复杂的嵌套可以用于显示目的，而不仅仅是用于过滤或搜索。一个例子是显示你的物品的详细信息。让我们比较两个用例，一个需要简单的属性，另一个需要更复杂的嵌套。一个卖 DVD 的公司不一定需要过度考虑结构。他们只需要确保记录包含足够的必要信息，以便用户找到电影。下面是一个只有两个属性的可能结构:

```
{
    "title": "Avengers: Infinity War",
    "cast": ["Robert Downey Jr", "Chris Hemsworth", "Mark Ruffalo", "Chris Evans", "Scarlett Johansson"]
} 
```

另一方面，提供电影详细信息的网站，如 IMDB 或网飞，需要显示更多结构化信息:

```
{
  "title": "Iron Man",
  "cast": [
    {
      "first_name": "John",
      "last_name": "Smith",
      "birth_year": "1978",
      "birth_place": "New York City, New York"
    },
    {
      "first_name": "Robert",
      "last_name": "Downey Jr",
      "birth_year": "1965",
      "birth_place": "New York City, New York"
    }
  ]
} 
```

后一条记录包含屏幕信息框的数据。唯一需要注意的是，当你的索引包含复杂的嵌套时，你需要指示引擎搜索结构的哪一部分。例如，您需要将“cast.first_name”和“cast.last_name”指定为可搜索，将“cast.birth_year”指定为可搜索和可过滤。

### [](#hierarchies)等级制度

许多多面搜索体验为用户提供了浏览类别和子类别层次的能力，帮助他们缩小或扩大搜索范围。您可以使用层次结构来创建基于类别的菜单系统。虽然设计和维护复杂的类别需要时间，但如果处理得当，层次结构可以创建一个很好的多面搜索和导航体验。看看超市如何对其数据进行分类，以帮助人们找到正确的在线通道:

```
[
{
    "name": "lemon",
    "categories.lvl0": "products",
    "categories.lvl1": "products > fruits"
},
{
    "name": "tomato",
    "categories.level0": "products",
    "categories.level1": "products > summer",
    "categories.level2": "products > summer -> vegetables"
}
] 
```

书商也会这样做，将每个作者分成不同的类别，有些有多个层次:

```
{
    "name": "Ursula K. Le Guin",
    "categories": [
        "level0": "Books",
        "level1": ["Books > Science Fiction", "Books > Literature & Fiction"],
        "level2": ["Books > Science Fiction > Time Travel", "Books > Literature & Fiction > Literary"]
    ]
} 
```

### [](#tagging-user-created-or-ai-generated)标记-用户创建或人工智能生成

并非所有方面都需要由你和你的公司来管理。对于如何定义和分类你的产品和服务，用户有他们自己的想法。比如，你可能会把《绝命毒师》归类为电视剧和犯罪剧。但是用户会有其他想法。在这里，我们将所有用户标签分组到属性“_tags”下:

```
{
    "title": "Breaking Bad",
    "_tags": ["drugs", “father figure”, “cancer”, “midlife crisis”, “violent”, “not for kids” ]
} 
```

而且不仅仅是用户添加 facet 标签。越来越多的机器学习算法和其他形式的人工智能结合用户行为、产品描述、市场数据等，找到有助于标记商品的方法。例如，当用户搜索、点击和查看你的目录时， [AI 自动标记](https://observablehq.com/@justingosses/comparing-human-generated-vs-a-i-generated-keyword-tags-on-c)可以通过添加令人惊讶的替代品来改善你的产品分类。你可以得到年龄组，目的，和其他你没有想到的产品特征。

一个流行的例子是[图像自动标记](https://www.quora.com/What-is-image-tagging)，其中图像检测算法为图像生成一组公共标记。像 Pinterest 这样提供 gif 和图片的网站就利用了这种技术。

想象一下，用户总是在每个重大节日前购买某些商品。人工智能生成的标签可能会生成以下内容。

```
{
    "name": "Scotch tape",
    "_tags": ["Christmas", “Halloween”, “New Years”]
} 
```

这些信息可以用来[推销和管理你的内容](https://www.algolia.com/doc/guides/managing-results/rules/merchandising-and-promoting/)，比如确保“透明胶带”在节日期间出现在搜索结果的前列。

此外，您可以将标记作为方面转换到记录中。这里，我们分析了标签并创建了一个新的方面“holiday_accessory”:

```
{
    "name": "Scotch tape",
    "holiday_accessory": true
} 
```

## [](#leveraging-your-multi-purposed-facets-data-model)利用您的多用途刻面数据模型

正如我们在本系列关于刻面搜索的[上一篇文章](https://www.algolia.com/blog/engineering/facets-and-faceted-search-making-every-attribute-count/)中所描述的，刻面对于搜索内容非常有用。拥有与用户输入的关键字相匹配的方面可以显著改善您的搜索。在功能层面上，引擎实际上并不关心它是在搜索一个类别、一个简单的方面还是一个非方面。它看着它被告知要看的地方。因此，指导您的引擎查看所需的特定属性非常重要。

考虑到这一点，我们讨论了[复杂的嵌套结构](#nesting-facets)如何返回关于记录中某个项目的故事的结果。亚马逊返回的结果不仅包括产品名称、描述、价格和图片。其结果还包括评级，航运信息，受欢迎程度和类别。

> 这里的秘密是将所有额外的信息放入一个可搜索的索引中。您可以重新调整每个属性的用途，将其用于搜索、过滤和/或显示目的。这避免了再次访问数据库的另一部分来检索附加信息。

我们将在这个分面搜索系列的下一篇文章中对此进行更多的讨论。

## [](#parting-words-on-the-facets-data-model)临别赠言刻面数据模型

简单和复杂的方面创建了一个[基于浏览的多方面搜索和导航体验](https://www.algolia.com/doc/guides/solutions/ecommerce/filtering-and-navigation/)——有或没有搜索。你可以以多种方式显示小平面——在屏幕的左侧、前方和中央，作为[下拉菜单](https://www.algolia.com/doc/guides/building-search-ui/ui-and-ux-patterns/facet-dropdown/js/)，菜单，以及许多其他创造性的方式，使用户只需单击鼠标就可以轻松浏览。

但是无论它们出现在屏幕上的什么地方，无论您在数据中如何设计它们，方面都使您的用户能够单击和浏览，并最终发现您的产品和服务的完整在线目录。这里的技术要点是，底层的 JSON 结构必须裁剪得很好，以支持您的前端需求。本文希望为任何部门或领域中的常见用例提供一个坚实的基础。

* * *

这是我们的 Facets & Data 系列的第二篇文章。本系列的重点是技术性的，概述了方面搜索的逻辑和方面数据模型。

*   第一篇文章是[刻面和刻面搜索，每个 JSON 属性都很重要](https://www.algolia.com/blog/engineering/facets-and-faceted-search-making-every-attribute-count/)——我们在这里定义了什么是刻面，并解释了刻面在结构化数据中的关键作用。它还说明了 JSON 是表示包括方面在内的索引数据的最灵活的方式。
*   在本文(第二篇文章)中，我们介绍了刻面最常见的数据结构:简单的刻面值、嵌套刻面、层次类别以及用户和人工智能标记，所有这些都用于刻面搜索的不同方面。
*   第三篇文章— [用动态分面实现分面搜索](https://www.algolia.com/blog/engineering/implementing-faceted-search-with-dynamic-faceting-with-code/) —数据仍然是我们关注的焦点，但我们也讨论过程。我们看一下查询搜索过程，从查询到执行再到响应，并展示如何动态生成方面。