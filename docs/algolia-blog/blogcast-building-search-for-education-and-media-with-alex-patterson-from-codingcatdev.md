# Blogcast:使用 Alex Patterson coding catdev 搜索教育/媒体

> 原文：<https://www.algolia.com/blog/engineering/blogcast-building-search-for-education-and-media-with-alex-patterson-from-codingcatdev/>

Alex Patterson 是 CodingCatDev 的创始人兼首席执行官，这是一个在线学习平台，面向希望了解不同技术的新手和有经验的开发者。Alex 还制作播客和视频，采访和指导全球 50 多名工程师。他也是云架构师，构建无头架构。

我们在 Jamstack Conf 2021 上遇到了 Alex。他加入了我们的展位，并介绍自己是 Algolia 的长期用户。我们很想了解更多，所以第二天我联系了他，他同意和我坐在一起讨论他的项目和 Algolia 解决方案。我们开始我们的博客，讨论为什么他选择了三个独特的领域的职业生涯——教育、播客和工程。然后我们进入他的技术选择，他给了我们一个他的 Algolia 实现的代码眼视图。我们结束了对他的解决方案的采访，并提出了他可以改善整体搜索体验的几种方法。

## [](#tell-us-about-yourself-and-your-online-business)告诉我们关于你自己和你的网上生意

我的主要背景是 web 开发，对前端开发有浓厚的兴趣。我还花了 10 多年时间作为 SAP ABAP 开发人员，专门研究 API 的 RFC。最近几年，很难远离云超伸缩性，所以我转向了解决方案架构师背景，专攻微服务。

我设计无头架构并构建企业应用程序。在我的网站上，我使用了许多用于数据存储、CMS、电子邮件、认证、搜索的无头服务——几乎所有的服务。我也建立自己的服务。Headless 允许我在不改变基于服务的中间层和后层的情况下改变底层技术(比如从 Gatsby 到 NextJS)。

## [](#how-did-you-get-into-education)你是怎么进入教育的？

我开始为一个非盈利组织编写代码，多年来我发现我想教授和回报社区。我也领导当地的开发小组。寻找合适的教育空间一直是一个挑战。这就是为什么我建立了我的网站，一个免费和基于订阅的在线学习平台。在某种程度上，我在弥补我在自己的科技教育中所缺失的东西。

我的教学工具是教程、完整课程和播客。播客是一个伟大的教育和培养工具。我和我的同事与专家交谈，深入研究各种各样的技术。我们还利用播客作为指导的机会。例如，我们采访了一位现在是 Google Dev 倡导者的油炸厨师，以及一位已经成为非常熟练的技术教育家的高中教师。

## [](#what-role-does-search-play-in-online-learning)搜索在在线学习中起到什么作用？

对我来说，搜索可能是最重要的部分。人们来到我们的平台，希望学习一些具体的东西，所以他们去我们的搜索栏找到他们的技术。就这么简单:搜索对于我们的学生如何参与我们的内容至关重要。

我们提供各种各样的内容，需要一个好的搜索栏。搜索让我们的访问者接触到这种多样性，并帮助他们建立自己的学习计划。我们有:

*   4 门完整课程，21 门教程
*   50 个播客/视频广播
*   38 博客

搜索向他们显示哪些 *材料组合* 他们可以用来继续他们的学业。这也有助于他们发现我们在每个播客末尾添加的 *完美精选、* 。

## [](#before-getting-into-your-algolia-implementation-tell-us-more-about-your-website%e2%80%99s-tech)

我开始 [CodingCat。Dev](https://codingcat.dev/) 与 Angular，Hugo，然后是 Gatsby，现在是 Next.js。我真的发现职业需要转向 React 背景，而当时 Gatsby 没有一个很好的混合渲染要求(SSR，SSG，CSR…)的解决方案。我还听了谷歌空间内关于网络性能的一些精彩谈话，以及 chrome 团队如何直接与 Vercel 合作开发 Next.js，后者最近被正式命名为 Aurora。

我们使用 Algolia 已经 5 年了。在每一次技术变革中，Algolia 一直是我唯一的搜索引擎。我已经测试了竞争对手，但他们无法击败搜索的速度和启动运行的简单性。构建搜索引擎不需要编码，它是 100%即插即用的。

至于我们的无头服务，这里有一个完整的列表:

*   燃烧基地
*   Sanity.io
*   阿尔戈利亚
*   云媒体
*   Youtube /主播
*   social bee . io–广播和社交
*   插入工具的 jam stack
*   发送蓝色邮件
*   不和谐

## [](#tell-us-about-your-algolia-search-implementation)给我们介绍一下你们的 Algolia 搜索 实现

我是因为火基才发现阿尔戈利亚的。我在 Firebase Cloud Firestore 上存储所有东西(在每个站点迭代中)。Firestore 没有很好的全文搜索解决方案。当时，Firebase 对此有一个解决方案，并建议使用 Algolia。

Algolia 非常适合我，因为我的网站需要大量的数据。Algolia 有一个惊人的 JavaScript 库用于客户端集成。我不想花费大量的时间来设置搜索，我发现文档和预构建的 React 组件非常适合我要完成的任务。我研究了弹性和类型感，仍然发现 Algolia 是最简单的解决方案。

## [](#show-us-the-code)给我们看代码

Firebase 具有云功能，允许在 Firestore 数据库中的数据更新时“触发”。然后我可以轻松地使用`algoliasearch Node.js`。

```
export const onPostWriteSearch = functions.firestore
  .document('posts/{postId}')
  .onWrite((snap, context) => {
    console.log('Adding Data for', context.params.postId);
    const post = snap.after.data();

    if (!post) {
      console.log('post missing data');
      return;
    }

    if (post.type === 'lesson') {
      console.log('post type is lesson, skipping');
      return;
    }

    post.objectID = context.params.postId;

    // Write to the algolia index
    const index = client.initIndex(ALGOLIA_INDEX_NAME);
    return index.saveObject(cleanTimestamp(post));
  });

export const onPostDeleteSearch = functions.firestore
  .document('posts/{postId}')
  .onDelete((snap, context) => {
    // Write to the algolia index
    console.log('Deleting Data for', context.params.postId);
    const index = client.initIndex(ALGOLIA_INDEX_NAME);
    return index.deleteObject(context.params.postId);
  });

```

## [](#next-steps)下一步

谢谢你，亚历克斯，聚在一起分享你的故事！

采访结束后，Alex 和我谈到了他的 Algolia 解决方案。类似于 Alex 对他的播客所做的，我们喜欢帮助我们的客人改进他们的搜索。随着他的业务增长，他渴望改进他的搜索功能，所以他很高兴看到他还可以用 Algolia 做些什么。以下是我提出的一些建议。它们包括给他的搜索结果增加更多种类(通过联合搜索)，并开始增加更多个性化的结果和浏览功能。

### [](#relevance)关联

我们深入研究了他的数据和索引配置。我们添加了新的可搜索属性，并将他的排名从“日期排序”改为“相关性排序”。然后，我们添加了一个自定义的流行度排名，并创建了一个副本，允许用户切换到“按日期排序”。

### [](#autocomplete-federated-search)自动完成/联合搜索

Alex 很惊讶地得知我们有一个前端插件，可以显示来自不同数据源的多个结果。这意味着当他的用户键入他们的查询时，他可以将他的教程、课程和播客显示为三个不同的结果集。我们的自动完成插件只需要各种索引的凭证和名称。他可以用自己的 CSS 来标记它，当然，他可以克隆这个库，并根据自己的需要进行定制。

### [](#full-text-searching)全文搜索

我们讨论了 Algolia 如何对大型文本执行全文搜索，这需要不同的索引策略。策略是将文本分成更小的块，每个块一条记录。每个组块通常是一个段落。这极大地增强了文本内搜索。我给他发了一篇关于[搜索大文本](https://www.algolia.com/developers-tech-blog/code-and-deep-dives/Building-search-for-technical-documentation-the-laravel-example)的文章，是我们 CTO 写的，还有这个教程。

### [](#indexing-updates)索引更新

Alex 面临的一大挑战是在不影响发布日期的情况下继续动态构建他的索引。例如，现在当他有一个处于“草稿”状态的帖子，或者一个未来的“发布日期”时，他的代码不会执行更新来删除这些帖子。我们向他展示了删除或仅更新某些属性的`DeleteObjects`和`PartialUpdates`方法。解决方案的另一部分是管理 Firebase 事件和更新。

```
// Configure this to match an existing Algolia index name
  const algoliaIndex = algolia.initIndex(algoliaConfig.index);
  try {
    let result;
    if (req.body?.deleted) {
      // remove if doc removed
      result = await algoliaIndex.deleteObject(req.body?.deleted);
    } else if (req.body?.updated && !publish) {
      // remove if not published after update
      result = await algoliaIndex.deleteObject(req.body?.updated);
    } else if (publish) {
      // create when in publish state
      result = await algoliaIndex.saveObject(req.body);
    }
    console.log(result ? `Result: ${JSON.stringify(result)}` : 'Nothing Done.');
    res.status(200).end();
  } catch (error) {
    console.log(error);
    res.json({ message: 'Error check logs' });
    res.status(500).end();
  }

```

### [](#algolia-talksearch-app)Algolia talk search app

又一个大赢家。由于 Alex 在 Youtube 上有很多视频和播客内容，我们建议他使用我们的 TalkSearch 工具，该工具可以抓取 YouTube 元数据及其转录或字幕，并建立一个用户可以搜索的 Algolia 索引。TalkSearch 允许他的访问者直接进入任何视频或播客的任何时刻。他对此真的很惊讶，想马上实施。除了 scraper，TalkSearch 还提供了一个前端解决方案，他可以根据自己的网站进行修改。

### [](#analytics-insights-recommendations-and-personalization)分析、洞察、建议和个性化

最后，Alex 立即看到了他如何利用分析和人工智能支持的推荐和个性化。Alex 可以使用分析来跟踪成功指标，改进他的内容，增加他的产品，并建立个性化的搜索/学习体验。Algolia 可以为学生建立一套建议，根据他们以前的考试成绩和教育选择，建议接下来的课程或相关课程。