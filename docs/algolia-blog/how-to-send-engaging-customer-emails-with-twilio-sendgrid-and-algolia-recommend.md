# 如何使用 Twilio SendGrid 和 Algolia 推荐发送引人入胜的客户电子邮件

> 原文：<https://www.algolia.com/blog/engineering/how-to-send-engaging-customer-emails-with-twilio-sendgrid-and-algolia-recommend/>

如果你要给你的客户发一封电子邮件，最好是相关的。产品推荐是为你的客户创建吸引人的电子邮件，邀请他们再次访问你的网站的好方法。 [Algolia 推荐](https://www.algolia.com/doc/guides/algolia-ai/recommend/?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)通过在客户网站分析上训练人工智能模型来建立个性化的产品推荐。在这篇博客中，你将看到如何构建一个后端服务，使用 [Twilio SendGrid](https://sendgrid.com/) 制作定制的推荐电子邮件并发送给你的客户。

## [](#what-were-building-with-twilio-and-algolia)我们正在与特维利奥和阿尔戈利亚一起建造什么

在这个演示中，我们将为一个电子商务网站构建一个参与式电子邮件服务。随着我们根据客户与我们网站的互动加入更多功能和分析，我们将探索更加定制化的推荐策略。

四种客户场景如下:

1.  他们浏览我们网站上的一个类别，但不购买
2.  他们购买一种产品，我们推荐他们经常购买的其他产品
3.  他们购买一种产品，我们推荐其他类似或相关的产品
4.  他们在我们的网站上有一个现有的用户配置文件

我们将依靠以下技术:

我们将分四步构建服务:

1.  设置 Algolia 搜索/推荐并配置 API
2.  检查用于生成推荐电子邮件的后端节点应用程序
3.  设置 Twilio SendGrid 并配置它们的 API
4.  运行服务发送电子邮件

## [](#the-setup)设置

我们将使用 node 将我们的服务构建为 express 应用程序。您可以[克隆解决方案库](https://github.com/algolia-samples/email-recommendations)来跟进。

```
git clone https://github.com/algolia-samples/email-recommendations

```

我们的服务遵循用 Node.js 编写的 express 应用程序的典型结构。

```
├── README.md
├── emails
│   ├── 1
│   │   ├── README.md
│   │   └── email.js
│   ├── 2
│   │   ├── README.md
│   │   └── email.js
│   ├── 3
│   │   ├── README.md
│   │   └── email.js
│   └── 4
│       ├── README.md
│       └── email.js
├── package-lock.json
├── package.json
├── sendgrid.js
├── server.js
└── templates
    ├── base.html
    ├── partials
    │   ├── products_3_columns.html
    │   └── products_full_width.html
    ├── post_order.html
    ├── pre_order.html
    └── re-engagement.html

```

`server.js`文件包含我们的路由、API 初始化和页面呈现。`templates/`目录是我们地狱犬的电子邮件模板。我们四封接洽电子邮件的实际内容保存在`emails/`目录中。我们在`sendgrid.js`管理发送电子邮件。

在我们继续之前，我们需要从我们的`package.json`安装我们的依赖项。

```
cd server/node
npm install

```

现在我们准备开始配置我们的服务所依赖的各种 API。我们将文件`.env.example`从这个项目的根目录复制到我们想要使用的服务器的目录中，并将其重命名为`.env`。例如，要使用节点实现:

```
cp .env.example server/node/.env

```

我们的`.env`文件有一些我们需要分配的键:

```
# Algolia credentials
# https://www.algolia.com/doc/guides/security/api-keys/#access-control-list-acl
ALGOLIA_APP_ID=
ALGOLIA_API_KEY=
ALGOLIA_INDEX_NAME=
ALGOLIA_RECOMMENDATION_API_KEY=

# SendGrid credentials
SENDGRID_API_KEY=
SENDGRID_FROM_EMAIL=

# Envs
```

```
STATIC_DIR=../../client

```

让我们从我们的数据和 Algolia 凭据开始。

## [](#setting-up-algolia)设置阿果

为了跟进，你需要在你现有的 Algolia 账户下建立一个索引，或者你可以[注册一个免费账户](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)并建立一个应用程序。对于这个博客中的例子，我们将使用一个现有的演示站点。

### [](#index-and-credentials)指标和凭证

本演示假设您已经有一个包含足够的[事件数据](https://www.algolia.com/doc/guides/sending-events/getting-started/#when-you-should-think-about-sending-events?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)的索引，可以使用 Algolia 推荐。如果你还没有索引或者没有足够的事件进行推荐，你仍然可以使用不需要经过训练的机器学习模型的策略。只需按照这些步骤[创建并填充一个索引](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)。

一旦我们有了索引，我们就可以在我们的`.env`文件中设置`ALGOLIA_INDEX_NAME`和`ALGOLIA_APP_ID`环境变量。我们还将把`ALGOLIA_API_KEY`设置为仅搜索的 API 键。我们可以在 Algolia 仪表盘的 [API Keys](https://www.algolia.com/users/sign_in) 页面上找到这些信息。

```
ALGOLIA_APP_ID=
ALGOLIA_API_KEY=
ALGOLIA_INDEX_NAME=

```

当我们在 **API 键**页面时，我们应该创建另一个键用于推荐。从 **API 键**页面，我们导航到**所有 API 键**选项卡并点击**新 API 键**，选择我们的索引，并将`recommendation`添加到我们的 ACL。现在我们可以将这个密钥添加到我们的`.env`文件中。

```
ALGOLIA_RECOMMENDATION_API_KEY=

```

### [](#verifying-our-models)验证我们的模型

一旦所有的 Algolia 产品都准备好了，让我们确保我们推荐的模特已经训练好并准备好了。我们点击 Algolia 仪表盘上的“推荐”灯泡图标，查看“推荐在哪里有效？”部分。如果模型还没有被训练，我们需要[训练模型](https://www.algolia.com/doc/guides/algolia-recommend/overview/)。

如果我们没有足够的事件来训练我们的模型，这也没关系。不需要 Algolia 推荐的模板我们还是可以用的。

## [](#setting-up-twilio-sendgrid)设置 Twilio 发送网格

我们使用 [Twilio SendGrid](https://docs.sendgrid.com/for-developers/sending-email/quickstart-nodejs) 作为电子邮件提供商来发送我们的渲染预览。如果你还没有帐户，你需要注册 SendGrid，然后[创建并存储一个 SendGrid API 密匙](https://docs.sendgrid.com/for-developers/sending-email/quickstart-nodejs#create-and-store-a-sendgrid-api-key)，允许你发送电子邮件。您还需要[验证您的电子邮件域](https://docs.sendgrid.com/for-developers/sending-email/quickstart-nodejs#verify-your-sender-identity)或[单一发件人电子邮件](https://docs.sendgrid.com/for-developers/sending-email/quickstart-nodejs#verify-your-sender-identity)，然后才能发送出站邮件。

将您的`SENDGRID_API_KEY`和已验证的`SENDGRID_FROM_EMAIL`添加到`.env`文件中:

```
SENDGRID_API_KEY=
SENDGRID_FROM_EMAIL=

```

## [](#generating-engagement-emails)生成订婚邮件

配置好所有的 API 后，我们就可以从`server.js`开始研究服务器代码了。在这里，我们设置我们的 express 服务器并初始化我们所有的 API 客户机:

```
// Setup Express, markdown renderer and Algolia clients.
const app = express();
const converter = new showdown.Converter();
const algoliaClient = algoliaearch(process.env.ALGOLIA_APP_ID, process.env.ALGOLIA_API_KEY)
const recommendClient = algoliarecommend(process.env.ALGOLIA_APP_ID, process.env.ALGOLIA_API_KEY);
const algoliaIndex = algoliaClient.initIndex(process.env.ALGOLIA_INDEX_NAME);

// Setup email delivery
sendgrid.setApiKey(process.env.SENDGRID_API_KEY);

```

在这个演示中，我们交互地生成和发送电子邮件，所以我们希望为我们的客户端应用程序提供服务。

```
// Home page
app.get("/", (req, res) => {
    const indexPath = path.resolve(process.env.STATIC_DIR + "/index.html");
    res.sendFile(indexPath);
});

```

我们的每个参与场景都有一个电子邮件模板和一个描述场景的`README.md`。客户端从`emails/`目录中加载四个模板，并将它们放在一个列表中，以显示在客户端应用程序中。

```
app.use(express.static(process.env.STATIC_DIR));
app.use(express.json());

const loadEmails = async () => {
    const EmailBaseDir = "./emails";
    const emailDirs = await fs.promises.readdir(EmailBaseDir);
    const emails = [];
    for(const emailDir of emailDirs) {
        const emailDirPath = path.join(EmailBaseDir, emailDir);
        const explanationFile = await fs.promises.readFile(path.join(emailDirPath, "README.md"), "utf8");
        const explanation = converter.makeHtml(explanationFile);
        const { email } = await import(`./${path.join(emailDirPath, "email.js")}`);
        emails.push({
            ...email,
            explanation,
        });
    };
    return emails;
};

```

从下拉列表中选择一个场景后，客户端使用 [nunjucks](https://mozilla.github.io/nunjucks/) 、来自`templates/`目录的适当的 [Cerberus](https://tedgoas.github.io/Cerberus/) 电子邮件模板以及为该场景生成建议所需的底层 API 调用来呈现所选电子邮件。

如果我们喜欢呈现的电子邮件的外观，我们可以使用 SendGrid 将它发送给自己。我们在**发送此电子邮件**字段中键入我们的电子邮件地址，然后点击**发送**。我们的服务会将呈现的电子邮件传递给 SendGrid API，并将其邮寄给我们！

现在我们已经了解了一般模式，让我们深入研究如何为四个场景中的每一个构建建议。

### [](#scenario-1-a-customer-browses-a-category)场景 1:一个客户浏览一个类别

在这种情况下，客户浏览了您网站上的一个类别，但没有购买任何东西。我们不太了解客户或他们正在寻找的具体商品。我们使用 [Algolia 的刻面](https://www.algolia.com/doc/guides/managing-results/refine-results/faceting/?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)推荐该类别中**评价最高的**产品。

如果客户浏览男士 t 恤，我们调用 Algolia 搜索 API，并使用索引的排名标准请求该类别的前六个项目。

```
    async (algoliaIndex, _) => {
      // The customer browsed the Men / Tee-shirt category
      const facetFilters = ["category:Men - T-Shirts"];
      // Get the top-rated products from the category
      const { hits } = await algoliaIndex.search("", { facetFilters, hitsPerPage: 6 });

```

我们使用 SendGrind 将这些建议通过电子邮件发送给客户，并提供返回我们网站的链接，以便客户继续浏览。

当我们没有足够的关于客户或他们正在寻找的特定产品的信息时，这是一个很好的后备策略。它不需要事件数据，只依赖于索引本身的类别信息。

### [](#scenario-2-recommend-products-frequently-bought-together-with-the-one-our-customer-just-purchased)场景 2:推荐与客户刚刚购买的产品“经常一起购买”的产品

在这个场景中，客户从我们的网站购买了一个产品。我们会推荐其他顾客经常购买的产品。我们将使用 [Algolia 推荐](https://www.algolia.com/doc/guides/algolia-recommend/overview/)来训练一个基于客户购买事件数据的机器学习算法。

有了已经训练好的模型，我们只需要几行代码就可以从当前产品获得基于我们网站上其他客户购买历史的推荐产品列表。

```
      const orderContent = ["D07940-6931-990"];
      // Get the similar items for this item.
      const { results: [ { hits }, ]} = await recommendClient.getRecommendations([{
        indexName: process.env.ALGOLIA_INDEX_NAME,
        objectID: orderContent[0],
        model: "bought-together",
        maxRecommendations: 3,
      }]);

```

同样，我们使用 SendGrind 将这些建议通过电子邮件发送给客户，并提供返回我们网站的链接，以便客户继续浏览。

### [](#scenario-3-recommend-products-related-to-the-one-our-customer-just-purchased)场景 3:推荐与客户刚刚购买的产品相关的产品

这类似于场景二。客户刚刚从我们的网站购买了一件产品。这次我们将使用这些信息来推荐与他们购买的产品相关的产品。这次我们将使用 Algolia 推荐的`related-products`型号。

与前面的场景一样，由于我们的模型已经训练好了，我们只需要几行代码就可以从当前产品获得基于我们网站上其他客户购买历史的推荐产品列表。

```
      const orderContent = ["D07940-6931-990"];
      // Get the related product for the bought product.
      const { results: [ { hits }, ]} = await recommendClient.getRecommendations([{
        indexName: process.env.ALGOLIA_INDEX_NAME,
        objectID: orderContent[0],
        model: "related-products",
        maxRecommendations: 3,
        fallbackParameters: {facetFilters:["category:Women - Jeans"]}
      }]);

```

注意这里我们包括了一个回退参数。如果模型中没有足够的推荐，我们可以用这个类别中的顶级项目填充我们的推荐列表。

现在，我们可以使用 SendGrind 将建议通过电子邮件发送给客户。

### [](#scenario-4-personalized-recommendations)场景四:个性化推荐

在最后一个场景中，一个客户以前访问过这个站点。当他们浏览我们的网站时，我们使用 [Algolia Insights](https://www.algolia.com/doc/guides/sending-events/getting-started/) 捕捉了他们的点击和转化行为。Algolia 利用这些事件为顾客建立了一个[个性化](https://www.algolia.com/doc/guides/personalization/what-is-personalization/)档案。我们将使用这个概要文件和推荐 API 来查找与客户的*相似性*相匹配的相关产品。

首先，我们使用我们的站点为该客户使用的唯一令牌从 [Algolia 个性化](https://www.algolia.com/doc/rest-api/personalization/#get-a-usertoken-profile)端点检索他们的个性化配置文件。(注意，我们使用推荐 API 凭证来检索这个概要文件。)

```
    async (algoliaIndex, recommendClient) => {
      // Get the customer's affinities profile
      // https://www.algolia.com/doc/rest-api/personalization/#get-a-usertoken-profile
      const userToken = 'user-1';
      const res = await fetch(
        `https://recommendation.eu.algolia.com/1/profiles/personalization/${userToken}`,
        {
          headers: {
            "X-Algolia-API-Key": process.env.ALGOLIA_RECOMMENDATION_API_KEY,
            "X-Algolia-Application-Id": process.env.ALGOLIA_APP_ID
          }
        }
      );
      const { scores } = await res.json();

```

响应是 JSON 格式的客户概要文件。它看起来会像这样:

```
{
  "userToken": "user_1",
  "lastEventAt": "2019-07-12T10:03:37Z",
  "scores": {
    "category": {
      "Women - Jeans": 1,
      "Men - T-Shirts": 10
    },
    "location": {
      "US": 6
    }
  }
}

```

该档案包括我们网站上按相似性得分排列的类别列表。我们将把排名靠前的类别转化为用户可以用来搜索产品的方面。

```
      // Transform the top customer affinities into a list of `facetFilters`
      let facetFilters = [];
      for (const key in scores) {
        const score = scores[key];
        let values = [];
        for (const valueKey in score) {
          values.push([ valueKey, score[valueKey] ]);
        };
        values.sort((kv1, kv2) => kv2[1] - kv1[1]);
        facetFilters.push(`${[key]}:${values[0][0]}`);
      };

```

Algolia Recommend 基于单个产品而非类别提供推荐。我们将从我们知道顾客喜欢的每个方面检索顶级产品，然后将它们发送到 Algolia Recommend 以找到类似产品。

```

      // Get the top-rated products matching the top-scoring customer profile facets
      const { hits: [topHit,] } = await algoliaIndex.search("", { facetFilters, hitsPerPage: 6 });
      // Get the related items matching the top-rated product
      const { results: [ { hits }, ]} = await recommendClient.getRecommendations([{
        indexName: process.env.ALGOLIA_INDEX_NAME,
        objectID: topHit.objectID,
        model: "related-products",
        maxRecommendations: 3
      }]);

```

现在，通过将我们对客户习惯的了解与我们网站上其他客户的购买行为相结合，我们获得了引人入胜的结果。

我们再一次将链接添加回我们的站点，并使用 SendGrid 发送电子邮件。

## [](#conclusion)结论

你可以根据对你的产品和用户行为的了解，生成吸引人的客户推荐邮件。像 [Algolia 推荐的那些机器学习模型](https://www.algolia.com/doc/guides/algolia-recommend/overview/)可以给你的电子邮件带来更多的相关性。现在你有了[工具](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-recommend&utm_id=sendgrid-recommend-emails)去建立你自己的推荐邮件服务！

想用推荐做更多事？看看我们使用 Twilio 的[代码块，或者更广泛地说，看看我们使用 Algolia 推荐](https://www.algolia.com/developers/code-exchange/?refinementList%5Bintegrations%5D%5B0%5D=Twilio&refinementList%5Bintegrations%5D%5B1%5D=Twilio%20SMS&page=1)的[代码块。在我们的开源代码交换平台上查看相关解决方案。](https://www.algolia.com/developers/code-exchange/?page=1&refinementList%5Bproduct_features%5D%5B0%5D=Recommend)

[![](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/?page=1&refinementList%5Bproduct_features%5D%5B0%5D=Recommend&refinementList%5Bintegrations%5D%5B0%5D=Sendgrid)