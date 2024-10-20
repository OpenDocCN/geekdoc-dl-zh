# Algolia+MongoDB–第 3 部分:数据管道实现

> 原文：<https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-data-pipeline-implementation/>

*我们邀请 Starschema 的朋友写一个结合使用 Algolia 和 MongoDB 的例子。我们希望您喜欢这个由全栈工程师 Soma Osvay 撰写的四部分系列。*

如果你想回顾或跳过，以下是其他链接:

[第 1 部分——用例、架构和当前挑战](https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-use-case-architecture-and-current-challenges/)

[第 2 部分–建议的解决方案和设计](https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-proposed-solution-and-design/)

[第 4 部分-前端实施和结论](https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-frontend-implementation-and-conclusion/)

* * *

在我开始之前，有一点需要注意:你可以在这里跟随实现。

在上一篇文章中，我们分析了我们的数据管道架构，并留下了一个关于我们将如何运行 Python 脚本来加载 Algolia 索引的未决问题。有三种选择:

1.  编写嵌入 ETL 过程的 Python 脚本，同时更新 Algolia 索引和 MongoDB。
2.  托管 Python 脚本，完全独立于我们现有的 ETL 工作流，将数据从 Mongo 拉到 Algolia。
3.  在 MongoDB 更新之后，使用 MongoDB 触发器和函数来更新 Algolia 索引。

在与我们的工程团队讨论后，我决定选择第一个选项，因为我们已经有了一个成熟的方法来运行当前的数据准备管道，在将数据加载到数据库之前，我们会使用大量现有的脚本来清理、聚合和格式化数据。在这里添加一个额外的脚本不会花费太多精力，所有的维护和监控工具都是现成的。在决定了架构步骤之后，我决定编写一个脚本，既执行到 Algolia 的初始数据加载，又保持索引最新，而不是为每个动作编写一个脚本。

令人欣慰的是，Algolia 通过公开一个 [`replace_all_objects`方法](https://www.algolia.com/doc/api-reference/api-methods/replace-all-objects/)来支持这种用例，该方法实际上首先创建一个新的临时索引，然后在构建完成后将其与活动索引交换。这使得旧索引和刷新索引之间的转换几乎是即时的，没有任何停机时间或数据不一致。

## [](#step-0-planning)第 0 步。规划

在开始实现我的 Python 脚本之前，我必须[注册一个免费的 Algolia 帐户](https://www.algolia.com/users/sign_up)，并创建一个样本数据集，我可以使用 [MongoDB Atlas](https://www.mongodb.com/atlas/database) 来填充我的索引。

我选择使用 Atlas 自带的默认 AirBnB 数据集，因为其格式和用例与我的现实生活数据非常相似。我还公开托管了样本数据集，供跟踪或想要试验的任何人使用:

*   **主机** : `algolialistingstest.vswcm0y.mongodb.net`
*   **用户名** : `ReadOnly`
*   **密码** : `AlgoliaTest`
*   **数据库** : `sample_airbnb`
*   **收藏** : `listingsAndReviews`

我决定使用 Jupyter 笔记本来实现这个脚本，因为它让我能够独立地测试我的代码片段，用 Markdown 注释我的代码，反复试验和建模数据结构，并轻松地将创建的 Python 代码导出为脚本文件。它的功能非常多，交互性也很强，我通常很喜欢使用它。我把它放在 [Google Collab](https://colab.research.google.com/) 上，所以我可以很容易地分享代码，而不需要任何人安装一个内部 Jupyter 环境。你可以在这里找到实现的脚本。我们使用实现的脚本来:

1.  使用 [Algolia Python API](https://www.algolia.com/doc/api-client/getting-started/install/python/?client=python) 连接到 Algolia，并验证连接。
2.  连接到 MongoDB 实例并检索样本数据。
3.  准备阿尔戈利亚指数。
4.  将数据集从 MongoDB 实例加载到 Algolia 中，并替换现有的索引。

## [](#step-1-connect-to-algolia)第一步。连接到 Algolia

第一步是生成 API 密钥:

1.  [注册](https://www.algolia.com/users/sign_up)一个免费的 Algolia 账户，或者[登录](https://www.algolia.com/users/sign_in)您现有的账户。
2.  登录后，将自动为您创建一个 Algolia 应用程序。您可以使用默认的未命名应用程序，也可以创建一个新的应用程序。
3.  转到您的应用程序的 [API 密钥](https://www.algolia.com/users/sign_in)部分，检索您的**应用程序 ID** 和**管理 API 密钥**。从下面的 Python 代码连接您的 Algolia 帐户时，您将需要使用这两个工具。

我们需要先安装 Algolia Python 客户端，但之后，我们的连接代码看起来是这样的:

```
# The Application ID of your Algolia Application
algolia_app_id = "[your_algolia_app_id_here]"
# The Admin API Key of your Algolia Application
algolia_admin_key = "[your_algolia_admin_key_here]"

# Define the Algolia Client and Index that we will use for this test
from algoliasearch.search_client import SearchClient

algolia_client = SearchClient.create(algolia_app_id, algolia_admin_key)
algolia_index = algolia_client.init_index("test_index")

# Test the index that we just created. We do this as part of the function, because these variables are not needed later
def test_algolia_index(index):
    # Clear the index, in case it contains any records
    index.clear_objects()
    # Create a sample record
    record = {"objectID": 1, "name": "test_record"}
    # Save it to the index
    index.save_object(record).wait()
    # Search the index for 'test_record'
    search = index.search("test_record")
    # Clear all items again to clear our test record
    index.clear_objects()
    # Verify that the first hit is our object
    if len(search["hits"]) == 1 and search["hits"][0]["objectID"] == "1":
        print("Algolia index test successful")
    else:
        raise Exception("Algolia test failed")

# Call our test function
test_algolia_index(algolia_index) 
```

## [](#step-2-connect-to-mongo-and-get-data)第二步。连接到 Mongo 并获取数据

首先，安装 Python MongoDB 客户端 [PyMongo](https://www.mongodb.com/docs/drivers/pymongo/) ，然后使用[这段代码](https://colab.research.google.com/drive/1hO996af5PzI1piGdFlTyr1InhGJmUED0#scrollTo=kbzXuPteDSWn)连接到我们的示例 MongoDB 数据库并读取示例数据。请注意，我们只获得 5000 个项目，这样我们就不会超出我们的免费层使用量:

```
# Define MongoDB connection parameters. These are wrapped in a function to keep the global namespace clean
# Change these values if you are not running your own MongoDB instance
db_host = "algolialistingstest.vswcm0y.mongodb.net"
db_name = "sample_airbnb"
db_user = "ReadOnly"
db_password = "AlgoliaTest"
collection_name = "listingsAndReviews"

connection_string = f"mongodb+srv://{db_user}:{db_password}@{db_host}/{db_name}?retryWrites=true&w=majority"

# Connect to MongoDB and get the MongoDB Database and Collection instances
from pymongo import MongoClient

# Create MongoDB Client
mongo_client = MongoClient(connection_string)
# Get database instance
mongo_database = mongo_client[db_name]
# Get collection instance
mongo_collection = mongo_database[collection_name]
# Retrieve the first 5000 records from collection items
mongo_query = mongo_collection.find()
initial_items = []
for item in mongo_query:
    if len(initial_items) < 5000:
        initial_items.append(item) 
```

## [](#step-3-transform-our-data-into-a-form-that-suits-algolia)**第三步。将我们的数据转换成适合 Algolia 的形式**

我们的 MongoDB 样本数据集中的对象包含许多属性，其中一些与我们的 Algolia 索引无关。我们只保留搜索或排名所需的内容。

*   属性将被保留，因为它也将是 Algolia 对象 ID。
*   这些属性将被保留用于搜索、刻面或显示:`name`、`space`、`description`、`neighborhood_overview`、`transit`、`property_type`、`address`、`accommodates`、`bedrooms`、`beds`、`number_of_reviews`、`bathrooms`、`price`、`weekly_price`、`security_deposit`、`cleaning_fee`、`images`。
*   Airbnb 条目上的`review_scores`属性将被转换为 scores 属性，该属性将包含给予该列表的星级数。
*   一个`_geoloc`属性将根据原始地址对象中的字段添加到对象中。这将用于地理搜索。
*   以下属性由于 Algolia 不需要，将被完全剥离:`summary`、`listings_url`、`notes`、`access`、`interaction`、`house_rules`、`room_type`、`bed_type`、`minimum_nights`、`maximum_nights`、`cancellation_policy`、`last_scraped`、`calendar_last_scraped`、`first_review`、`last_review`、`amenities`、`extra_people`、`guests_included`、`host`、`availability`、`review_scores`、`reviews`。

这里是[这个转换代码](https://colab.research.google.com/drive/1hO996af5PzI1piGdFlTyr1InhGJmUED0#scrollTo=96FGwTeVErxx):

```
# We define a function first that is able to strip long texts longer than 350 characters. This is done because the sample data has some records with very long descriptions, which is irrelevant to our use-case and takes up a lot of space to display
def strip_long_text(obj, trailWithDot):
    if isinstance(obj, str):
        # Strip texts longer than 350 characters after the next full stop (.)
        ret = obj[:350].rsplit(".", 1)[0]
        if trailWithDot and len(ret) > 0 and not ret.endswith("."):
            ret = "."
        return ret
    else:
        return obj

# We define a function to validate number values coming from MongoDB. MongoDB stores numbers in Decimal128 format, which is not accepted by Algolia (only as string). This function:
# 1\. Converts numbers to float from Decimal128
# 2\. Introduces a maximum value for these numbers, as some values in MongoDB are outliers and incorrectly filled out and it gives range filters an unreal max value.
def validate_number(num, maxValue):
    if num is None:
        return num
    else:
        val = float(str(num))
        if val > maxValue:
            return maxValue
        return val

def prepare_algolia_object(mongo_object):
    # Create an instance of the Algolia object to index, and set its objectID based on the _id of the mongo object
    r = {}
    r["objectID"] = mongo_object["_id"]
    # prepare the string attributes
    for string_property in [
        ["name", True],
        ["space", True],
        ["description", True],
        ["neighborhood_overview", True],
        ["transit", True],
        ["address", False],
        ["property_type", False],
    ]:
        if string_property[0] in mongo_object:
            r[string_property[0]] = strip_long_text(
                mongo_object[string_property[0]], string_property[1]
            )

    # prepare the integer properties
    for num_property in [
        ["accommodates", 100],
        ["bedrooms", 20],
        ["beds", 100],
        ["number_of_reviews", 1000000],
        ["bathrooms", 100],
        ["price", 1000],
        ["weekly_price", 1000],
        ["security_deposit", 1000],
        ["cleaning_fee", 1000],
    ]:
        if num_property[0] in mongo_object:
            r[num_property[0]] = validate_number(
                mongo_object[num_property[0]], num_property[1]
            )

    # prepare the Sortable attributes (except for scores rating)

    # set rating if any
    if (
        "review_scores" in mongo_object
        and "review_scores_rating" in mongo_object["review_scores"]
    ):
        stars = round(mongo_object["review_scores"]["review_scores_rating"] / 20, 0)
        r["scores"] = {
            "stars": stars,
            "has_one": stars >= 1,
            "has_two": stars >= 2,
            "has_three": stars >= 3,
            "has_four": stars >= 4,
            "has_five": stars >= 5,
        }
    # set images
    if "images" in mongo_object:
        r["images"] = mongo_object["images"]
    # set GeoLocation if any
    if "address" in mongo_object:
        if "location" in mongo_object["address"]:
            if mongo_object["address"]["location"]["type"] == "Point":
                r["_geoloc"] = {
                    "lng": mongo_object["address"]["location"]["coordinates"][0],
                    "lat": mongo_object["address"]["location"]["coordinates"][1],
                }
    return r 
```

## [](#step-4-define-our-index-properties)**第四步。定义我们的索引属性**

现在让我们告诉 Algolia 如何处理我们赋予它的属性。我们将设置`[attributesToRetrieve](<https://www.algolia.com/doc/api-reference/api-parameters/attributesToRetrieve/>)`，Algolia 将在我们的 UI 中显示的每个搜索结果返回的属性，为这些属性的数组:`summary`、`description`、`space`、`neighborhood`、`transit`、`address`、`number_of_reviews`、`scores`、`price`、`cleaning_fee`、`property_type`、`accommodates`、`bedrooms`、`beds`、`bathrooms`、`security_deposit`、`images/picture_url`、`_geoloc`。我们的`[attributesForFaceting](<https://www.algolia.com/doc/api-reference/api-parameters/attributesForFaceting/>)`数组将包含`property_type`、`address/country`、`scores/stars`、`price`和`cleaning_fee`。

我们还将设置`[searchableAttributes](<https://www.algolia.com/doc/api-reference/api-parameters/searchableAttributes/>)`，这是计算查询时要考虑的属性。Algolia 不会浪费时间在这个列表之外寻找潜在的搜索匹配，因此它加快了查询速度，并让我们从最高到最低设置优先级顺序:

1.  (最高优先级属性)`name`、`address/street`、`address/suburb`
2.  `address/market`，`address/country`
3.  `description`(这将是一个[无序属性](https://www.algolia.com/doc/api-reference/api-parameters/searchableAttributes/#modifiers))
4.  `space`(另一个无序属性)
5.  `neighborhood_overview`(另一个无序属性)
6.  (最低优先级)`transit`

我们还将更新索引的默认[排名逻辑](https://www.algolia.com/doc/api-reference/api-parameters/ranking/):

1.  (第一要务)`geo`–就近提供搜索结果是我们的首要任务
2.  `typo`
3.  `words`
4.  `filters`
5.  `proximity`
6.  `attribute`
7.  `exact`
8.  (最不优先)`custom`

我们还更新了我们的索引，以[忽略复数](https://www.algolia.com/doc/api-reference/api-parameters/ignorePlurals/)(你可能不会想太多，但是当它像你的用户不期望的那样工作时，你的用户肯定会想的)。你可以在[官方 Algolia API 参考页面](https://www.algolia.com/doc/api-reference/api-parameters/)上找到其他很棒的资源和设置。下面是我们为这个编写的[代码:](https://colab.research.google.com/drive/1hO996af5PzI1piGdFlTyr1InhGJmUED0#scrollTo=PGPsY-RyEw61)

```
algolia_index.set_settings(
    {
        "searchableAttributes": [
            "name,address.street,address.suburb",
            "address.market,address.country",
            "unordered(description)",
            "unordered(space)",
            "unordered(neighborhood_overview)",
            "transit",
        ],
        "attributesForFaceting": [
            "property_type",
            "searchable(address.country)",
            "scores.stars",
            "price",
            "cleaning_fee",
        ],
        "attributesToRetrieve": [
            "images.picture_url",
            "summary",
            "description",
            "space",
            "neighborhood",
            "transit",
            "address",
            "number_of_reviews",
            "scores",
            "price",
            "cleaning_fee",
            "property_type",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "security_deposit",
            "_geoloc",
        ],
        "ranking": [
            "geo",
            "typo",
            "words",
            "filters",
            "proximity",
            "attribute",
            "exact",
            "custom",
        ],
        "ignorePlurals": True,
    }
) 
```

## [](#step-5-load-the-dataset-into-algolia-from-mongodb)第五步。将数据集从 MongoDB 加载到 Algolia

[这段简短的代码](https://colab.research.google.com/drive/1hO996af5PzI1piGdFlTyr1InhGJmUED0#scrollTo=4dSgOI0-FNC4)将数据集加载到 Algolia 索引中，替换现有的索引，因此没有过时的记录。

```
# Prepare the Algolia objects
algolia_objects = list(map(prepare_algolia_object, initial_items))
algolia_index.replace_all_objects(algolia_objects, {"safe": True}).wait() 
```

## [](#script-evaluation-performance)剧本评价&表演

总的来说，我发现从 Python 加载 Algolia 索引是一项非常简单的任务，尽管我的 Python 技能有些生疏。实际上，我的大部分时间都花在了准备 AirBnB 列表对象上，并把它们转化成我在 Algolia 想要的东西。如果我使用我们自己的数据集，这可能会简单得多，因为不需要太多的转换。

我了解到 Algolia 公开了一个很棒的 Python API——它比我预期的更容易使用，并且包含很棒的[文档](https://www.algolia.com/doc/api-client/getting-started/install/python/?client=python),一步一步地指导我完成整个过程。准备和加载索引所需的代码很少，对我来说很直观。它在加载索引时表现也很好:加载和替换 5000 条记录的整个索引只需要不到 5 秒钟，即使是在资源有限的云托管服务器上运行。当我在我们的一些高速服务器上用快速互联网连接运行它时，只需要大约 2 秒钟。我们的生产数据集要大得多(大约 40k 条记录)，但是我们准备列表数据的标准管道每天都要运行一个多小时，所以我确信我们的整体性能不会受到 Algolia 的影响。到目前为止，它的简单性和速度远远超过了任何缺点。

【T2![](img/7665551c18b687f25dcadc15cb213b7d.png)

在本系列的第一篇文章[中，我谈到了我们的用例、架构以及我们面临的搜索挑战。](https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-use-case-architecture-and-current-challenges/)

在本系列的第二篇文章中，我介绍了 PoC 的设计规范，并讨论了实现的可能性。

在本系列的第四篇文章[中，我将实现一个示例前端，这样我们就可以从用户的角度评估产品，如果开发人员选择这个选项，就可以给他们一个先发制人的机会。](https://www.algolia.com/blog/engineering/supercharging-search-for-ecommerce-solutions-with-algolia-and-mongodb-frontend-implementation-and-conclusion/)