# 使用 CSV 文件、MongoDB 或 GraphQL 建立新冠肺炎地理搜索索引

> 原文：<https://www.algolia.com/blog/engineering/building-a-covid-19-geosearch-index-using-csv-files-mongodb-or-graphql/>

我的孩子们最近回到了学校，这意味着新冠肺炎再次在我的脑海中鲜活起来。我认为建立一个新冠肺炎地理搜索网站来探索使用地图界面的病例计数可能会很有趣。第一步是选择数据源并构建索引。从结构良好的数据中创建托管索引对于 Algolia 如何快速准确地返回搜索结果至关重要。创建一个索引也可能是我们作为开发人员试图启动和运行搜索的第一个挑战。

我想为我的网站建立使用[约翰·霍普斯金大学](https://www.jhu.edu/)的聚集新冠肺炎数据集的索引。事实证明，我们有几种不同的方式来消费这个数据集。这为我们探索从不同类型的后端(包括平面文件、数据库和 API)获取数据的几种模式提供了机会。作为开发人员，我们通常不会选择从哪里获取数据。应用程序带有预先存在的数据源和定义的接口。我们甚至可能需要与多个数据源进行交互，以构建应用程序所需的搜索类型。

顺便说一下，Algolia 继续为非营利的新冠肺炎相关应用程序提供[免费专业计划](https://www.algolia.com/blog/algolia/supporting-our-communities-during-this-time-of-need/)，如果你正在寻找建立(或已经建立)类似的东西。

## [](#getting-started)入门

在 Algolia，我们将数据摄取称为[三步索引流程](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/):

1.  从我们的数据源获取数据
2.  转换提取的数据
3.  将我们的记录发送到索引

在这篇博客的其余部分，我们将通过一些策略从 JHU 提供的每种可用数据格式中获取数据。接下来，你需要在现有的 Algolia 账户下创建一个索引，或者[注册一个免费账户](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel&utm_id=covid-geosearch)并安装一个应用程序。

你可以在新冠肺炎地理搜索演示报告的[目录下找到我们下面讨论的所有摄取脚本。随意克隆它并测试完成的演示。您需要从`example.env`创建一个`.env`文件，并在运行脚本之前添加您的 Algolia 应用程序信息和 API 密钥。](https://github.com/chuckmeyer/covid-geosearch/tree/development/scripts)

## [](#fetching-data-from-flat-files)从平面文件中提取数据

约翰·霍普斯金方便地将他们的全球和美国数据发布到至少每天更新的 GitHub 回购协议上。数据本身以 CSV 格式存储。这些文件包含地图前端所需的地理编码坐标。

例如，这是我的家乡俄亥俄州的记录:

```
`39049,Franklin,Ohio,US,2021-08-31 04:21:46,39.96995815,-83.01115755,139178,1531,,,"Franklin, Ohio, US",10569.763874248532,1.0863785943180675`

```

数据科学家发现在存储库中使用平面文件存储数据很有吸引力，因为它强调版本控制和跟踪随时间的变化。如果有更好的信息可用，他们可以更新历史记录，同时还可以捕捉到记录已经更改的事实。高阶数据库可能很难跟踪这种变化。我们将编写一个快速脚本，将原始 CSV 数据直接接收到我们的索引中。我们可以使用 Python [pandas](https://pandas.pydata.org/) 库来做到这一点。

```
#!python3
import pandas

DATA_FILE = '../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/08-30-2021.csv'

def main():
  df = pandas.read_csv(DATA_FILE)
  print(df)

if __name__ == "__main__":
    main()

```

他们用日期戳命名每个文件，这意味着我们最终需要一种方法来找到包含最新数据的文件名。目前，我们只是把摄入时间锁定在某一天。尽管文件本身有很多数据。我们想把它精简到我们搜索需要的信息。由于我们不能在摄取步骤中这样做，我们将把它保存到[转换步骤](#transforming-our-data)中，然后继续下一个数据源。

## [](#fetching-data-from-a-database)从数据库中提取数据

MongoDB 的开发人员已经将 JHU 数据集导入到一个托管的 [MongoDB 文档存储库](https://www.mongodb.com/developer/article/johns-hopkins-university-covid-19-data-atlas/)。从数据库而不是原始文件消费有几个优点。首先，他们负责从 Github repo 接收 CSV 文件，并确保每小时更新一次数据。住在文档存储的下游意味着我们不必做整理平面文件和查找最新数据的工作。此外，MongoDB 为我们提供了一个受管理的端点和一个已知的接口，减少了开发人员的摩擦，因为我们不再需要处理平面文件的定制数据结构。

我们可以使用 [pymongo](https://pypi.org/project/pymongo/) 直接连接到他们的文档存储，并使用它的元数据来检索最新的记录。

```
#!python3
from pymongo import MongoClient

MDB_URL = "mongodb+srv://readonly:readonly@covid-19.hip2i.mongodb.net/covid19"

def main():
  client = MongoClient(MDB_URL)
  db = client.get_database("covid19")
  stats = db.get_collection("global_and_us")
  metadata = db.get_collection("metadata")

  # Get the last date loaded:
  meta = metadata.find_one()
  last_date = meta["last_date"]

  results = stats.find(
    {
      "date":last_date,
    }
    print(results)

if __name__ == "__main__":
    main()

```

构建搜索索引的一个关键部分是将数据精简到我们搜索数据所需的数量。因为我们使用数据库，所以不需要像平面文件那样接收整个文档。让我们将查询范围缩小到位置信息、地理编码坐标和确诊病例数。我们还可以跳过任何缺少位置数据的记录:

```
  results = stats.find(
    {
      "date":last_date,
      "loc":{"$exists": True, "$ne": [] }
    }, {
      "combined_name": 1, 
      "county": 1,
      "state": 1,
      "country": 1,
      "confirmed": 1,
      "loc": 1
    }

```

当我们以后转换数据记录时，这将使我们的工作更容易。

## [](#fetching-data-from-rest-apis)从 REST APIs 获取数据

MongoDB 还提供了一个 RESTful API[来检索相同的数据，而不必连接到文档存储。API 的优势在于只需要一个安全的 HTTP 连接就可以使用。挑战来自于复杂的数据请求，这些请求需要对 API 设计有深入的了解，我们将在下面看到。](https://www.mongodb.com/developer/article/johns-hopkins-university-covid-19-rest-api/)

首先，让我们构建我们的 REST 请求:

```
#!python3
import json
import requests

METADATA_URL = 'https://webhooks.mongodb-stitch.com/api/client/v2.0/app/covid-19-qppza/service/REST-API/incoming_webhook/metadata'
REST_URL = 'https://webhooks.mongodb-stitch.com/api/client/v2.0/app/covid-19-qppza/service/REST-API/incoming_webhook/global_and_us'

def main():
  meta = requests.get(METADATA_URL)
  last_date = meta.json()['last_date']
  query = {
    'min_date': last_date,
    'max_date': last_date,
    'hide_fields': '_id, fips, country_code, country_iso2, country_iso3, population, deaths, confirmed_daily, deaths_daily, recovered, recovered_daily'
  }
  response = requests.get(REST_URL, params=query)
  print(response.json())

if __name__ == "__main__":
    main()

```

同样，我们使用元数据来确保我们只获取最新的数据。注意，REST API 公开了一个`hide_fields`参数，我们可以用它作为负过滤器来获取我们需要的字段。这并不像列出我们想要的字段那样简单，但是这样设计 API 可能是有原因的。对于 API，我们必须使用给定的接口。幸运的是，正如我们将在下一节看到的，这是开发人员一直在思考的问题。

## [](#fetching-data-from-graphql-apis)从 GraphQL APIs 获取数据

我们探索的最后一个接口是经过认证的 GraphQL API。GraphQL 提供了与 REST API 相同的 HTTPS 可访问接口，但是增加了一个[规范化查询语言](https://graphql.org/)，这样就不需要对单个 API 非常熟悉。我们再次使用元数据来查找最新数据的日期，并只检索我们需要的字段。但是，因为这个 API 是经过身份验证的，所以我们必须首先访问一个身份验证端点并获取一个短期访问令牌:

```
#!python3
import json
import requests

GRAPHQL_AUTH = "https://realm.mongodb.com/api/client/v2.0/app/covid-19-qppza/auth/providers/anon-user/login"
GRAPHQL_URL  = "https://realm.mongodb.com/api/client/v2.0/app/covid-19-qppza/graphql"

def main():
  response = requests.get(GRAPHQL_AUTH)
  access_token =  response.json()['access_token']

  headers = {}
  headers["Accept"] = "application/json"
  headers["Content-Type"] = "application/json"
  headers["Authorization"] = "Bearer {}".format(access_token)

  metadata = requests.post(GRAPHQL_URL, headers=headers, json={'query': 'query { metadatum{ last_date }}'})
  if metadata.status_code != 200:
    raise Exception(f"Query failed to run with a {response.status_code}.")
  last_date = metadata.json()['data']['metadatum']['last_date']

  query = '''query {
    global_and_us(query: { date: "''' + last_date + '''" }, limit:5000)
    { confirmed county state country combined_name loc { type coordinates }}
}'''

  response = requests.post(GRAPHQL_URL, headers=headers, json={'query': query})
  if response.status_code != 200:
    raise Exception(f"Query failed to run with a {response.status_code}.")
  print(response.json())

if __name__ == "__main__":
    main()

```

这个界面并没有给桌面带来多少新的东西。与 REST API 相比，它有一点是标准化的。当我们跨多个不同的后端进行查询，或者针对移动或低带宽用例对数据进行精简切片时，GraphQL 确实大放异彩。在搜索之后引入额外的数据可能很有趣，但是我们可以用其他开销更少的接口来实现。

## [](#choosing-our-data-source)选择我们的数据源

这些摄取方法各有利弊。直接使用平面文件可以让我们最接近数据，并包含一些下游数据源中没有的全球市政数据。然而，使用这些文件意味着需要额外的代码来判断哪个文件有最新的数据。我们还需要做我们自己的数据标准化。

通过 MongoDB 使用文件提供了一个简单的接口，并建立在平面文件的托管数据接收层之上，但是会牺牲一些全局数据的保真度。对于这个用例，REST 和 GraphQL APIs 并没有提供比我们从消费上游资源中得到的更多的东西。所有这三个(MongoDB 和两个 API 方法)都提供了查询接口来过滤记录，并只检索索引所需的属性。

当我们考虑随着时间的推移保持数据同步时，我们可能会发现 MongoDB 接口很有吸引力，因为我们可以使用元数据来确保我们总是获取最新的案例计数。尽管我喜欢通过消费原始 CSV 文件获得的额外数据，但我们必须建立一种方法来监控回购，以确保我们总是获得最新的数据。

## [](#transforming-our-data)转换我们的数据

选择数据源后，我们的下一个任务是[为我们的索引构建数据](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/#simplifying-your-records)。Algolia 索引必须是 JSON 格式，每个记录有一个唯一的`objectID`。将这些 ID 映射到数据源中的现有 ID 是有意义的。Algolia API 可以[为我们](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/in-depth/what-is-in-a-record/#unique-record-identifier)创建这些 id，但是我们不建议这样做。除了这些要求之外，这里的目标是将我们的记录精简到最简单的形式，仍然在性能搜索和有用的结果之间提供正确的平衡。

我们正在创建新冠肺炎数据的地理表示，因此最起码，我们需要地理编码坐标和我们要显示的每条记录的确诊病例数。为了使数据更容易使用，我们应该包括位置信息，如显示在地图上的县或省，以及州和国家，以提高搜索的相关性。

下面是我们用于 CSV 平面文件摄取的转换代码，同样使用了`pandas`:

```
  covid_records = []
  for index, row in df.iterrows():
    # Skip locations w/o coordinates
    if pandas.isna(row['Lat']):
      print('Skipping {}: No geocode'.format(row['Combined_Key']))
    else:
      covid_record = {}
      covid_geocode = {}
      print(row['Combined_Key'])
      covid_record['`objectId`'] = row['Combined_Key']
      # Let's not use the combined key for US counties, instead let's use county and state  
      if pandas.isna(row['Admin2']):
        covid_record['location'] = row['Combined_Key']
      else:
        covid_record['location'] = row['Admin2'] + ', ' + row['Province_State']
      covid_record['country'] = row['Country_Region']
      covid_record['confirmed_cases'] = int(row['Confirmed'])
      covid_geocode['lat'] = row['Lat']
      covid_geocode['lng'] = row['Long_']
      covid_record['_geoloc'] = covid_geocode
      covid_records.append(covid_record)

```

摄取代码在所有摄取模式中保持相当一致。这是有意义的，因为它们都继承了作为根源文件的 CSV 文件的形状。下面是我们的 MongoDB 源代码的相同代码:

```
  covid_records = []
  for row in response.json():
    # Unassigned and Unknown records are alread scrubbed in this DB
    # Skip 'US' and 'Canada' since they have incomplete data
    # and locations w/o coordinates
    if row['combined_name'] != 'US' and row['combined_name'] != 'Canada' and 'loc' in row:
      covid_record = {}
      covid_geocode = {}
      print(row['combined_name'])
      covid_record['`objectId`'] = row['combined_name']
      # Let's not use the combined key for US counties, instead let's use county and state  
      if 'county' in row:
        covid_record['location'] = row['county'] + ', ' + row['state']
      else:
        covid_record['location'] = row['combined_name']
      covid_record['country'] = row['country']
      covid_record['confirmed_cases'] = row['confirmed']
      covid_geocode['lat'] = row['loc']['coordinates'][1]
      covid_geocode['lng'] = row['loc']['coordinates'][0]
      covid_record['_geoloc'] = covid_geocode
      covid_records.append(covid_record)
    else:
      print('Skipping {}: No geocode'.format(row['combined_name']))

```

在这两种情况下，我们都只剩下一个带有搜索和发现所需属性的规范化 Python 字典。

## [](#building-our-index)建筑节令

最后一步是将我们的规范化数据发送到 Algolia 搜索 API。此时，代码是使用 [Python SDK](https://www.algolia.com/doc/guides/getting-started/quick-start/tutorials/quick-start-with-the-api-client/python/?client=python) 的样板文件:

```
  client = SearchClient.create(os.getenv('APP_ID'), os.getenv('API_KEY'))
  index = client.init_index(os.getenv('ALGOLIA_INDEX_NAME'))
  index.clear_objects()
  index.save_objects(covid_records)

```

这段代码清除所有现有记录，并使用最新数据重建索引。通过使用 Python 客户端与 Algolia API 进行交互，我们不必担心记录批处理和 API 重试之类的事情，从而减少了这里所需的代码量。它还[确保零停机时间](https://support.algolia.com/hc/en-us/articles/4406981911697-What-is-the-standard-SLA-)。

大多数索引默认值对于这个用例来说都很好。我想做的唯一改变是将我们的可搜索属性限制为`country`和`location`，并根据案例数量对结果进行降序排列。高病例数倾向于映射到大型人口中心，这使得我们缩小地图时数据更有用。

以下是我的索引配置片段:

```
{
...
  "searchableAttributes": ["unordered(country)", "unordered(location)"],
  "ranking": ["typo", "geo", "words", "filters", "proximity", "attribute", "exact", "custom"],
  "customRanking": ["desc(confirmed_cases)"],
...
}

```

## [](#next-steps)下一步步骤

在最初的数据接收之后，我们还需要决定如何随着底层数据的变化来更新我们的索引。根据我们启动更新和修改索引的方式，我们可以遵循许多模式。

### [](#rebuilding-our-index)重建我们的索引

因为我们的新冠肺炎数据集有固定数量的记录，所以每隔几个小时或一天几次清除和重建索引并不是特别繁重。我们可以使用预定的[无服务器函数](https://www.serverless.com/examples/aws-python-scheduled-cron)或在构建过程中嵌入我们的脚本，以固定的时间间隔来完成这项工作。如果我们想要更接近实时更新，我们可以在新冠肺炎回购协议上设置一个 GitHub 动作，每当我们合并一个新的 PR 时触发我们的索引重建。

### [](#atomic-rebuilds-for-production-data)生产数据的原子重建

对于较大的生产索引，我们希望避免索引重建时的停机时间或更新失败的可能性。出于这个原因，我们将使用[替换所有对象](https://www.algolia.com/doc/api-reference/api-methods/replace-all-objects/)方法来执行原子重新索引。这种方法不是就地重建索引，而是创建一个新的索引，并且只在保存所有记录后进行切换，从而减少了停机时间，并确保操作实际上是原子性的。

### [](#incremental-updates)增量更新

有时，对单个记录甚至单个数据字段执行[增量更新](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/incremental-updates/)比重建整个索引更有意义。例如，如果我们在一段时间内跟踪确诊病例，我们将需要考虑历史数据的偶尔更新。没有必要为一个小的数据变化重建整个索引。使用 webhook 订阅或通过直接监控文件来监控更改更有意义。然后，随着数据的变化，我们可以批量增加或更新记录。在这种模式中，惟一的`objectId`对于在更新期间引用现有记录变得至关重要。

### [](#building-a-front-end)建筑前端

现在我们有了索引，我们可以专注于构建我们的前端。Algolia React InstantSearch 库中的[地理搜索小部件](https://www.algolia.com/doc/api-reference/widgets/geo-search/react/)让我们可以使用谷歌地图相当轻松地构建一个前端。您可以先睹为快回购中的[，但是我们将把它留到以后再实现。](https://github.com/chuckmeyer/covid-geosearch)

在我们的开源代码交换平台上查看相关解决方案。

[![](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/?page=1&refinementList%5Bproduct_features%5D%5B0%5D=Geo%20Search)