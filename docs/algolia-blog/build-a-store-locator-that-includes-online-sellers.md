# 建立一个包含在线卖家的商店定位器

> 原文：<https://www.algolia.com/blog/engineering/build-a-store-locator-that-includes-online-sellers/>

假设您的任务是构建一个应用程序，帮助消费者找到提供特定服务的机构。其中一些机构是当地的实体店面，另一些则是服务于同一地区的纯在线机构。

这个问题最初是由 Alejo Arias 在 Algolia [论坛](https://discourse.algolia.com/t/aroundlatlng-or-faceted/13655)上提出的:

> 我的索引中有一系列提供者:
> 
> *   一些是国家供应商(他们有一个`national: true`属性)
> *   有些是州范围内的提供商(他们有一个`state`属性和他们服务的州的列表，例如`[“NY”, “FL”]`)
> *   有些是本地的(它们的`_geoloc`属性中有特定的 lat/lng)
> 
> **我希望在我的搜索结果中包含与我的用户位置相近的本地提供商相匹配的任何内容，同时还提供(相同的结果)州和国家提供商**。
> 
> 添加一个`aroundLatLng`滤镜会自动移除其他结果，无论我尝试什么样的面或滤镜。
> 
> 我怎样才能实现这一点？
> 基本上我想有这样的东西:`aroundLatLng: x,y OR state: NY OR national: true`

那么，如何将实体商店的地理搜索结果与基于布尔或文本的查询结果结合起来呢？如何构建一个界面来统一显示它们呢？

## [](#geographic-data-and-algolia-search)地理数据和 Algolia 搜索

正如 Alejo 提到的，您可以通过在记录上添加一个特殊的`_geoloc`属性来使用 Algolia 进行地理搜索。您可以将一组或多组纬度/经度元组放入该属性，以指示链接到记录的地理位置。

然后，您使用 Algolia 客户端库来查询这些地理编码记录——过滤固定点周围的半径(`aroundLatLong`)或形状内的区域(`insideBoundingBox`或`insidePolygon`)。文档详细介绍了这些方法之间的区别。你也可以通读[这些文章](https://www.algolia.com/blog/engineering/building-a-store-locator-in-react-using-algolia-mapbox-and-twilio-part-1/)带你构建一个纯地理的商店定位器。

但是，您不能从同一个查询中提取地理和非地理结果。如果您正在搜索邻近性，缺少`_geoloc`属性的记录将不会显示在结果集中。

那么，当并非所有记录都有地理坐标时，如何执行这种搜索呢？

## [](#a-single-index-solution)单指标解

你可以通过地理搜索来做所有的事情。通过将`_geoloc`数据添加到州和国家记录中，你可以使用地理查询来搜索一切。例如，将全州范围的机构放置在每个州中心的坐标上。这是我添加到论坛帖子中的最初解决方案，但是这个解决方案有几个问题:

1.  Alejo 特别提到一些提供商跨越多个州
2.  将提供商放在州的中心会给居住在州边界附近的消费者带来不准确的结果
3.  国家供应商需要每个州的记录

## [](#a-multi-index-solution)多指标解决方案

或者，您可以构建一个多索引解决方案，其中一个索引用于包含地理数据的实体店面，另一个索引用于州和国家提供商。然后，您可以单独搜索这两个数据源，并混合结果集。这种方法要求每次搜索两次 Algolia 查询，但它将允许我们保证从两种类型的提供商那里得到结果。

## [](#preparing-your-indices)准备您的指数

首先，你需要一个机构数据集。您可以使用一些资源从头构建一个。你可以从匿名地址数据开始，这个报告包含了美国大约 3000 个地址。然后，通过[小脚本](https://github.com/chuckmeyer/agency-finder/tree/main/scripts)运行这些地址，添加虚构的代理名称，并随机将一些代理标记为“首选”。

```
def transform_records(addresses):
  address_records = []
  for address in addresses:
    record = {}
    record_geocode = {}
    # One in ten chance agency is preferred 
    record['preferred'] = 10 == random.randint(1,10)

    record['objectID'] = random_name.generate_name().title()
    if record['preferred']:
      record['name'] = f"{record['objectID']} Agency (Preferred)"
    else:
      record['name'] = f"{record['objectID']} Agency"
    record['address'] = address.get('address1')
    record['city'] = address.get('city')
    record['state'] = address.get('state')
    record['zip'] = address.get('postalCode')
    record_geocode['lat'] = address['coordinates']['lat']
    record_geocode['lng'] = address['coordinates']['lng']
    record['_geoloc'] = record_geocode
    address_records.append(record)
  return address_records

```

您可以使用另一个脚本为第二个索引生成州级和州级机构。两个数据集都驻留在[这个 repo](https://github.com/chuckmeyer/agency-finder/tree/main/data) 中。您可以在现有的 Algolia 帐户下从这些数据集创建指数，或者[注册一个免费帐户](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel&utm_id=agency-finder)并建立一个新的`agency_finder`应用程序。

## [](#building-the-front-end)建筑前端

既然您已经填充了索引，那么是时候构建前端了。Algolia 在 InstantSearch 库中的`geoSearch`组件包括一个助手组件，用于初始化 Google Maps API，渲染地图，并将该地图绑定到 Algolia 索引中的地理位置查询。这是我之前用来构建一个[新冠肺炎案例可视化器](https://www.algolia.com/blog/engineering/building-a-covid-19-geosearch-index-using-csv-files-mongodb-or-graphql/)的同一个组件。但是，对于这个项目，您希望用户输入地址，并使用 Google Places API 为他们解析地理位置信息。事实证明，在 InstantSearch 中使用开箱即用的组件具有挑战性，因此您将从头开始构建自己的界面。

这篇[博客文章](https://www.tracylum.com/blog/2017-05-20-autocomplete-an-address-with-a-react-form/)为我们提供了一个在 React 中构建地址自动完成表单的可靠模型。您将使用它作为您的 [`AgencyFinderForm`](https://github.com/chuckmeyer/agency-finder/tree/main/src/AgencyFinderForm.js) 组件的基础，以呈现地址自动完成输入字段以及只读字段来显示结果地址。纬度/经度存储在 state 中，但没有显示在表单上

您可以通过使用 React 组件周围的 Google [包装器](https://cloud.google.com/blog/products/maps-platform/loading-google-maps-platform-javascript-modern-web-applications)来初始化`google`对象并添加位置 API，从而使博客中的代码现代化。

```
   renderForm = (status) => {
    switch (status) {
      case Status.SUCCESS:
        return ;
      default:
        return <h3>{status} ...</h3>;
      };
  }

  render() {
    return (
      <div>
        <h1>Find an Agency</h1>
        <p className='instructions'>🔍 Search for your address to find the closest agencies.</p>
        <div className='left-panel'>
          <Wrapper apiKey={process.env.REACT_APP_GOOGLE_API_KEY} render={this.renderForm} libraries={["places"]} />
        </div>
        <div className='right-panel'>
          <AgencyFinderResults hits={this.state.results} />
        </div>
      </div>
    )
  }
}

```

接下来，向基本表单添加一个`clear`按钮。

```
  handleClear() {
    this.setState(this.initialState);
    var input = document.getElementById('autocomplete');
    input.value = '';
    google.maps.event.removeListener(this.autocompleteListener);
    this.initAutocomplete();
  }

```

最后，您将使用以下代码从 Places API 清理处理`address_components`:

```
  handlePlaceSelect() {
    const addressObject = this.autocomplete.getPlace();
    const address = addressObject.address_components.reduce((seed, { short_name, types }) => {
      types.forEach(t => {
        seed[t] = short_name;
      });
      return seed;
    }, {});
    [this setState](this.setState)({
      streetAddress: `${address.street_number} ${address.route}`,
      city: address.locality ? address.locality : address.sublocality_level_1,
      state: address.administrative_area_level_1,
      zipCode: address.postal_code,
      geoCode: addressObject.geometry.location.lat() + ', ' + addressObject.geometry.location.lng(),
    });
  }

```

## [](#querying-for-results)查询结果

当用户选择了一个位置，并且组件状态中存储了纬度、经度和地址信息之后，就可以查询索引了。您使用来自 [Javascript API 客户端](https://www.algolia.com/doc/api-reference/api-methods/multiple-queries/?client=javascript)的`multipleQueries`方法将两个查询批处理在一起并组合结果。这仍然算作两次查询，但它减少了到 API 的往返次数。

```
handleSubmit(event) {
    const queries = [{
      indexName: statesIndex,
      query: this.state.state,
      params: {
        hitsPerPage: 10
      }
    }, {
      indexName: geoIndex,
      query: '',
      params: {
        aroundLatLng: this.state.geoCode,
        facetFilters: [ this.state.preferred ? 'preferred:true' : '' ],
        hitsPerPage: 10,
      }
    }];

    this.searchClient.multipleQueries(queries).then(({ results }) => {
      let allHits = [];
      results.map((result) => {
        return allHits.push(...result.hits);
      });
      this.props.handleCallback(allHits);
    });
  }

```

首先，初始化两个查询。请注意`multipleQueries`方法如何允许我们混合地理和基于字符串的查询，甚至为您的“首选”机构添加一个
可选的`facetFilter`。然后将查询数组传递给客户机。响应包括来自每个
查询的单个结果，但是您可以将来自两个结果集的`hits`分解到一个数组中，并将它们传递给`AgencyFinderResults`组件。

## [](#putting-it-all-together)把所有的东西放在一起

现在，您有了一个可靠的概念验证 React 组件，可以将地理和非地理结果分层到一个结果集中。此时，您可以通过添加 Google 地图来显示地理结果，从而改进示例。您还可以转回到单个索引，使用`multipleQueries`功能用不同的参数多次查询同一个索引。

完整的例子可以在这个 [Github repo](https://github.com/chuckmeyer/agency-finder) 中找到，或者你可以尝试一个[现场演示](https://agency-finder.vercel.app/)。