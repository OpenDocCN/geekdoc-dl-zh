# 特色聚焦:Algolia 搜索的可选过滤器

> 原文：<https://www.algolia.com/blog/engineering/feature-spotlight-optional-filters/>

上周在回答 Algolia 论坛的帖子时，我对 Algolia 搜索的可选过滤器[进行了深入研究。与过滤器和方面过滤器类似，可选过滤器在查询时应用，允许您使用各种上下文信息来改善结果。与其他过滤器不同，可选过滤器不会从结果集中删除记录。相反，它们允许您说，“如果记录匹配这个过滤器，*然后*将它们在排名中上移或下移”，而不改变返回的记录数。](https://www.algolia.com/doc/guides/managing-results/rules/merchandising-and-promoting/in-depth/optional-filters/)

一些有趣的使用案例:

*   将提及特定用户的帖子移动到结果集的顶部
*   客户当地商店提供的升级产品
*   对缺货的产品进行降级
*   基于外部推荐引擎将特定记录固定在列表的顶部

您可以使用过滤器和方面过滤器来减少查询时结果集中的记录数量，然后使用可选的过滤器来处理剩余记录的排名。您甚至可以应用[过滤评分](https://www.algolia.com/doc/guides/managing-results/refine-results/filtering/in-depth/filter-scoring/)来进一步控制记录的顺序。

下面是一个将`product_type`和`price_range`的方面过滤器应用于来自 Shopify 商店的索引的例子。代码选择两个记录的`objectID`进行提升，然后将它们作为可选过滤器注入查询。如果这些产品中的任何一个是与查询中的其他条件相匹配的结果集的一部分，那么这些产品将被推到排名的顶部。代码使用过滤评分来确保`featuredProduct`总是出现在`alternateProduct`的上方。

```
import algoliasearch from 'algoliasearch';

const featuredProduct = '41469303161004';
const alternateProduct = '41469346644140';

const client = algoliasearch('H2M6B61JEG', 'b1bdfc3258823bb4468815a664dce649');

// Standard replica
const index = client.initIndex('shopify_algolia_products_price_asc_standard');

// with params
index.search(query, {
    facetFilters: [[
        "product_type:HardGood",
        "price_range:75:100"
    ]],
    optionalFilters: [[
        `objectID:${featuredProduct}<score=500>`,
        `objectID:${alternateProduct}<score=200>`
    ]],
    hitsPerPage: 50,
}).then(({ hits }) => {
    console.log(hits.map(item => `- ${item.title} | ${item.product_type} | ${item.objectID} | ${item.price_range}`).join('\n'));
});

```

注意，这段代码使用了按价格排序的 Shopify 索引的标准副本。可选的过滤器不能很好地与[虚拟副本](https://www.algolia.com/doc/guides/managing-results/refine-results/sorting/in-depth/replicas/)索引一起使用，因为两者都在查询时重新排列记录，导致不可预知的结果。如果您计划使用可选的过滤器，您应该使用在索引时应用排序的标准副本。

您还可以使用否定的可选过滤器来降低记录的排名。例如，如果您想要将当前用户撰写的帖子在排名中推低，但不完全删除它们:

```
index.search('', {
    filters: `date_timestamp > ${Math.floor(d.setDate(d.getDate() - 7) / 1000)}`,
    optionalFilters: [
      `author:-${user.name}`
    ],
    hitsPerPage: 50,
}).then(({ hits }) => {
    console.log(hits};
});

```

可选的过滤器是一个强大的工具，可以添加到您的排名工具带上，但是请记住，任何查询时间的计算都会影响搜索性能。例如，您不应该对可能返回超过 100，000 个结果的搜索使用过滤评分。尽可能将排名标准移至索引配置。仅当您需要在查询时使用更实时的上下文进一步优化结果时，才使用可选的过滤器。

如果你在你的搜索界面中找到了一个很好的(或不太好的)可选过滤器用例，请告诉我。