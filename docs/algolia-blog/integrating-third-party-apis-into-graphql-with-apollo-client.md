# 使用 Apollo 客户端将第三方 API 集成到 graph QL

> 原文：<https://www.algolia.com/blog/engineering/integrating-third-party-apis-into-graphql-with-apollo-client/>

要将快速且功能丰富的第三方 API 添加到任何代码库中，开发人员需要在一个旨在包含这些自给自足的 API 的生态系统中工作。

开发人员还需要依靠像带有 Apollo Client 和 React 的 GraphQL 这样的技术来帮助他们管理各种功能和数据源。

在大多数情况下，GraphQL 是天赐之物。作为一个灵活的数据层，它有助于集中和标准化后端和前端之间的数据交换。

然而，要求所有数据交换都流经 GraphQL 有时会阻碍外部 API 发挥其作为关键功能组件的全部潜力。在我们的例子中，我们希望集成一个基于云的搜索 API，它的性能远远优于 GraphQL。

我们在本文中解决的问题是，你如何使用一个独立的外部 API，与 GraphQL — 并行，也就是说，不通过 GraphQL。

### [](#why-we-decided-to-bypass-graphql-with-apollo-client-and-how-we-achieved-interoperability)为什么我们决定用 Apollo 客户端绕过 GraphQL，以及我们如何实现互操作性

我们将本文分为两部分:

*   解释为什么我们决定绕过 GraphQL 来允许我们的前端直接调用外部 API 的云服务
*   我们用来构建外部 API 和 GraphQL 与 Apollo 客户端之间的互操作性的代码

请随意跳到代码。但是背景故事可能会帮助那些面临同样问题的人:*如何将 GraphQL 与提供自己数据集的自给自足的 API 结合起来？*

## [](#why-graphql-why-an-external-search-api)为什么是 GraphQL？为什么是外部搜索 API？

说我们的在线服务是数据密集型的是一种保守的说法。Openbase 帮助开发者找到完全符合他们需求的开源库和包。他们可以使用强大的指标和用户评论来搜索和比较开源包。这些信息有很多来源，包括 Github 和我们自己的数据库。

还有其他的担忧:

*   每周下载量
*   维护
*   捆绑大小
*   分门别类

### [](#using-graphql-with-apollo-client)使用 GraphQL 与 Apollo 客户端

[Openbase](https://openbase.com) 是一个 GraphQL 商店。我们用`[@apollo/client](https://openbase.com/js/@apollo/client)`访问数据，用 [`React`](https://openbase.com/js/react) 进行渲染。我们的许多 React 组件都是使用 [`@apollo/client#readFragment`](https://www.apollographql.com/docs/react/caching/cache-interaction/#readfragment) 从我们的 GraphQL api 引用数据构建的。这使得我们的组件与`@apollo/client`和 GraphQL 紧密耦合，与其他数据源的可重用性更低。

示例:

`PackageTile.tsx`的代码:

```
import { useApolloClient } from '@apollo/client'
import React, { FC } from 'react'

// Generated using GraphQL Code Generator
import { AlgoliaProductFragment, AlgoliaProductFragmentDoc } from '@openbase/graphql'

interface PackageTileProps {
	packageId: string
}

const PackageTile: FC<PackageTileProps> = ({ packageId }) => {
	const apolloClient = useApolloClient()

	const packageFragment = apolloClient.readFragment<AlgoliaProductFragment>({
		fragment: AlgoliaProductFragmentDoc,
		id: `Package:${packageId}`
	})

	return (
		<div>
			<div>{packageFragment.name}</div>
			<div>{packageFragment.description}</div>
			<div>{packageFragment.bundleSize.gzipSizeInBytes}</div>
		</div>
	)
}

```

### [](#building-search-with-a-third-party-cloud-based-search-api)使用第三方、基于云的搜索 API 构建搜索

有了这些数据，搜索对我们的业务是必不可少的。我们的开发者用户从搜索栏开始。但是他们做的不止这些——他们分类、过滤、浏览多个结果和分类页面，所有这些都来自外部搜索 API。最重要的是，我们的产品提供评级和推荐，并允许用户个性化他们最喜欢的语言的结果。这些额外的功能大部分直接来自外部搜索 API 的结果，结合我们自己的数据源，可以通过 GraphQL 访问。

## [](#the-two-options-for-combining-graphql-and-an-external-search-api)这两个选项用于组合 GraphQL 和一个外部搜索 API

有两种*互斥的*方法可以将外部 API 集成到 GraphQL 前端层:

1.使用前端的 API 作为`@apollo/client`
2 旁边的辅助数据源。使用我们的 GraphQL API 后面的后端 API，并维护一个前端数据源

### [](#option-1-the-reason-for-bypassing-graphql)选项 1——绕过 GraphQL 的原因

由 [Algolia](https://www.algolia.com/developers/) 提供的外部搜索 API**快速**，在用户输入时即时显示搜索结果。将 Algolia 放在我们的 GraphQL API 后面会成为搜索的瓶颈，大大降低搜索速度。

1.  Algolia 的 API 显示超过 300，000 个请求的平均搜索速度为**_ 32 毫秒**(在撰写本文时)
2.  我们的 GraphQL API 请求有时需要 100-200 毫秒

相比之下，这种延迟是站不住脚的。

为了获得额外的速度，这对我们的产品至关重要，我们真的需要直接调用 API。

### [](#option-2-the-reasons-to-put-the-external-api-behind-graphql)选项 2–将外部 API 置于 GraphQL 之后的原因

对于绕过 GraphQL，我们有两个顾虑:

*   **缓存。**构建在 GraphQL 缓存上的现有组件不知道通过我们的搜索 API 检索到的记录。
*   **无图式。搜索 API 的记录是无模式的。它的索引是平面的，包含搜索、显示和排序所需的最小属性集。虽然对于速度、相关性和可伸缩性来说是必要的，但是一个平面的、无模式的数据集不容易与我们的 GraphQL 模式中的对象类型相一致，而我们的组件是建立在该模式之上的。**

此外，将搜索功能放在 GraphQL 的后面，将搜索结果集成到前端组件中，正好可以脱离 GraphQL 的框架:前端只是向 GraphQL 发出请求，检索相同的对象类型，开发人员的体验保持一致。

### [](#the-decision)决定

因此，我们面临以下选择:要么选择更简单的解决方案(将 API 插入 GraphQL ),要么再努力一点，从 API 中获得我们期望的全部功能。

我们毫不犹豫地决定为用户提供即时搜索结果。

## [](#how-we-integrated-the-external-api-into-our-graphql-front-end-components)我们如何将外部 API 集成到我们的 GraphQL 前端组件中

Algolia 使用不属于 GraphQL 层的索引。从这一点开始，我们将引用 [`algolia`](https://www.algolia.com/doc/api-client/getting-started/what-is-the-api-client/javascript/?client=javascript) 。

我们的 GraphQL 客户端叫做`@apollo/client`。

基于以上考虑，我们引入了一个`algolia-to-@apollo/client`互操作层，这样我们就能够:

1.  直接在前端使用 API(以保持**快速**)
2.  使用我们现有的针对 GraphQL 构建的 react 组件来维护与我们应用程序的其余部分一致的开发人员和用户体验

在高层次上，互操作性是这样工作的:

1.  从`algolia-indices`查询记录
2.  使用自定义映射函数和从 [`@graphql-codegen/cli`](https://openbase.com/js/@graphql-codegen/cli) 生成的 GraphQL 片段将记录中的数据写入我们的`@apollo/client`缓存
3.  所有组件都使用 GraphQL 从`@apollo/client`的缓存中读取数据

下面是我们如何建立我们的指数来实现这一点(用`[typescript](https://openbase.com/js/typescript)`建立):

`algoliaToGraphQL.tsx`的代码:

```
import type { ObjectWithObjectID, SearchOptions, SearchResponse } from '@algolia/client-search'
import type { RequestOptions } from '@algolia/transporter'
import type { ApolloClient, DocumentNode } from '@apollo/client'
import type { SearchClient, SearchIndex } from 'algoliasearch'

// Narrowly type our indices for intellisense
export type AlgoliaTypedIndex<TRecord> = Omit<SearchIndex, 'search'> & {
	search: (query: string) => Readonly<Promise<SearchResponse<TRecord & ObjectWithObjectID>>>
}

// Define the schema for configuring the custom mappings
export interface AlgoliaToGraphQLFieldConfig<
	TIndex extends ObjectWithObjectID,
	TFragment extends Record<string, any>
> {
	__typename: string
	fragment: DocumentNode
	fragmentName?: string
	id?: (value: TIndex) => string
	write: (value: TIndex) => MaybePromise<Maybe<TFragment>>
}

// Define how the custom mapping will type-translate to narrowly-typed indices
export interface AlgoliaToGraphQLFields {
	[indexName: string]: AlgoliaToGraphQLFieldConfig<any, any>
}

export type AlgoliaToGraphQLResult<T extends AlgoliaToGraphQLFields> = {
	[P in keyof T]: T[P] extends AlgoliaToGraphQLFieldConfig<infer I, any>
		? AlgoliaTypedIndex<I>
		: never
}

// Create algolia-indices that will inject hits data to our `@apollo/client` cache
const createIndex = async <
	TIndex extends ObjectWithObjectID,
	TFragment extends Record<string, any>
>(
	algolia: SearchClient,
	apollo: ApolloClient<object>,
	name: string,
	config: AlgoliaToGraphQLFieldConfig<TIndex, TFragment>
): Promise<AlgoliaTypedIndex<TIndex>> => {
	const {
		__typename,
		fragment,
		fragmentName,
		id = (value) => `${__typename}:${value.objectID}`,
		write,
	} = config

	const writeFragment = async (value: TIndex): Promise<void> => {
		const fragmentData = await write(value)

		!!fragmentData && apollo.writeFragment<TFragment>({
			fragment,
			fragmentName,
			data: { __typename, ...fragmentData },
			id: id(value),
		})
	}

	const index = algolia.initIndex(name) as AlgoliaTypedIndex<TIndex>

	return {
		...index,
		// Override search to write everything into cache.
		async search(query, opts) {
			const result = await index.search(query, opts)

			await Promise.all(result.hits.map(async (hit) => writeFragment(hit)))

			return result
		},
	}
}

// Generate all of the new algolia indices from a config
export const algoliaToGraphQL = async <T extends AlgoliaToGraphQLFields>(
	algolia: SearchClient,
	apollo: ApolloClient<object>,
	config: T
): Promise<AlgoliaToGraphQLResult<T>> => {
	const indices = await Promise.all(
		Object.entries(config).map(async ([indexName, fieldConfig]) => {
			const index = await createIndex(algolia, apollo, indexName, fieldConfig)

			return [indexName, index] as readonly [string, AlgoliaTypedIndex<any>]
		})
	)

	return indices.reduce(
		(acc, [indexName, index]) => ({ ...acc, [indexName]: index }),
		{} as AlgoliaToGraphQLResult<T>
	)
}

```

一旦我们有办法将 [`algoliasearch`](https://www.algolia.com/doc/api-client/getting-started/instantiate-client-index/javascript/?client=javascript) 数据注入到我们的`@apollo/client`缓存中，我们只需键入 Algolia 记录的形状，定义要写入的 GraphQL 片段，将记录映射到片段，并构建新的索引。

`types.ts`的代码:

```
import type { ObjectWithObjectID } from '@algolia/client-search'

// Types for our algolia records (for demonstration only)

interface BaseAlgoliaTypes {
	AlgoliaProduct: {
		name: string // String!
		description?: Maybe<string> // String
		bundleSize?: Maybe<number> // Int
		starRating: number // Float!
	}
}

export type AlgoliaTypes = {
	[P in keyof BaseAlgoliaTypes]: BaseAlgoliaTypes[P] & ObjectWithObjectID
}

```

用于映射的 GraphQL 片段的代码:

```
fragment AlgoliaProduct on Package {
id
name
description
bundleSize {
gzipSizeInBytes
}
starRating
}

```

`indices.ts`的代码:

```
import type { ApolloClient } from '@apollo/client'
import type { SearchClient } from 'algoliasearch'
import { algoliaToGraphQL, AlgoliaToGraphQLFieldConfig } from './algoliaToGraphQL'
import type { AlgoliaTypes } from './types'

// Generated using GraphQL Code Generator
import { AlgoliaProductFragment, AlgoliaProductFragmentDoc } from '@openbase/graphql'

// Map algolia package records to our `AlgoliaProduct` fragment in GraphQL
const packages: AlgoliaToGraphQLFieldConfig<
	AlgoliaTypes['AlgoliaProduct'],
	AlgoliaProductFragment
> = {
	__typename: 'Package',
	fragment: AlgoliaProductFragmentDoc,
	write(value: AlgoliaTypes['AlgoliaProduct']): AlgoliaProductFragment {
		return {
			id: value.objectID,
			name: value.name,
			description: value.description,
			bundleSize: {
				gzipSizeInBytes: value.bundleSize,
			},
			starRating: value.starRating,
		}
	},
}

// Create all of our indices
export const getIndices = (algolia: SearchClient, apollo: ApolloClient<object>) => {
	return algoliaToGraphQL(algolia, apollo, {
		packages,
		// Different sortings using algolia virtual replicas
		packages_by_bundleSize_desc: packages,
		packages_by_starRating_desc: packages,
	})
}

```

`AlgoliaApolloProvider.tsx`的代码:

```
import { useApolloClient } from '@apollo/client'
import algolia from 'algoliasearch'
import React, {
	createContext,
	FC,
	ReactNode,
	useContext,
	useEffect,
	useState
} from 'react'
import { getIndices } from './indices'

export type AlgoliaApolloContextIndex = InferFromPromise<ReturnType<typeof getIndices>>

const AlgoliaApolloContext = createContext<Maybe<AlgoliaApolloContextIndex>>(null)

export interface AlgoliaApolloProviderProps {
	children?: ReactNode
}

// Wrap your application with this, to be able to use the interoperability anywhere
export const AlgoliaApolloProvider: FC<AlgoliaApolloProviderProps> = ({
	children,
}) => {
	const [index, setIndex] = useState<Maybe<AlgoliaApolloContextIndex>>(null)

	const apollo = useApolloClient()

	useEffect(() => {
		const newClient = algolia(process.env.ALGOLIA_APP_ID, process.env.ALGOLIA_API_KEY)

		getIndices(client, apollo).then((newIndices) => setIndex(newIndices))
	}, [apollo])

	return (
		<AlgoliaApolloContext.Provider value={index}>
			{children}
		</AlgoliaApolloContext.Provider>
	)
}

// Hook to use in any component that needs to search algolia
export const useAlgoliaIndex = () => useContext(AlgoliaApolloContext)

```

从这一点开始，每当搜索 Algolia 时，使用`@apollo/client`缓存的组件将开箱即用:

```
import { NextPage } from 'next'
import React from 'react'
import { useAlgoliaIndex } from './AlgoliaApolloProvider'

const Page: NextPage = () => {
	const index = useAlgoliaIndex()
	const [hits, setHits] = useState<any[]>([])

	useEffect(() => {
		index?.packages
			.search('react', { length: 20, offset: 0 })
			.then((results) => setHits(results.hits.slice()))
	}, [index])

	return (
		<div>
			{!!hits && hits.map((hit) => (
				<PackageTile key={hit.objectID} packageId={hit.objectID} />
			))}
		</div>
	)
}

export default Page

```

这样，我们实现了我们为快速搜索设定的目标，开发人员和用户体验与 Openbase 的其他部分保持一致。我们修改的索引的 API 与来自`algoliasearch`的未修改的索引相同，并且我们的组件可以保持不变 — 特定于我们的单个`@apollo/client`数据源。

想出一套高质量和相关的 Github repos 是一个无限的挑战。Algolia 的`algoliasearch`为 Openbase 消除了一个显著的复杂性，以构建一个允许用户找到满足其需求的最佳开源工具的体验。GraphQL 商店采用的集成 Algolia 或任何其他 API 的方法可能取决于您自己的要求和条件。