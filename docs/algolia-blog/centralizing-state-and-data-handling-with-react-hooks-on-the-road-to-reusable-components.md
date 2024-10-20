# 使用 React 挂钩集中状态和数据处理，创建可重用组件

> 原文：<https://www.algolia.com/blog/engineering/centralizing-state-and-data-handling-with-react-hooks-on-the-road-to-reusable-components/>

应用程序开发通常是被动的。我们看到了需求，我们会尽快提供解决方案。在这个快速的软件周期中，我们收集需求，并在需求出现时立即实现它们。我说的不是又快又脏。我指的是使用最好的 [RAD](https://en.wikipedia.org/wiki/Rapid_application_development) 实践——快速应用程序开发。

RAD 循环如下:你实现伟大的核心特性( [MVP 风格](https://en.wikipedia.org/wiki/Minimum_viable_product))，依靠多年的经验来创建可维护的代码。但是随着时间的推移，会发生一些事情:需求发生变化，编写了更多的代码，代码库开始违背您直觉上优秀但可能不完全健壮的架构。所以你开始重构。此外，您会发现技术在变化，提供了新的方法来使您的代码更简单、更清晰、更强大。

**进入游戏规则** [**反应钩子**](https://reactjs.org/docs/hooks-intro.html) 。而且，一个快速增长的业务需要你用大量的新特性重写你的应用程序。

*重写*——从头开始。生活提供了第二次机会。

## [](#how-react-hooks-saved-our-administration-application)React Hooks 如何保存我们的管理应用

应用程序开发也可以是主动(被动)的。我们的管理应用程序是数据密集型的。以前，许多独立的(和竞争的)组件独立地管理它们的数据——连接、格式化、显示、更新等等..

### [](#the-requirements-of-an-admin-application)一个 Admin 应用程序的要求

管理应用程序是集中数据处理的理想选择。管理员需要按原样查看数据，因此屏幕视图通常与底层数据的结构相匹配。因此，虽然我们面向客户端的仪表板为业务用户提供了功能视图，但管理员需要以一致和直观的方式查看用户或客户端订阅信息。

我们需要的是一个更具可扩展性的解决方案。由于我们从多个来源提取数据——所有这些都可以通过一个具有多个端点的 API 来访问——我们希望集中数据处理的常见方面。这不仅给我们带来了直接的好处(更好的测试、缓存、同步、标准输入)，还促进和简化了未来的数据集成。

### [](#a-customized-hook)定制挂钩

我们实现了一个名为`useData`的定制 React 钩子，它管理并因此集中了所有数据检索 API 调用、数据交换、类型检查、缓存和其他此类基于数据的功能。单单缓存就极大地提高了面向用户的速度。

同样重要的是，速度和集中化使我们的前端开发人员能够在界面的不同部分重用他们的组件和 UI 元素。这种可重用性创建了功能丰富、用户友好的 UI/UX，前端开发人员无需在每个组件中维护唯一的状态信息。最后，在幕后，数据可重用性使得驱动前端功能的模型一致。

我们将在以后的文章中讨论 React 钩子的前端好处；这篇文章是关于我们如何用一个可靠的和可伸缩的数据处理层来服务前端。

### [](#how-our-usedata-hook-centralized-the-process)我们的`useData`钩子如何集中这个过程

我们使用不同的数据源，有些比其他的更复杂，但是都遵循相同的 JsonAPI 规范。此外，他们都有相同的需求，即:

*   检索数据
*   反序列化并格式化它
*   验证其格式
*   执行错误处理(数据质量、网络)
*   与应用刷新和其他数据/工作流同步
*   缓存数据并保持其最新

说够了，下面是我们的`useData`钩子代码:

```
import { useCallback } from 'react';
import { useQuery, useQueryClient } from 'react-query';
import { ZodObject, infer as Infer } from 'zod';
import { useApi } from 'hooks';
import { metaBuilder, MetaInstance } from 'models';

interface Options {
  forceCallApi?: boolean;
  preventGetData?: boolean;
}

interface ApiData<T> {
  data?: T;
  meta?: MetaInstance;
}

export interface DataResult<Output> {
  data?: Output;
  meta: any;
  loading: boolean;
  errors: Error[];
  refresh: () => Promise<void>;
}

export const useData = <Model extends ZodObject<any>, ModelType = Infer<Model>, Output extends ModelType = ModelType>(
  builder: (data: ModelType) => Output,
  url: string,
  { forceCallApi = false, preventGetData = false }: Options = {}
): DataResult<Output> => {
  const queryClient = useQueryClient();

  const { getData } = useApi(url);

  const getDataFromApi = useCallback(async (): Promise<ApiData<Output>> => {
    // here we get the data (and meta) using getData, and handle errors and various states
    return { data: builder(apiData), meta: metaBuilder(apiMeta) }
  }, [getData, builder, queryClient, url, forceCallApi]);

  const { data: getDataResult, isLoading, error } = useQuery<ApiData<Output>, Error>(
    [url, forceCallApi],
    getDataFromApi,
    { enabled: !preventGetData, cacheTime: forceCallApi ? 0 : Infinity }
  );

  const refresh = useCallback(async () => {
    await queryClient.refetchQueries([url, forceCallApi], {
      exact: true,
    });
  }, [queryClient, url, forceCallApi]);

  return {
    data: getDataResult?.data,
    meta: getDataResult?.meta,
    loading: isLoading,
    errors: ([error]).filter((error) => error !== null) as Error[],
    refresh,
  };
};

```

正如您所看到的，这个钩子接受三个参数，当这些参数组合在一起时，我们可以获得以下所有功能:

*   一个“构建器”功能，用于转换和增强供我们的组件使用的数据
*   检索数据的 API 端点的 URL
*   可选参数。例如，在调用 API 之前忽略缓存或等待一些其他数据准备好

结果是我们的组件不再需要管理所有这些。我们已经抽象和封装了复杂性。

`useData`钩子返回一些我们可以在组件中使用的值:

*   一些状态:加载和错误(如果有)
*   数据(如果有)
*   元信息(如果有，例如分页信息)
*   刷新函数(通过再次调用 API 来刷新数据)

## [](#building-the-data)建筑数据

让我们更深入地看看这段代码做什么以及我们如何使用它。

### [](#schema-validation-with-zod)用 Zod 进行模式验证

得到数据是一回事。确保数据的结构或类型正确是另一个问题。复杂的数据类型需要像 [yup](https://github.com/jquense/yup) 或 [zod](https://github.com/colinhacks/zod) 这样的验证工具，它们强制执行有效和干净的方法，并提供工具和错误处理基于错误类型的运行时错误。我们的前端依赖于强类型数据集，因此验证阶段对我们来说至关重要。

我们用 [zod](https://github.com/colinhacks/zod) 。Zod 用于建立数据模型。例如，我们应用程序的模型可能是这样的:

```
import { object, string, number } from 'zod';

const Application = object({
  applicationId: string(),
  name: string(),
  ownerEmail: string(),
  planVersion: number(),
  planName: string(),
});

```

然后，为了构建我们的构建器函数，我们在 zod 模型上使用内部构建的通用助手。这个助手有两个参数:

*   我们的数据模型(上面例子中的应用)
*   用于丰富该模型的转换函数。

在我们的例子中，变压器看起来像这样:

```
import { infer as Infer } from 'zod';

const transformer = (application: Infer<typeof Application>) => ({
  ...application,
  get plan() {
    return `${application.planName} v${application.planVersion}`;
  },
});

```

丰富的另一个例子是如果一个模型有一个日期:我们通常希望它公开一个 javascript 日期而不是字符串日期。

我们有两个版本的助手函数(一个用于对象，一个用于数组)。下面是第一个:

```
import type { ZodType, TypeOf, infer as Infer } from 'zod';
import { SentryClient } from 'utils/sentry';

export const buildObjectModel = <
  Model extends ZodType<any>,
  ModelType = Infer<Model>,
  Output extends ModelType = ModelType
>(
  model: Model,
  transformer: (data: TypeOf<Model>) => Output
): ((data: ModelType) => Output) => {
  return (data: ModelType) => {
    const validation = model.safeParse(data);
    if (!validation.success) {
      SentryClient.sendError(validation.error, { extra: { data } });
      console.error('zod error:', validation.error, 'data object is:', data);
      return transformer(data);
    }
    return transformer(validation.data);
  };
};

```

zod 的类型化输出非常干净，看起来像我们自己编写的 typescript 类型，另外 zod 使用我们的模型解析 JSON。为了安全起见，我们使用 zod 的`safeParse`方法，该方法允许我们在解析步骤出错的情况下“按原样”发送回 JSON。我们的错误跟踪工具 [Sentry](https://sentry.io/welcome/) 也会收到一个错误。

在我们的例子中，我们的构建函数看起来像这样:

```
export const applicationBuilder = buildObjectModel(Application, transformer);
// and for the record, here is how to get the type output by this builder:
export type ApplicationModel = ReturnType<typeof applicationBuilder>;
// which looks like this in your code editor:
// type ApplicationModel = {
//   plan: string;
//   applicationId: string;
//   name: string;
//   ownerEmail: string;
//   planVersion: number;
//   planName: string;
// }

```

### [](#calling-the-api)调用 API

在内部，我们使用另一个定制钩子`useApi`(不到 200 行代码)来处理 GET/POST/PATCH/DELETE。在这个钩子中，我们使用 [axios](https://github.com/axios/axios) 调用后端 API 并执行所有典型的 [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) 功能。例如，在读取端，Axios 将我们收到的数据进行反序列化，然后将它从 JSON API 规范转换为更经典的 JSON，并从 snake_case 转换为 camelCase。它还处理我们收到的任何元信息。

此外，从流程的角度来看，它管理请求取消和调用 API 时的错误。

### [](#caching-the-data)缓存数据

至此，我们可以总结一下:`useApi`钩子获取数据，然后通过构建器进行验证和丰富；并且使用[反应查询](https://github.com/tannerlinsley/react-query)缓存结果数据。

我们实现了 react-query 来缓存前端的数据，使用 API 端点 URL 作为缓存键。React-query 使用上面提到的`useApi`钩子来获取、同步、更新和缓存远程数据，允许我们用非常小的代码库利用所有这些功能。

在此基础上，我们要做的就是实现 react-query 的提供者。为此，我们构建了一个小的 react 组件:

```
import { FC } from 'react';
import { QueryClient, QueryClientProvider, QueryClientProviderProps } from 'react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      refetchInterval: false,
      refetchIntervalInBackground: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: false,
    },
  },
});

type IProps = Omit<QueryClientProviderProps, 'client'> & {
  client?: QueryClient;
};

export const GlobalContextProvider: FC<IProps> = ({
  children,
  client = queryClient,
  ...props
}) => (
  <QueryClientProvider {...props} client={client}>
    {children}
  </QueryClientProvider>
);

```

最重要的是，它管理我们的缓存。我们有许多组件需要相同的数据，所以我们希望避免不必要的网络流量来检索相同的信息。性能始终是关键。限制执行不必要的网络调用的潜在错误也是如此。现在，有了缓存，如果一个组件请求数据，我们的缓存将存储该数据并将其提供给请求相同信息的其他组件。在后台，React-query 当然会确保缓存中的数据保持最新。

总而言之，这里有一个使用这个`useData`钩子和我们上面定义的应用程序模型构建的组件的例子:

```
import { FC } from 'react';

interface ApplicationProps {
  applicationId: string;
}

export const ApplicationCard: FC<ApplicationProps> = ({ applicationId }) => {
  const { loading, data: application, errors } = useData(applicationBuilder, `/applications/${applicationId}`);

  return loading ? (
    <div>loading...</div>
  ) : errors.length > 0 ? (
    <div>{errors.map(error => (<div>{error}</div>))}</div>
  ) : (
    <div>
      <div>{application.applicationId}</div>
      <div>{application.ownerEmail}</div>
      <div>{application.name}</div>
      <div>{application.plan}</div>
    </div>
  );
};

```

如你所见，我们的`useData`钩子让我们标准化加载和错误状态，鼓励我们编写处理这些状态的**可重用组件**。例如，我们有可重用的`StateCard`和`StateContainer`组件。有了现在容易获得的数据，我们就可以着手集成那些可重用的组件，并专注于构建一个出色的前端体验——干净、功能全面、可扩展。