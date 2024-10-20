# 用 Algolia 构建 GitHub 问题推荐机器人

> 原文：<https://www.algolia.com/blog/engineering/building-a-github-issue-recommendation-bot-with-algolia/>

GitHub 问题是静态内容。如果他们不必如此呢？

当我们(DevRels Chuck Meyer 和 Bryan Robinson)发现 Dev.to 正在举办一个 [GitHub Actions 黑客马拉松](https://dev.to/devteam/join-us-for-the-2021-github-actions-hackathon-on-dev-4hn4)时，我们知道我们需要尝试一下。

我们知道，我们希望找到一个有用的工具，将 Algolia 融入到行动中。对于承担什么样的项目有明显的想法。我们想通了索引内容、产品或降价的通常做法。它们都会对网站创建者有所帮助。但是，它们对开源维护者有帮助吗？大概？

我们如何改善他们的整体工作流程？

然后我们突然想到:如果我们能为常见问题提供推荐问题会怎么样？我们能减轻维护人员回答类似问题的负担吗？在大型存储库中，有多少问题作为“重复”关闭？Algolia 能为问题创建者提供一个相关的、有用的问题列表吗？

剧透:是的，完全正确！

## [](#the-workflow-structure)工作流结构

当开发人员向存储库添加问题时，我们需要执行三个步骤。

首先，我们需要搜索相关问题的 Algolia 索引。然后，我们将这些结果捆绑到 Markdown 中，并将其传递给一个操作，以创建对初始问题的评论。最后，我们需要将该问题放入我们的索引中，以便将来进行搜索。

这些步骤中的每一步都需要一个动作。阿尔戈利亚特有的动作，我们需要从头开始创造。评论编写动作，我们决定使用令人惊叹的 Peter Evan 的[创建或更新评论动作](https://github.com/marketplace/actions/create-or-update-comment)——事实证明，GitHub 在他们的许多关于动作的文档中都使用了这个动作。

让我们开始新的行动。

## [](#performing-a-search-query)执行搜索查询

我们工作流程的第一步是向 Algolia 发送一个搜索查询。我们为此创建了一个自定义操作[获取 Algolia 问题记录](https://github.com/marketplace/actions/get-algolia-issue-records)。

要使用这个动作，我们需要向它发送四个必需的输入(和可选的第五个)。

*   `app_id`:您的 Algolia 账户中应用程序的 ID。这最好作为一个秘密存储在您的存储库中
*   `api_key`:对你的 Algolia 应用中的索引具有搜索权限的 API 键。这是最好的储存在一个秘密在您的储存库。
*   `index_name`:要搜索的 Algolia 索引的名称。为了保持一致性，我们推荐使用`github.event.repository.name`变量。
*   `issue_title`:用`github.event.issue.title`找到的煽动性问题的标题。
*   `max_results`:(可选)返回评论的结果数(默认为 3)

我们获取这些变量，并基于煽动性问题的标题执行`similarQuery`搜索。然后我们在 Markdown(GitHub 注释所需的格式)中创建一个注释体和一个条目列表。该输出被传递给 [Peter Evans 的创建或更新评论操作](https://github.com/marketplace/actions/create-or-update-comment)。

```
const { inspect } = require('util');
const core = require('@actions/core');
const algoliasearch = require('algoliasearch');

async function run() {
  try {
    const inputs = {
      appId: core.getInput('app_id'),
      apiKey: core.getInput('api_key'),
      indexName: core.getInput('index_name'),
      issueTitle: core.getInput('issue_title'),
      maxResults: core.getInput('max_results'),
    };
    core.info(`Inputs: ${inspect(inputs)}`);

    if (!inputs.appId && !inputs.apiKey && !inputs.indexName) {
      core.setFailed('Missing one or more of Algolia app id, API key, or index name.');
      return;
    }

    inputs.maxResults = inputs.maxResults || 3;

    const client = algoliasearch(inputs.appId, inputs.apiKey);
    const index = client.initIndex(inputs.indexName);

    index.search('', { 
        similarQuery: inputs.issueTitle,
        hitsPerPage: inputs.maxResults
      }).then(({hits}) => {
      core.info(`Searching for record`);
      core.info(`Hits: ${inspect(hits)}`);
      const message = `## Other issues similar to this one:\n${hits.map(hit => `* [${hit.title}](${hit.url})`).join('\n')}`
      const listItems = `${hits.map(hit => `* [${hit.title}](${hit.url})`).join('\n')}\n`
      core.info(message)
      core.info(listItems)
      core.setOutput('comment_body', message);
      core.setOutput('issues_list', listItems);
    })
      .catch(err => {
        core.setFailed(err.message);
      }
    );
  } catch (error) {
    core.debug(inspect(error));
    core.setFailed(error.message);
    if (error.message == 'Resource not accessible by integration') {
      core.error(`See this action's readme for details about this error`);
    }
  }
}

run();
```

## [](#adding-the-issue-to-your-algolia-index)将问题添加到您的 Algolia 索引中

对于我们工作流程的最后一步，我们将这个新问题添加到 Algolia 索引中，以供将来搜索。我们为此创建了另一个 GitHub 动作:[创建或更新 Algolia 索引记录](https://github.com/marketplace/actions/create-or-update-algolia-index-record)。这个动作自动地直接向索引添加/更新记录，而不是从 JSON 文件中写入/读取。这在我们处理关于回购的元数据(问题、拉请求、评论)而不是为应用程序本身构建索引的情况下是有意义的。

要使用这个动作，我们需要[创建一个 Algolia API 键](https://www.algolia.com/doc/guides/security/api-keys/#creating-and-managing-api-keys))，它具有在我们的索引中添加/更新记录的权限。此外，我们需要获得许可，才能为回购创建新的指数。否则，我们必须提前创建它，并将索引名硬编码到我们的配置中。

除了新的 API 键，我们还需要一些其他输入来使用该操作:

*   根据上面的操作，你应该已经把它作为一个秘密保存在你的存储库中了
*   `api_key`:这是允许将记录保存到索引的新密钥。这是最好的储存在一个秘密在您的储存库。
*   `index_name`:添加/更新此记录的 Algolia 索引的名称。为了保持一致性，我们推荐使用`github.event.repository.name`变量。
*   `record`:表示要添加到索引中的 JSON 记录的字符串。

如果 API 键有权限，该操作将为存储库创建一个索引。我们将添加问题标题和 URL(链接回来)作为`record`。在我们的工作流程中，这是一个多行字符串，但必须是有效的 JSON，动作才能工作(详情请参见[准备数据](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/#algolia-records))。

我们接受所有这些输入，并通过 Algolia API 执行一个`saveObject`调用。我们使用`issue ID`作为索引中的`objectID`。如果我们以后添加工作流来更新或删除事件，这使得将记录与此问题联系起来变得容易。

```
const { inspect } = require('util');
const core = require('@actions/core');
const algoliasearch = require('algoliasearch');

async function run() {
  try {
    const inputs = {
      appId: core.getInput('app_id'),
      apiKey: core.getInput('api_key'),
      indexName: core.getInput('index_name'),
      record: core.getInput('record'),
    };
    core.debug(`Inputs: ${inspect(inputs)}`);

    if (!inputs.appId && !inputs.apiKey && !inputs.indexName) {
      core.setFailed('Missing one or more of Algolia app id, API key, or index name.');
      return;
    }

    core.info(`Writing record to index ${inputs.indexName}`)
    const client = algoliasearch(inputs.appId, inputs.apiKey);
    const index = client.initIndex(inputs.indexName);

    index.saveObject(JSON.parse(inputs.record), {'autoGenerateObjectIDIfNotExist': true})
      .then(({ objectID }) => {
        core.setOutput('object_id', objectID);
        core.info(
          `Created record in index ${inputs.indexName} with objectID ${objectID}.`
        );
      })
      .catch((err) => {
        core.setFailed(`Failed to save object: ${err}`);
      });

  } catch (error) {
    core.debug(inspect(error));
    core.setFailed(error.message);
    if (error.message == 'Resource not accessible by integration') {
      core.error(`See this action's readme for details about this error`);
    }
  }
}

run();
```

接下来，我们将这两个新操作与现有的评论创建操作组合在一起，以构建我们的工作流。

## [](#the-full-workflow-file)完整的工作流程文件

为了完成这项工作，我们需要一个`job`和三个`steps`。每个步骤都将使用这些操作之一。

```
name: related-issues
on:
  # Triggers the workflow on push or pull request events but only for the main branch
  issues:
    types: 
      - opened

jobs:
  get-related-issues:
    permissions: 
      # Gives the workflow write permissions only in issues
      issues: write
    runs-on: ubuntu-latest
    steps:
      # Performs a search in an Algolia Index based on Issue Title
      # The Index should have historical Issues
      # Returns two outputs:
      # issues_list: a markdown list of issues
      # comment_body: a generic comment body with the list of issues
      - id: search
        name: Search based on issue title
        uses: brob/algolia-issue-search@v1.0
        with: 
          # Requires an Algolia account with an App ID and write key
          app_id: ${{ secrets.ALGOLIA_APP_ID }}
          api_key: ${{ secrets.ALGOLIA_API_KEY }}
          index_name: ${{ github.event.repository.name }}
          issue_title: ${{ github.event.issue.title }}
      - name: Create or Update Comment
        uses: peter-evans/create-or-update-comment@v1.4.5
        with:
          # GITHUB_TOKEN or a repo scoped PAT.
          token: ${{ github.token }}
          # The number of the issue or pull request in which to create a comment.
          issue-number: ${{ github.event.issue.number }}
          # The comment body. Can use either issues_list or comment_body
          body: |
            # While you wait, here are related issues:
            ${{ steps.search.outputs.issues_list }}
            Thank you so much! We'll be with you shortly!
      # An Action to create a record in an Algolia Index
      # This is a generic Action and can be used outside of this workflow
      - name: Add Algolia Record
        id: ingest
        uses: chuckmeyer/add-algolia-record@v1
        with:
          app_id: ${{ secrets.ALGOLIA_APP_ID }}
          api_key: ${{ secrets.ALGOLIA_API_KEY }}
          index_name: ${{ github.event.repository.name }}
          # Record needs to be a string of JSON
          record: |
            {
              "title": "${{ github.event.issue.title }}", 
              "url": "${{ github.event.issue.html_url }}", 
              "labels": "${{ github.event.issue.labels }}",
              "objectID": "${{ github.event.issue.number }}"
            }
```

## [](#next-steps)下一步

我们希望这对维护者有所帮助，但我们也希望它能启发其他人找到更好的方法来建议静态领域的内容，如 GitHub 问题。

如果您想体验完整的工作流程，可以在[这个资源库](https://github.com/brob/github-issues-search-bot)中查看。在 [GitHub marketplace](https://github.com/marketplace?type=actions&query=Algolia) 中，搜索和摄取操作都是可用的。

搜索和发现可以成为 GitHub 及其他自动化工作流程中有趣的一部分。在我们的开源代码交换平台上查看相关解决方案。

[![](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/?query=github&page=1)