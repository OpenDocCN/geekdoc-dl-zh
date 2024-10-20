# 如何将 Power BI 数据导出到 Excel

> 原文：<https://web.archive.org/web/20221129052847/https://www.datacamp.com/blog/how-to-export-power-bi-data-to-excel>

Power BI 是当今使用最广泛的商业智能工具之一。由微软开发的 Power BI 可以让您管理、分析和可视化大量数据，而无需编码。使用 Power BI 的另一个主要优势是 Excel 专家能够从 Power BI 切换到 Excel，反之亦然。

Power BI 和 Excel 的工作方式有哪些不同？在本教程中，我们将探索将 Power BI 数据和报告导出到 Excel 的常见用例。我们开始吧！

## 将 Power BI 数据和报告导出到 Excel 的常见使用案例

出于几个原因，您需要将数据和报告从 Power BI 导出到 Excel:

*   **在 Excel 中分析数据:**即使 Power BI 在存储和处理数据方面拥有优于 Excel 的能力，使用 Power BI 的[在 Excel 中分析](https://web.archive.org/web/20221212140011/https://docs.microsoft.com/en-us/power-bi/collaborate-share/service-analyze-in-excel)功能，Excel 仍可用于创建报告和仪表板。而且，Excel 专家在不完全抛弃 Excel 的情况下，[逐渐学习 Power BI](https://web.archive.org/web/20221212140011/https://www.datacamp.com/learn/power-bi) 可能更直观。
*   **数据导出:**一位同事对报告背后的数据感兴趣，但他们没有安装 Power BI。将基础数据导出到 Excel 是一个简单的解决方案。
*   **跨工具使用:**如果你在一个有各种类型数据从业者的团队中工作。某些团队成员可能会使用其他工具，比如 R 或 Python。将 Power BI 数据和报告导出到 Excel 将允许他们这样做。

## 从 Power BI 仪表板导出数据

![Exporting Data from a Power BI Dashboard](img/0e63753b12bda280187db372b0029899.png)

[Power BI 仪表板](https://web.archive.org/web/20221212140011/https://www.datacamp.com/blog/best-practices-for-designing-dashboards)允许您将可视化、报告和表格放在一个地方，一目了然。但是，您可能经常想要研究仪表板中可视化背后的底层数据，或者将其发送给同事。以下是[从 Power BI 仪表板](https://web.archive.org/web/20221212140011/https://docs.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-export-data?tabs=powerbi-desktop)导出数据的步骤:

*   转到 [](https://web.archive.org/web/20221212140011/https://powerbi.microsoft.com/en-us) 您的 Power BI 实例，使用您的帐户凭证登录。如果您没有帐户，您需要创建一个。
*   转到有问题的 Power BI 仪表板，选择您有兴趣从中导出数据的数据可视化。
*   点击所选数据可视化右上角的**更多选项(…)** 。
*   选择选项**导出到. csv.**
*   然后你可以打开这个**。Excel 中的 csv** 文件。

## 从 Power BI 报告中导出数据

![Exporting Data from a Power BI Report](img/5c1d35e7c360ee2c967b9707d10783f5.png)

[Power BI 报告](https://web.archive.org/web/20221212140011/https://docs.microsoft.com/en-us/power-bi/consumer/end-user-reports)经常与仪表板混淆，因为它们都包含数据可视化和表格。报告可以有多个页面，并允许查看者找到不同的方式来过滤、突出显示和切片数据。此外，它们是创建您的综合摘要以与利益相关者共享的绝佳选择。

从 Power BI 报告中导出数据的步骤如下:

*   从 Power BI 桌面访问报告。如果您没有安装 Power BI Desktop，请转到[此处](https://web.archive.org/web/20221212140011/https://www.microsoft.com/en-us/download/details.aspx?id=58494)。
*   选择您有兴趣从中导出数据的数据可视化。
*   单击所选数据可视化的切片设置
*   选择选项**导出数据**。

## 将功率 BI 表复制到 Excel 中

![Copying Power BI Tables into Excel](img/638b08dbc6bb87a252a806f932d2140d.png)

将 Power BI 表复制到 Excel 的步骤非常简单直观:

*   转到 Power BI Desktop。
*   选择您想要的功率 BI 表，并转到左侧面板上的**数据视图**选项。
*   右键单击所选的表；在这种情况下，它被称为“销售”
*   选择“复制表格”选项
*   在 Excel 中创建一个新的工作表，点击粘贴图标或 **Ctrl + V** 粘贴表格内容

## 使用“在 Excel 中分析”功能从 Power BI 导出到 Excel

在 Power BI 服务中，有一个附加功能可以将 Power BI 数据集导入 Excel。这对于将数据处理到 Excel 中，然后使用 Excel 中已处理的数据生成数据可视化效果非常有用。这是利用 Power BI 的“[在 Excel 中分析](https://web.archive.org/web/20221212140011/https://docs.microsoft.com/en-us/power-bi/collaborate-share/service-analyze-in-excel)”功能的三种不同方式。

### 1.我的工作区中的 Excel 分析功能

![Analyze in Excel feature from My Workspace](img/c18e14a4f9042387cf89be4d0ea5e09f.png)

Power BI 中的“我的工作区”是一个视图，包含您过去创建的数据集、报告和仪表板的集合。您可以使用 Excel 中的分析功能将这些数据集和报表中的任何一个导出到 Excel 中。方法如下:

1.  前往[app.powerbi.com](https://web.archive.org/web/20221212140011/https://powerbi.microsoft.com/en-us)，在那里你所有的作品集都会出现在你的 Power BI 账户上。
2.  从菜单中选择下载。如果您还没有这样做，请点击**在 Excel 中分析更新**。此操作是强制性的，否则，在 Excel 中分析功能将不起作用。
3.  准备就绪后，选择要在 Excel 中分析的 Power BI 数据集。
4.  选择数据集旁边的**更多选项(…)** ，然后点击“在 Excel 中分析”。
5.  打开新的 excel 文件时，启用编辑和内容。如果您在 Excel 更新中安装了 Analyze，应该不会有问题。但如果您仍有问题，请查看此[文章](https://web.archive.org/web/20221212140011/https://docs.microsoft.com/en-us/power-bi/collaborate-share/desktop-troubleshooting-analyze-in-excel#connection-cannot-be-made-error)以获得进一步的指导。

### 2.从数据集视图分析 Excel 功能

![Analyze in Excel feature from the view of the dataset](img/2c70664a906294b544f74e80801ac8ee.png)

另一种方法是在工作场所中单击数据集的名称。将打开一个新页面，您可以在页面上方的菜单栏中选择“在 Excel 中分析”。

### 3.报表中的 Excel 分析功能

![Analyze in Excel feature from the report](img/852aec4e9cee362f3088170ed7641fdf.png)

第三种也是最后一种方法是打开一个 Power BI 报告，并在菜单栏中选择 Export → Analyze in Excel。

## 更多电源 BI 资源

我们希望本教程对你有用。本文讨论了将 Power BI 数据导出到 Excel 的不同方法。当您希望在 Power BI 之外存储和分析数据时，该功能非常有用。您可以利用 DataCamp 的资源更深入地了解 Power BI。

*   [[BLOG]什么是 Power BI？Power BI 的完整指南](https://web.archive.org/web/20221212140011/https://www.datacamp.com/blog/all-about-power-bi)
*   [【网上研讨会】成为拥有 Power BI 的数据分析师](https://web.archive.org/web/20221212140011/https://www.datacamp.com/resources/webinars/become-data-analyst-with-power-bi)
*   [【博客】如何从 Excel 过渡到 Power BI](https://web.archive.org/web/20221212140011/https://www.datacamp.com/blog/how-to-transition-from-excel-to-power-bi)
*   [[博客]权力 BI vs Tableau:你该选哪个？](https://web.archive.org/web/20221212140011/https://www.datacamp.com/blog/power-bi-vs-tableau-which-one-should-you-choose)