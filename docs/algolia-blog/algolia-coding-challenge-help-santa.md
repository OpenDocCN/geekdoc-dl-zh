# Algolia 编码挑战:帮助圣诞老人获得灵感！🎅

> 原文：<https://www.algolia.com/blog/engineering/algolia-coding-challenge-help-santa/>

又到了一年中的这个时候了！我们很高兴在假期开始新的❄️编码挑战❄️。

🎅圣诞老人已经在家工作了 *个月* 试图为圣诞节做准备，他觉得他需要休息一下。我们需要帮助圣诞老人在他的伟大旅程开始前获得灵感！

使用 Algolia 搜索和相应的前端库回答下列问题，帮助圣诞老人回到正轨。

对于每位参与者，Algolia 将向 HackYourFuture 慈善机构捐款，这是一个为有才华的难民和其他受教育和进入劳动力市场机会有限的弱势群体提供的免费网络开发项目。

📱获胜者的奖品是一部 iPhone 13 或一部三星 S21，将被放在圣诞树下。

填写 [**此** **表格**](https://forms.gle/27yN673e5ncQNdMR6) 注册挑战，提交您的答案，并获得访问本次比赛的官方 Algolia Discord 服务器的权限。挑战赛的开放时间为 12 月 6 日至 12 月 19 日，太平洋标准时间晚上 11:59。如果你想加入这个挑战，但以前没有用过 Algolia，你可以[在这里免费注册](https://www.algolia.com/developers/get-started/)。

# [](#challenge-1-%f0%9f%8e%a4)**挑战#1🎤**

圣诞老人是《星际迷航》的超级粉丝。他记得他最喜欢的《星际迷航》演员有一次做了一个 Ted 演讲，有一个鼓舞人心的评分在 800 到 900 之间，被标记为 *讲故事* 。

他最喜欢的《星际迷航》演员的鼓励正是他所需要的！如果圣诞老人不是一个建设者，他什么都不是——也许通过构建一个类似于 [的搜索界面，这个演示](https://preview.algolia.com/ted/) 他就能找到话题！

看完视频后，圣诞老人很有感触 *于是* 启发他决定去拜访这位演员！

### [](#data-source)**数据源**

[**Ted 演讲**](https://github.com/algolia/datasets/blob/master/tedtalks/talks.json)

### [](#question)**问题**

*演讲者的名字是什么，* (此人也是著名的《星际迷航》演员)，那个圣诞老人要去拜访什么？*(通过[表单](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)提交)*

### [](#bonus-question)**加分题**

被标记为 *协作* 和 *社会* 的 TED 演讲中被观看次数最多的是什么？

# [](#challenge-2-%f0%9f%93%bd)**挑战#2📽**

在准备旅行时，圣诞老人想起了他“好”名单上的一个孩子小布莱恩，他要求一部由他计划拜访的男演员和女演员艾米·波勒主演的电影。圣诞老人需要另一个搜索界面来查找这部电影的片名(以及何时上映)。

也许他去见他们的时候可以让演员签一份拷贝！

### [](#data-source)**数据源**

**[电影](https://github.com/algolia/datasets/blob/master/movies/) :**

*这个数据集对于 Algolia 免费层来说太大了，所以您可能想直接使用托管版本* *，而不必向您的应用程序推送任何东西。*

*您可以使用以下凭证:*

**App ID:** `latency`

**API 键:** `56f24e4276091e774e8157fe4b8b11f6`

**指标名称:** `movies`

### [](#question)**提问**

布莱恩圣诞节想要的电影的 *片名**上映年份* 是什么？*(通过[表单](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)提交)*

### [](#bonus-question)**加分题**

什么是 *片名* 和 *评分最高的电影*2008 年上映的犯罪片？

# [](#challenge-3-%f0%9f%8d%b7)**挑战#3🍷**

在所有人当中，圣诞老人不可能不带礼物就出现在他最喜欢的《星际迷航》演员的门口！他决定耍点小聪明，给他们弄一瓶波尔多葡萄酒，产自*fron sac*领域(演员的最爱)，在上述电影上映的同一年装瓶。他想起了一个手边的[demo](https://github.com/algolia/hack-bdx/tree/node)他以前看过的类似的东西。

### [](#data-source)**数据来源**

**[酒](https://github.com/algolia/datasets/tree/master/wine/)**

### [](#question)**问题**

与电影《挑战#2》同年上映的 Fronsac *域* 中的波尔多葡萄酒 叫什么？*(通过[表单](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)提交)*

### [](#bonus-question)**加分题**

【2010 年起 30 美元以下的 *最高评级* 葡萄酒是什么？

# [](#challenge-4-%f0%9f%9b%a9)**挑战#4 🛩**

圣诞老人手里拿着酒，准备开始他的旅程。但是在哪里？他知道他最喜欢的《星际迷航》演员最近进行了以下飞行:

*   ATL → SEA
*   AHN → LGA
*   JFK → MCN

他可以使用这些信息和地图数据(类似于 t [他的演示](https://preview.algolia.com/geo-search/) )，来计算出明星所在的美国主要城市。圣诞老人会成为一名伟大的侦探！

### [](#data-source)**数据源**

[**机场**](https://github.com/algolia/datasets/tree/master/airports)

### [](#question)**问题**

最接近上述三个航班的美国主要城市是哪一个？*(通过[表单](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)提交)*

### [](#bonus-question)**加分题**

美国离卡森国家森林最近的机场是哪里？

# [](#challenge-5-%f0%9f%8e%b8)**挑战#5🎸**

一旦圣诞老人知道他要去哪里，他就用他的 UberReindeers 应用程序抓住雪橇，上路了！

当他到达这个城市时，圣诞老人想起了他还需要为“好”名单上的一个女孩挑选的另一份礼物。小朱莉想要的圣诞礼物就是 2019 年 8 月 14 日至 8 月 18 日期间在同一城市演出的音乐家的门票。

### [](#data-source)**数据源**

[**演唱会:**](https://github.com/algolia/datasets/blob/master/concerts)

*这个数据集对于 Algolia 免费层来说太大了，所以您可能想直接使用托管版本* *，而不必向您的应用程序推送任何东西。*

*您可以使用以下凭证:*

**App ID:** `latency`

**API 键:** `059c79ddd276568e990286944276464a`

**指标名称:** `concert_events_instantsearchjs`

### [](#question)**问题**

朱莉圣诞节想见到的艺术家的 *名字是什么* ？*(通过[表单](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)提交)*

### [](#bonus-question)**加分题**

酷玩乐队 2018 年最后一次 *城市* 在哪里演出？

# [](#conclusion-%f0%9f%8e%84)**结论🎄**

成功了！在与他最喜欢的演员(和一点点波尔多葡萄酒)进行了一次鼓舞人心的访问后，圣诞老人感到神清气爽，并为他的大日子做好了准备。他还能在最后一分钟完成圣诞购物，这总是一个额外的奖励！

感谢帮助圣诞老人，感谢帮助他人通过 [**向本次编码挑战提交您的答案**](https://forms.gle/27yN673e5ncQNdMR6) 。祝你假期愉快，新年再见！🥳

### [](#coding-challenge-official-rules)编码挑战官方规则

编码挑战的正式规则可在[这里](https://utm.io/ud2kQ)获得。通过参加编码挑战，您同意 Algolia 编码挑战的官方规则。