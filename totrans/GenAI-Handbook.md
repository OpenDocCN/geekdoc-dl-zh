<!--yml
category: 未分类
date: 2026-01-04 09:39:43
-->

# GenAI Handbook

> 来源：[https://genai-handbook.github.io/](https://genai-handbook.github.io/)

William Brown
[@willccbb](https://x.com/willccbb) | [willcb.com](https://willcb.com)
[v0.1](https://github.com/genai-handbook/genai-handbook.github.io) (June 5, 2024)

# Introduction

This document aims to serve as a handbook for learning the key concepts underlying modern artificial intelligence systems. Given the speed of recent development in AI, there really isn’t a good textbook-style source for getting up-to-speed on the latest-and-greatest innovations in LLMs or other generative models, yet there is an abundance of great explainer resources (blog posts, videos, etc.) for these topics scattered across the internet. My goal is to organize the “best” of these resources into a textbook-style presentation, which can serve as a roadmap for filling in the prerequisites towards individual AI-related learning goals. My hope is that this will be a “living document”, to be updated as new innovations and paradigms inevitably emerge, and ideally also a document that can benefit from community input and contribution. This guide is aimed at those with a technical background of some kind, who are interested in diving into AI either out of curiosity or for a potential career. I’ll assume that you have some experience with coding and high-school level math, but otherwise will provide pointers for filling in any other prerequisites. Please let me know if there’s anything you think should be added!

## The AI Landscape

As of June 2024, it’s been about 18 months since [ChatGPT](http://chat.openai.com) was released by [OpenAI](https://openai.com/) and the world started talking a lot more about artificial intelligence. Much has happened since: tech giants like [Meta](https://llama.meta.com/) and [Google](https://gemini.google.com/) have released large language models of their own, newer organizations like [Mistral](https://mistral.ai/) and [Anthropic](https://www.anthropic.com/) have proven to be serious contenders as well, innumerable startups have begun building on top of their APIs, everyone is [scrambling](https://finance.yahoo.com/news/customer-demand-nvidia-chips-far-013826675.html) for powerful Nvidia GPUs, papers appear on [ArXiv](https://arxiv.org/list/cs.AI/recent) at a breakneck pace, demos circulate of [physical robots](https://www.figure.ai/) and [artificial programmers](https://www.cognition-labs.com/introducing-devin) powered by LLMs, and it seems like [chatbots](https://www.businessinsider.com/chat-gpt-effect-will-likely-mean-more-ai-chatbots-apps-2023-2) are finding their way into all aspects of online life (to varying degrees of success). In parallel to the LLM race, there’s been rapid development in image generation via diffusion models; [DALL-E](https://openai.com/dall-e-3) and [Midjourney](https://www.midjourney.com/showcase) are displaying increasingly impressive results that often stump humans on social media, and with the progress from [Sora](https://openai.com/sora), [Runway](https://runwayml.com/), and [Pika](https://pika.art/home), it seems like high-quality video generation is right around the corner as well. There are ongoing debates about when “AGI” will arrive, what “AGI” even means, the merits of open vs. closed models, value alignment, superintelligence, existential risk, fake news, and the future of the economy. Many are concerned about jobs being lost to automation, or excited about the progress that automation might drive. And the world keeps moving: chips get faster, data centers get bigger, models get smarter, contexts get longer, abilities are augmented with tools and vision, and it’s not totally clear where this is all headed. If you’re following “AI news” in 2024, it can often feel like there’s some kind of big new breakthrough happening on a near-daily basis. It’s a lot to keep up with, especially if you’re just tuning in.

With progress happening so quickly, a natural inclination by those seeking to “get in on the action” is to pick up the latest-and-greatest available tools (likely [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Gemini 1.5 Pro](https://deepmind.google/technologies/gemini/pro/), or [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family) as of this writing, depending on who you ask) and try to build a website or application on top of them. There’s certainly a lot of room for this, but these tools will change quickly, and having a solid understanding of the underlying fundamentals will make it much easier to get the most out of your tools, pick up new tools quickly as they’re introduced, and evaluate tradeoffs for things like cost, performance, speed, modularity, and flexibility. Further, innovation isn’t only happening at the application layer, and companies like [Hugging Face](https://huggingface.co/), [Scale AI](https://scale.com/), and [Together AI](https://www.together.ai/) have gained footholds by focusing on inference, training, and tooling for open-weights models (among other things). Whether you’re looking to get involved in open-source development, work on fundamental research, or leverage LLMs in settings where costs or privacy concerns preclude outside API usage, it helps to understand how these things work under the hood in order to debug or modify them as needed. From a broader career perspective, a lot of current “AI/ML Engineer” roles will value nuts-and-bolts knowledge in addition to high-level frameworks, just as “Data Scientist” roles have typically sought a strong grasp on theory and fundamentals over proficiency in the ML framework *du jour*. Diving deep is the harder path, but I think it’s a worthwhile one. But with the pace at which innovation has occurred over the past few years, where should you start? Which topics are essential, what order should you learn them in, and which ones can you skim or skip?

## The Content Landscape

Textbooks are great for providing a high-level roadmap of fields where the set of “key ideas” is more stable, but as far as I can tell, there really isn’t a publicly available post-ChatGPT “guide to AI” with textbook-style comprehensiveness or organization. It’s not clear that it would even make sense for someone to write a traditional textbook covering the current state of AI right now; many key ideas (e.g. QLoRA, DPO, vLLM) are no more than a year old, and the field will likely have changed dramatically by the time it’d get to print. The oft-referenced [Deep Learning](https://www.deeplearningbook.org/) book (Goodfellow et al.) is almost a decade old at this point, and has only a cursory mention of language modeling via RNNs. The newer [Dive into Deep Learning](http://d2l.ai) book includes coverage up to Transformer architectures and fine-tuning for BERT models, but topics like RLHF and RAG (which are “old” by the standards of some of the more bleeding-edge topics we’ll touch on) are missing. The upcoming [“Hands-On Large Language Models”](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) book might be nice, but it’s not officially published yet (available online behind a paywall now) and presumably won’t be free when it is. The Stanford [CS224n](https://web.stanford.edu/class/cs224n/index.html#coursework) course seems great if you’re a student there, but without a login you’re limited to slide-decks and a reading list consisting mostly of dense academic papers. Microsoft’s [“Generative AI for Beginners”](https://microsoft.github.io/generative-ai-for-beginners/#/) guide is fairly solid for getting your hands dirty with popular frameworks, but it’s more focused on applications rather than understanding the fundamentals.

The closest resource I’m aware of to what I have in mind is Maxime Labonne’s [LLM Course](https://github.com/mlabonne/llm-course) on Github. It features many interactive code notebooks, as well as links to sources for learning the underlying concepts, several of which overlap with what I’ll be including here. I’d recommend it as a primary companion guide while working through this handbook, especially if you’re interested in applications; this document doesn’t include notebooks, but the scope of topics I’m covering is a bit broader, including some research threads which aren’t quite “standard” as well as multimodal models.

Still, there’s an abundance of other high-quality and accessible content which covers the latest advances in AI — it’s just not all organized. The best resources for quickly learning about new innovations are often one-off blog posts or YouTube videos (as well as Twitter/X threads, Discord servers, and discussions on Reddit and LessWrong). My goal with this document is to give a roadmap for navigating all of this content, organized into a textbook-style presentation without reinventing the wheel on individual explainers. Throughout, I’ll include multiple styles of content where possible (e.g. videos, blogs, and papers), as well as my opinions on goal-dependent knowledge prioritization and notes on “mental models” I found useful when first encountering these topics.

I’m creating this document **not** as a “generative AI expert”, but rather as someone who’s recently had the experience of ramping up on many of these topics in a short time frame. While I’ve been working in and around AI since 2016 or so (if we count an internship project running evaluations for vision models as the “start”), I only started paying close attention to LLM developments 18 months ago, with the release of ChatGPT. I first started working with open-weights LLMs around 12 months ago. As such, I’ve spent a lot of the past year sifting through blog posts and papers and videos in search of the gems; this document is hopefully a more direct version of that path. It also serves as a distillation of many conversations I’ve had with friends, where we’ve tried to find and share useful intuitions for grokking complex topics in order to expedite each other’s learning. Compiling this has been a great forcing function for filling in gaps in my own understanding as well; I didn’t know how FlashAttention worked until a couple weeks ago, and I still don’t think that I really understand state-space models that well. But I know a lot more than when I started.

## Resources

Some of the sources we’ll draw from are:

*   Blogs:
    *   [Hugging Face](https://huggingface.co/blog) blog posts
    *   [Chip Huyen](https://huyenchip.com/blog/)’s blog
    *   [Lilian Weng](https://lilianweng.github.io/)’s blog
    *   [Tim Dettmers](https://timdettmers.com/)’ blog
    *   [Towards Data Science](https://towardsdatascience.com/)
    *   [Andrej Karpathy](https://karpathy.github.io/)’s blog
    *   Sebastian Raschka’s [“Ahead of AI”](https://magazine.sebastianraschka.com/) blog
*   YouTube:
    *   Andrej Karpathy’s [“Zero to Hero”](https://karpathy.ai/zero-to-hero.html) videos
    *   [3Blue1Brown](https://www.youtube.com/c/3blue1brown) videos
    *   Mutual Information
    *   StatQuest
*   Textbooks
    *   The [d2l.ai](http://d2l.ai) interactive textbook
    *   The [Deep Learning](https://www.deeplearningbook.org/) textbook
*   Web courses:
    *   Maxime Labonne’s [LLM Course](https://github.com/mlabonne/llm-course)
    *   Microsoft’s [“Generative AI for Beginners”](https://microsoft.github.io/generative-ai-for-beginners/#/)
    *   Fast.AI’s [“Practical Deep Learning for Coders”](https://course.fast.ai/)
*   Assorted university lecture notes
*   Original research papers (sparingly)

I’ll often make reference to the original papers for key ideas throughout, but our emphasis will be on expository content which is more concise and conceptual, aimed at students or practitioners rather than experienced AI researchers (although hopefully the prospect of doing AI research will become less daunting as you progress through these sources). Pointers to multiple resources and media formats will be given when possible, along with some discussion on their relative merits.

## Preliminaries

### Math

Calculus and linear algebra are pretty much unavoidable if you want to understand modern deep learning, which is largely driven by matrix multiplication and backpropagation of gradients. Many technical people end their formal math educations around multivariable calculus or introductory linear algebra, and it seems common to be left with a sour taste in your mouth from having to memorize a suite of unintuitive identities or manually invert matrices, which can be discouraging towards the prospect of going deeper in one’s math education. Fortunately, we don’t need to do these calculations ourselves — programming libraries will handle them for us — and it’ll instead be more important to have a working knowledge of concepts such as:

*   Gradients and their relation to local minima/maxima
*   The chain rule for differentiation
*   Matrices as linear transformations for vectors
*   Notions of basis/rank/span/independence/etc.

Good visualizations can really help these ideas sink in, and I don’t think there’s a better source for this than these two YouTube series from [3Blue1Brown](https://www.youtube.com/@3blue1brown/playlists):

*   [Essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&pp=iAQB)
*   [Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&pp=iAQB)

If your math is rusty, I’d certainly encourage (re)watching these before diving in deeper. To test your understanding, or as a preview of where we’re headed, the shorter [Neural networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) video series on the same channel is excellent as well, and the latest couple videos in the series give a great overview of Transformer networks for language modeling.

These [lecture notes](https://links.uwaterloo.ca/math227docs/set4.pdf) from Waterloo give some useful coverage of multivariable calculus as it relates to optimization, and [“Linear Algebra Done Right”](https://linear.axler.net/LADR4e.pdf) by Sheldon Axler is a nice reference text for linear algebra. [“Convex Optimization”](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) by Boyd and Vandenberghe shows how these topics lay the foundations for the kinds of optimization problems faced in machine learning, but note that it does get fairly technical, and may not be essential if you’re mostly interested in applications.

Linear programming is certainly worth understanding, and is basically the simplest kind of high-dimensional optimization problem you’ll encounter (but still quite practical); this illustrated [video](https://www.youtube.com/watch?v=E72DWgKP_1Y) should give you most of the core ideas, and Ryan O’Donnell’s [videos](https://www.youtube.com/watch?v=DYAIIUuAaGA&list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&index=66) (17a-19c in this series, depending on how deep you want to go) are excellent if you want to go deeper into the math. These lectures ([#10](https://www.youtube.com/watch?v=v-chgwlwqTk), [#11](https://www.youtube.com/watch?v=IWzErYm8Lyk)) from Tim Roughgarden also show some fascinating connections between linear programming and the “online learning” methods we’ll look at [later](#online-learning-and-regret-minimization), which will form the conceptual basis for [GANs](#generative-adversarial-nets) (among many other things).

### Programming

Most machine learning code is written in Python nowadays, and some of the references here will include Python examples for illustrating the discussed topics. If you’re unfamiliar with Python, or programming in general, I’ve heard good things about Replit’s [100 Days of Python](https://replit.com/learn/100-days-of-python) course for getting started. Some systems-level topics will also touch on implementations in C++ or CUDA — I’m admittedly not much of an expert in either of these, and will focus more on higher-level abstractions which can be accessed through Python libraries, but I’ll include potentially useful references for these languages in the relevant sections nonetheless.

## Organization

This document is organized into several sections and chapters, as listed below and in the sidebar. You are encouraged to jump around to whichever parts seem most useful for your personal learning goals. Overall, I’d recommend first skimming many of the linked resources rather than reading (or watching) word-for-word. This should hopefully at least give you a sense of where your knowledge gaps are in terms of dependencies for any particular learning goals, which will help guide a more focused second pass.

# Section I: Foundations of Sequential Prediction

**Goal:** Recap machine learning basics + survey (non-DL) methods for tasks under the umbrella of “sequential prediction”.

Our focus in this section will be on quickly overviewing classical topics in statistical prediction and reinforcement learning, which we’ll make direct reference to in later sections, as well as highlighting some topics that I think are very useful as conceptual models for understanding LLMs, yet which are often omitted from deep learning crash courses – notably time-series analysis, regret minimization, and Markov models.

## Statistical Prediction and Supervised Learning

Before getting to deep learning and large language models, it’ll be useful to have a solid grasp on some foundational concepts in probability theory and machine learning. In particular, it helps to understand:

*   Random variables, expectations, and variance
*   Supervised vs. unsupervised learning
*   Regression vs. classification
*   Linear models and regularization
*   Empirical risk minimization
*   Hypothesis classes and bias-variance tradeoffs

For general probability theory, having a solid understanding of how the Central Limit Theorem works is perhaps a reasonable litmus test for how much you’ll need to know about random variables before tackling some of the later topics we’ll cover. This beautifully-animated 3Blue1Brown [video](https://www.youtube.com/watch?v=zeJD6dqJ5lo) is a great starting point, and there are a couple other good probability videos to check out on the channel if you’d like. This set of [course notes](https://blogs.ubc.ca/math105/discrete-random-variables/) from UBC covers the basics of random variables. If you’re into blackboard lectures, I’m a big fan of many of Ryan O’Donnell’s CMU courses on YouTube, and this [video](https://www.youtube.com/watch?v=r9S2fMQiP2E&list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&index=13) on random variables and the Central Limit Theorem (from the excellent “CS Theory Toolkit” course) is a nice overview.

For understanding linear models and other key machine learning principles, the first two chapters of Hastie’s [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf) (”Introduction” and “Overview of Supervised Learning”) should be enough to get started.

Once you’re familiar with the basics, this [blog post](https://ryxcommar.com/2019/09/06/some-things-you-maybe-didnt-know-about-linear-regression/) by anonymous Twitter/X user [@ryxcommar](https://twitter.com/ryxcommar) does a nice job discussing some common pitfalls and misconceptions related to linear regression. [StatQuest](https://www.youtube.com/@statquest/playlists) on YouTube has a number of videos that might also be helpful.

Introductions to machine learning tend to emphasize linear models, and for good reason. Many phenomena in the real world are modeled quite well by linear equations — the average temperature over past 7 days is likely a solid guess for the temperature tomorrow, barring any other information about weather pattern forecasts. Linear systems and models are a lot easier to study, interpret, and optimize than their nonlinear counterparts. For more complex and high-dimensional problems with potential nonlinear dependencies between features, it’s often useful to ask:

*   What’s a linear model for the problem?
*   Why does the linear model fail?
*   What’s the best way to add nonlinearity, given the semantic structure of the problem?

In particular, this framing will be helpful for motivating some of the model architectures we’ll look at later (e.g. LSTMs and Transformers).

## Time-Series Analysis

How much do you need to know about time-series analysis in order to understand the mechanics of more complex generative AI methods?

**Short answer:** just a tiny bit for LLMs, a good bit more for diffusion.

For modern Transformer-based LLMs, it’ll be useful to know:

*   The basic setup for sequential prediction problems
*   The notion of an autoregressive model

There’s not really a coherent way to “visualize” the full mechanics of a multi-billion-parameter model in your head, but much simpler autoregressive models (like ARIMA) can serve as a nice mental model to extrapolate from.

When we get to neural [state-space models](#structured-state-space-models), a working knowledge of linear time-invariant systems and control theory (which have many connections to classical time-series analysis) will be helpful for intuition, but [diffusion](#diffusion-models) is really where it’s most essential to dive deeper into into stochastic differential equations to get the full picture. But we can table that for now.

This blog post ([Forecasting with Stochastic Models](https://towardsdatascience.com/forecasting-with-stochastic-models-abf2e85c9679)) from Towards Data Science is concise and introduces the basic concepts along with some standard autoregressive models and code examples.

This set of [notes](https://sites.ualberta.ca/~kashlak/data/stat479.pdf) from UAlberta’s “Time Series Analysis” course is nice if you want to go a bit deeper on the math.

## Online Learning and Regret Minimization

It’s debatable how important it is to have a strong grasp on regret minimization, but I think a basic familiarity is useful. The basic setting here is similar to supervised learning, but:

*   Points arrive one-at-a-time in an arbitrary order
*   We want low average error across this sequence

If you squint and tilt your head, most of the algorithms designed for these problems look basically like gradient descent, often with delicate choices of regularizers and learning rates require for the math to work out. But there’s a lot of satisfying math here. I have a soft spot for it, as it relates to a lot of the research I worked on during my PhD. I think it’s conceptually fascinating. Like the previous section on time-series analysis, online learning is technically “sequential prediction” but you don’t really need it to understand LLMs.

The most direct connection to it that we’ll consider is when we look at [GANs](#generative-adversarial-nets) in Section VIII. There are many deep connections between regret minimization and equilibria in games, and GANs work basically by having two neural networks play a game against each other. Practical gradient-based optimization algorithms like Adam have their roots in this field as well, following the introduction of the AdaGrad algorithm, which was first analyzed for online and adversarial settings. In terms of other insights, one takeaway I find useful is the following: If you’re doing gradient-based optimization with a sensible learning rate schedule, then the order in which you process data points doesn’t actually matter much. Gradient descent can handle it.

I’d encourage you to at least skim Chapter 1 of [“Introduction to Online Convex Optimization”](https://arxiv.org/pdf/1909.05207.pdf) by Elad Hazan to get a feel for the goal of regret minimization. I’ve spent a lot of time with this book and I think it’s excellent.

## Reinforcement Learning

Reinforcement Learning (RL) will come up most directly when we look at finetuning methods in [Section IV](#finetuning-methods-for-llms), and may also be a useful mental model for thinking about “agent” [applications](#tool-use-and-agents) and some of the “control theory” notions which come up for [state-space models](#structured-state-space-models). Like a lot of the topics discussed in this document, you can go quite deep down many different RL-related threads if you’d like; as it relates to language modeling and alignment, it’ll be most important to be comfortable with the basic problem setup for Markov decision processes, notion of policies and trajectories, and high-level understanding of standard iterative + gradient-based optimization methods for RL.

This [blog post](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) from Lilian Weng is a great starting point, and is quite dense with important RL ideas despite its relative conciseness. It also touches on connections to AlphaGo and gameplay, which you might find interesting as well.

The textbook [“Reinforcement Learning: An Introduction”](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton and Barto is generally considered the classic reference text for the area, at least for “non-deep” methods. This was my primary guide when I was first learning about RL, and it gives a more in-depth exploration of many of the topics touched on in Lilian’s blog post.

If you want to jump ahead to some more neural-flavored content, Andrej Karpathy has a nice [blog post](https://karpathy.github.io/2016/05/31/rl/) on deep RL; this [manuscript](https://arxiv.org/pdf/1810.06339) by Yuxi Li and this [textbook](https://arxiv.org/pdf/2201.02135) by Aske Plaat may be useful for further deep dives.

If you like 3Blue1Brown-style animated videos, the series [“Reinforcement Learning By the Book”](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr) is a great alternative option, and conveys a lot of content from Sutton and Barto, along with some deep RL, using engaging visualizations.

## Markov Models

Running a fixed policy in a Markov decision process yields a Markov chain; processes resembling this kind of setup are fairly abundant, and many branches of machine learning involve modeling systems under Markovian assumptions (i.e. lack of path-dependence, given the current state). This [blog post](https://thagomizer.com/blog/2017/11/07/markov-models.html) from Aja Hammerly makes a nice case for thinking about language models via Markov processes, and this [post](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/) from “Essays on Data Science” has examples and code building up towards auto-regressive Hidden Markov Models, which will start to vaguely resemble some of the neural network architectures we’ll look at later on.

This [blog post](https://www.tweag.io/blog/2019-10-25-mcmc-intro1/) from Simeon Carstens gives a nice coverage of Markov chain Monte Carlo methods, which are powerful and widely-used techniques for sampling from implicitly-represented distributions, and are helpful for thinking about probabilistic topics ranging from stochastic gradient descent to diffusion.

Markov models are also at the heart of many Bayesian methods. See this [tutorial](https://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf) from Zoubin Ghahramani for a nice overview, the textbook [“Pattern Recognition and Machine Learning”](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) for Bayesian angles on many machine learning topics (as well as a more-involved HMM presentation), and this [chapter](https://www.deeplearningbook.org/contents/graphical_models.html) of the Goodfellow et al. “Deep Learning” textbook for some connections to deep learning.

# Section II: Neural Sequential Prediction

**Goal:** Survey deep learning methods + applications to sequential and language modeling, up to basic Transformers.

Here, more than any other section of the handbook, I’ll defer mostly wholesale to existing sequences of instructional resources. This material has been covered quite well by many people in a variety of formats, and there’s no need to reinvent the wheel.

There are a couple different routes you can take from the basics of neural networks towards Transformers (the dominant architecture for most frontier LLMs in 2024). Once we cover the basics, I’ll mostly focus on “deep sequence learning” methods like RNNs. Many deep learning books and courses will more heavily emphasize convolutional neural nets (CNNs), which are quite important for image-related applications and historically were one of the first areas where “scaling” was particularly successful, but technically they’re fairly disconnected from Transformers. They’ll make an appearance when we discuss [state-space models](#structured-state-space-models) and are definitely important for vision applications, but you’ll mostly be okay skipping them for now. However, if you’re in a rush and just want to get to the new stuff, you could consider diving right into decoder-only Transformers once you’re comfortable with feed-forward neural nets — this the approach taken by the excellent [“Let’s build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) video from Andrej Karpathy, casting them as an extension of neural n-gram models for next-token prediction. That’s probably your single best bet for speedrunning Transformers in under 2 hours. But if you’ve got a little more time, understanding the history of RNNs, LSTMs, and encoder-decoder Transformers is certainly worthwhile.

This section is mostly composed of signposts to content from the following sources (along with some blog posts):

*   The [“Dive Into Deep Learning” (d2l.ai)](http://d2l.ai) interactive textbook (nice graphics, in-line code, some theory)
*   3Blue1Brown’s [“Neural networks”](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) video series (lots of animations)
*   Andrej Karpathy’s [“Zero to Hero”](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video series (live coding + great intuitions)
*   [“StatQuest with Josh Starmer”](https://www.youtube.com/@statquest) videos
*   The Goodfellow et al. [“Deep Learning”](https://www.deeplearningbook.org/) textbook (theory-focused, no Transformers)

If your focus is on applications, you might find the interactive [“Machine Learning with PyTorch and Scikit-Learn”](https://github.com/rasbt/machine-learning-book/tree/main) book useful, but I’m not as familiar with it personally.

For these topics, you can also probably get away with asking conceptual questions to your preferred LLM chat interface. This likely won’t be true for later sections — some of those topics were introduced after the knowledge cutoff dates for many current LLMs, and there’s also just a lot less text on the internet about them, so you end up with more “hallucinations”.

## Statistical Prediction with Neural Networks

I’m not actually sure where I first learned about neural nets — they’re pervasive enough in technical discussions and general online media that I’d assume you’ve picked up a good bit through osmosis even if you haven’t studied them formally. Nonetheless, there are many worthwhile explainers out there, and I’ll highlight some of my favorites.

*   The first 4 videos in 3Blue1Brown’s [“Neural networks”](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series will take you from basic definitions up through the mechanics of backpropagation.
*   This [blog post](https://karpathy.github.io/neuralnets/) from Andrej Karpathy (back when he was a PhD student) is a solid crash-course, well-accompanied by his [video](https://www.youtube.com/watch?v=VMj-3S1tku0) on building backprop from scratch.
*   This [blog post](https://colah.github.io/posts/2015-08-Backprop/) from Chris Olah has a nice and concise walk-through of the math behind backprop for neural nets.
*   Chapters 3-5 of the [d2l.ai](http://d2l.ai) book are great as a “classic textbook” presentation of deep nets for regression + classification, with code examples and visualizations throughout.

## Recurrent Neural Networks

RNNs are where we start adding “state” to our models (as we process increasingly long sequences), and there are some high-level similarities to hidden Markov models. This blog post from [Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) is a good starting point. [Chapter 9](https://d2l.ai/chapter_recurrent-neural-networks/index.html) of the [d2l.ai](http://d2l.ai) book is great for main ideas and code; check out [Chapter 10](https://www.deeplearningbook.org/contents/rnn.html) of “Deep Learning” if you want more theory.

For videos, [here](https://www.youtube.com/watch?v=AsNTP8Kwu80)’s a nice one from StatQuest.

## LSTMs and GRUs

Long Short-Term Memory (LSTM) networks and Gated Recurrent Unit (GRU) networks build upon RNNs with more specialized mechanisms for state representation (with semantic inspirations like “memory”, “forgetting”, and “resetting”), which have been useful for improving performance in more challenging data domains (like language).

[Chapter 10](https://d2l.ai/chapter_recurrent-modern/index.html) of [d2l.ai](http://d2l.ai) covers both of these quite well (up through 10.3). The [“Understanding LSTM Networks”](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post from Chris Olah is also excellent. This [video](https://www.youtube.com/watch?v=8HyCNIVRbSU) from “The AI Hacker” gives solid high-level coverage of both; StatQuest also has a video on [LSTMs](https://www.youtube.com/watch?v=YCzL96nL7j0), but not GRUs. GRUs are essentially a simplified alternative to LSTMs with the same basic objective, and it’s up to you if you want to cover them specifically.

Neither LSTMs or GRUs are really prerequisites for Transformers, which are “stateless”, but they’re useful for understanding the general challenges neural sequence of neural sequence and contextualizing the Transformer design choices. They’ll also help motivate some of the approaches towards addressing the “quadratic scaling problem” in [Section VII](#s7).

## Embeddings and Topic Modeling

Before digesting Transformers, it’s worth first establishing a couple concepts which will be useful for reasoning about what’s going on under the hood inside large language models. While deep learning has led to a large wave of progress in NLP, it’s definitely a bit harder to reason about than some of the “old school” methods which deal with word frequencies and n-gram overlaps; however, even though these methods don’t always scale to more complex tasks, they’re useful mental models for the kinds of “features” that neural nets might be learning. For example, it’s certainly worth knowing about Latent Dirichlet Allocation for topic modeling ([blog post](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2)) and [tf-idf](https://jaketae.github.io/study/tf-idf/) to get a feel for what numerical similarity or relevance scores can represent for language.

Thinking about words (or tokens) as high-dimensional “meaning” vectors is quite useful, and the Word2Vec embedding method illustrates this quite well — you may have seen the classic “King - Man + Woman = Queen” example referenced before. [“The Illustrated Word2Vec”](https://jalammar.github.io/illustrated-word2vec/) from Jay Alammar is great for building up this intuition, and these [course notes](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf) from Stanford’s CS224n are excellent as well. Here’s also a nice [video](https://www.youtube.com/watch?v=f7o8aDNxf7k) on Word2Vec from ritvikmath, and another fun one [video](https://www.youtube.com/watch?v=gQddtTdmG_8) on neural word embeddings from Computerphile.

Beyond being a useful intuition and an element of larger language models, standalone neural embedding models are also widely used today. Often these are encoder-only Transformers, trained via “contrastive loss” to construct high-quality vector representations of text inputs which are useful for retrieval tasks (like [RAG](#retrieval-augmented-generation)). See this [post+video](https://docs.cohere.com/docs/text-embeddings) from Cohere for a brief overview, and this [blog post](https://lilianweng.github.io/posts/2021-05-31-contrastive/) from Lilian Weng for more of a deep dive.

## Encoders and Decoders

Up until now we’ve been pretty agnostic as to what the inputs to our networks are — numbers, characters, words — as long as it can be converted to a vector representation somehow. Recurrent models can be configured to both input and output either a single object (e.g. a vector) or an entire sequence. This observation enables the sequence-to-sequence encoder-decoder architecture, which rose to prominence for machine translation, and was the original design for the Transformer in the famed [“Attention is All You Need”](https://arxiv.org/abs/1706.03762) paper. Here, the goal is to take an input sequence (e.g. an English sentence), “encode” it into a vector object which captures its “meaning”, and then “decode” that object into another sequence (e.g. a French sentence). [Chapter 10](https://d2l.ai/chapter_recurrent-modern/index.html) in [d2l.ai](http://d2l.ai) (10.6-10.8) covers this setup as well, which sets the stage for the encoder-decoder formulation of Transformers in [Chapter 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) (up through 11.7). For historical purposes you should certainly at least skim the original paper, though you might get a bit more out of the presentation of its contents via [“The Annotated Transformer”](https://nlp.seas.harvard.edu/annotated-transformer/), or perhaps [“The Illustrated Transformer”](https://jalammar.github.io/illustrated-transformer/) if you want more visualizations. These [notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) from Stanford’s CS224n are great as well.

There are videos on [encoder-decoder](https://www.youtube.com/watch?v=L8HKweZIOmg) architectures and [Attention](https://www.youtube.com/watch?v=PSs6nxngL6k) from StatQuest, a full walkthrough of the original Transformer by [The AI Hacker](https://www.youtube.com/watch?v=4Bdc55j80l8).

However, note that these encoder-decoder Transformers differ from most modern LLMs, which are typically “decoder-only” – if you’re pressed for time, you may be okay jumping right to these models and skipping the history lesson.

## Decoder-Only Transformers

There’s a lot of moving pieces inside of Transformers — multi-head attention, skip connections, positional encoding, etc. — and it can be tough to appreciate it all the first time you see it. Building up intuitions for why some of these choices are made helps a lot, and here I’ll recommend to pretty much anyone that you watch a video or two about them (even if you’re normally a textbook learner), largely because there are a few videos which are really excellent:

*   | 3Blue1Brown’s [“But what is a GPT?”](https://www.youtube.com/watch?v=wjZofJX0v4M) and [“Attention in transformers, explained visually”](Attention in transformers, visually explained | Chapter 6, Deep Learning) – beautiful animations + discussions, supposedly a 3rd video is on the way |

*   Andrej Karpathy’s [“Let’s build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) video – live coding and excellent explanations, really helped some things “click” for me

Here’s a [blog post](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) from Cameron Wolfe walking through the decoder-only architecture in a similar style to the Illustrated/Annotated Transformer posts. There’s also a nice section in d2l.ai ([11.9](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html)) covering the relationships between encoder-only, encoder-decoder, and decoder-only Transformers.

# Section III: Foundations for Modern Language Modeling

**Goal:** Survey central topics related to training LLMs, with an emphasis on conceptual primitives.

In this section, we’ll explore a number of concepts which will take us from the decoder-only Transformer architecture towards understanding the implementation choices and tradeoffs behind many of today’s frontier LLMs. If you first want a birds-eye view the of topics in section and some of the following ones, the post [“Understanding Large Language Models”](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka is a nice summary of what the LLM landscape looks like (at least up through mid-2023).

## Tokenization

Character-level tokenization (like in several of the Karpathy videos) tends to be inefficient for large-scale Transformers vs. word-level tokenization, yet naively picking a fixed “dictionary” (e.g. Merriam-Webster) of full words runs the risk of encountering unseen words or misspellings at inference time. Instead, the typical approach is to use subword-level tokenization to “cover” the space of possible inputs, while maintaining the efficiency gains which come from a larger token pool, using algorithms like Byte-Pair Encoding (BPE) to select the appropriate set of tokens. If you’ve ever seen Huffman coding in an introductory algorithms class I think it’s a somewhat useful analogy for BPE here, although the input-output format is notably different, as we don’t know the set of “tokens” in advance. I’d recommend watching Andrej Karpathy’s [video](https://www.youtube.com/watch?v=zduSFxRajkE) on tokenization and checking out this tokenization [guide](https://blog.octanove.org/guide-to-subword-tokenization/) from Masato Hagiwara.

## Positional Encoding

As we saw in the past section, Transformers don’t natively have the same notion of adjacency or position within a context windows (in contrast to RNNs), and position must instead represented with some kind of vector encoding. While this could be done naively with something like one-hot encoding, this is impractical for context-scaling and suboptimal for learnability, as it throws away notions of ordinality. Originally, this was done with sinusoidal positional encodings, which may feel reminiscent of Fourier features if you’re familiar; the most popular implementation of this type of approach nowadays is likely Rotary Positional Encoding, or RoPE, which tends to be more stable and faster to learn during training.

Resources:

*   [blog post](https://harrisonpim.com/blog/understanding-positional-embeddings-in-transformer-models) by Harrison Pim on intution for positional encodings
*   [blog post](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) by Mehreen Saeed on the original Transformer positional encodings
*   [blog post](https://blog.eleuther.ai/rotary-embeddings/) on RoPE from Eleuther AI original Transformer: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
*   [animated video](https://www.youtube.com/watch?v=GQPOtyITy54) from DeepLearning Hero

## Pretraining Recipes

Once you’ve committed to pretraining a LLM of a certain general size on a particular corpus of data (e.g Common Crawl, FineWeb), there are still a number of choices to make before you’re ready to go:

*   Attention mechanisms (multi-head, multi-query, grouped-query)
*   Activations (ReLU, GeLU, SwiGLU)
*   Optimizers, learning rates, and schedulers (AdamW, warmup, cosine decay)
*   Dropout?
*   Hyperparameter choices and search strategies
*   Batching, parallelization strategies, gradient accumulation
*   How long to train for, how often to repeat data
*   …and many other axes of variation

As far as I can tell, there’s not a one-size-fits-all rule book for how to go about this, but I’ll share a handful of worthwhile resources to consider, depending on your interests:

*   While it predates the LLM era, the blog post [“A Recipe for Training Neural Networks”](https://karpathy.github.io/2019/04/25/recipe/) is a great starting point for framing this problem, as many of these questions are relevant throughout deep learning.
*   [“The Novice’s LLM Training Guide”](https://rentry.org/llm-training) by Alpin Dale, discussing hyperparameter choices in practice, as well as the finetuning techniques we’ll see in future sections.
*   [“How to train your own Large Language Models”](https://blog.replit.com/llm-training) from Replit has some nice discussions on data pipelines and evaluations for training.
*   For understanding attention mechanism tradeoffs, see the post [“Navigating the Attention Landscape: MHA, MQA, and GQA Decoded”](https://iamshobhitagarwal.medium.com/navigating-the-attention-landscape-mha-mqa-and-gqa-decoded-288217d0a7d1) by Shobhit Agarwal.
*   For discussion of “popular defaults”, see the post [“The Evolution of the Modern Transformer: From ‘Attention Is All You Need’ to GQA, SwiGLU, and RoPE”](https://deci.ai/blog/evolution-of-modern-transformer-swiglu-rope-gqa-attention-is-all-you-need/) from Deci AI.
*   For details on learning rate scheduling, see [Chapter 12.11](https://d2l.ai/chapter_optimization/lr-scheduler.html) from the d2l.ai book.
*   For discussion of some controversy surrounding reporting of “best practices”, see this [post](https://blog.eleuther.ai/nyt-yi-34b-response/) from Eleuther AI.

## Distributed Training and FSDP

There are a number of additional challenges associated with training models which are too large to fit on individual GPUs (or even multi-GPU machines), typically necessitating the use of distributed training protocols like Fully Sharded Data Parallelism (FSDP), in which models can be co-located across machines during training. It’s probably worth also understanding its precursor Distributed Data Parallelism (DDP), which is covered in the first post linked below.

Resources:

*   official FSDP [blog post](https://engineering.fb.com/2021/07/15/open-source/fsdp/) from Meta (who pioneered the method) https://sumanthrh.com/post/distributed-and-efficient-finetuning/
*   [blog post](https://blog.clika.io/fsdp-1/) on FSDP by Bar Rozenman, featuring many excellent visualizations
*   [report](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness) from Yi Tai on the challenges of pretraining a model in a startup environment
*   [technical blog](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html) from Answer.AI on combining FSDP with parameter-efficient finetuning techniques for use on consumer GPUs

## Scaling Laws

It’s useful to know about scaling laws as a meta-topic which comes up a lot in discussions of LLMs (most prominently in reference to the “Chinchilla” [paper](https://arxiv.org/abs/2203.15556)), more so than any particular empirical finding or technique. In short, the performance which will result from scaling up the model, data, and compute used for training a language model results in fairly reliable predictions for model loss. This then enables calibration of optimal hyperparameter settings without needing to run expensive grid searches.

Resources:

*   [Chinchilla Scaling Laws for Large Language Models](https://medium.com/@raniahossam/chinchilla-scaling-laws-for-large-language-models-llms-40c434e4e1c1) (blog overview by Rania Hossam)
*   [New Scaling Laws for LLMs](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) (discussion on LessWrong)
*   [Chinchilla’s Wild Implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) (post on LessWrong)
*   [Chinchilla Scaling: A Replication Attempt](https://epochai.org/blog/chinchilla-scaling-a-replication-attempt) (potential issues with Chinchilla findings)
*   [Scaling Laws and Emergent Properties](https://cthiriet.com/blog/scaling-laws) (blog post by Clément Thiriet)
*   [“Scaling Language Models”](https://www.youtube.com/watch?v=UFem7xa3Q2Q) (video lecture, Stanford CS224n)

## Mixture-of-Experts

While many of the prominent LLMs (such as Llama3) used today are “dense” models (i.e. without enforced sparsification), Mixture-of-Experts (MoE) architectures are becoming increasingly popular for navigating tradeoffs between “knowledge” and efficiency, used perhaps most notably in the open-weights world by Mistral AI’s “Mixtral” models (8x7B and 8x22B), and [rumored](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) to be used for GPT-4\. In MoE models, only a fraction of the parameters are “active” for each step of inference, with trained router modules for selecting the parallel “experts” to use at each layer. This allows models to grow in size (and perhaps “knowlege” or “intelligence”) while remaining efficient for training or inference compared to a comparably-sized dense model.

See this [blog post](https://huggingface.co/blog/moe) from Hugging Face for a technical overview, and this [video](https://www.youtube.com/watch?v=0U_65fLoTq0) from Trelis Research for a visualized explainer.

# Section IV: Finetuning Methods for LLMs

**Goal:** Survey techniques used for improving and "aligning" the quality of LLM outputs after pretraining.

In pre-training, the goal is basically “predict the next token on random internet text”. While the resulting “base” models are still useful in some contexts, their outputs are often chaotic or “unaligned”, and they may not respect the format of a back-and-forth conversation. Here we’ll look at a set of techniques for going from these base models to ones resembling the friendly chatbots and assistants we’re more familiar with. A great companion resource, especially for this section, is Maxime Labonne’s interactive [LLM course](https://github.com/mlabonne/llm-course?tab=readme-ov-file) on Github.

## Instruct Fine-Tuning

Instruct fine-tuning (or “instruction tuning”, or “supervised finetuning”, or “chat tuning” – the boundaries here are a bit fuzzy) is the primary technique used (at least initially) for coaxing LLMs to conform to a particular style or format. Here, data is presented as a sequence of (input, output) pairs where the input is a user question to answer, and the model’s goal is to predict the output – typically this also involves adding special “start”/”stop”/”role” tokens and other masking techniques, enabling the model to “understand” the difference between the user’s input and its own outputs. This technique is also widely used for task-specific finetuning on datasets with a particular kind of problem structure (e.g. translation, math, general question-answering).

See this [blog post](https://newsletter.ruder.io/p/instruction-tuning-vol-1) from Sebastian Ruder or this [video](https://www.youtube.com/watch?v=YoVek79LFe0) from Shayne Longpre for short overviews.

## Low-Rank Adapters (LoRA)

While pre-training (and “full finetuning”) requires applying gradient updates to all parameters of a model, this is typically impractical on consumer GPUs or home setups; fortunately, it’s often possible to significantly reduce the compute requirements by using parameter-efficient finetuning (PEFT) techniques like Low-Rank Adapters (LoRA). This can enable competitive performance even with relatively small datasets, particularly for application-specific use cases. The main idea behind LoRA is to train each weight matrix in a low-rank space by “freezing” the base matrix and training a factored representation with much smaller inner dimension, which is then added to the base matrix.

Resources:

*   LoRA paper walkthrough [(video, part 1)](https://youtu.be/dA-NhCtrrVE?si=TpJkPfYxngQQ0iGj)
*   LoRA code demo [(video, part 2)](https://youtu.be/iYr1xZn26R8?si=aG0F8ws9XslpZ4ur)
*   [“Parameter-Efficient LLM Finetuning With Low-Rank Adaptation”](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) by Sebastian Raschka
*   [“Practical Tips for Finetuning LLMs Using LoRA”](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) by Sebastian Raschka

Additionally, an “decomposed” LoRA variant called DoRA has been gaining popularity in recent months, often yielding performance improvements; see this [post](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) from Sebastian Raschka for more details.

## Reward Models and RLHF

One of the most prominent techniques for “aligning” a language model is Reinforcement Learning from Human Feedback (RLHF); here, we typically assume that an LLM has already been instruction-tuned to respect a chat style, and that we additionally have a “reward model” which has been trained on human preferences. Given pairs of differing outputs to an input, where a preferred output has been chosen by a human, the learning objective of the reward model is to predict the preferred output, which involves implicitly learning preference “scores”. This allows bootstrapping a general representation of human preferences (at least with respect to the dataset of output pairs), which can be used as a “reward simulator” for continual training of a LLM using RL policy gradient techniques like PPO.

For overviews, see the posts [“Illustrating Reinforcement Learning from Human Feedback (RLHF)”](https://huggingface.co/blog/rlhf) from Hugging Face and [“Reinforcement Learning from Human Feedback”](https://huyenchip.com/2023/05/02/rlhf.html) from Chip Huyen, and/or this [RLHF talk](https://www.youtube.com/watch?v=2MBJOuVq380) by Nathan Lambert. Further, this [post](https://sebastianraschka.com/blog/2024/research-papers-in-march-2024.html) from Sebastian Raschka dives into RewardBench, and how reward models themselves can be evaluated against each other by leveraging ideas from Direct Preference Optimization, another prominent approach for aligning LLMs with human preference data.

## Direct Preference Optimization Methods

The space of alignment algorithms seems to be following a similar trajectory as we saw with stochastic optimization algorithms a decade ago. In this an analogy, RLHF is like SGD — it works, it’s the original, and it’s also become kind of a generic “catch-all” term for the class of algorithms that have followed it. Perhaps DPO is AdaGrad, and in the year since its release there’s been a rapid wave of further algorithmic developments along the same lines (KTO, IPO, ORPO, etc.), whose relative merits are still under active debate. Maybe a year from now, everyone will have settled on a standard approach which will become the “Adam” of alignment.

For an overview of the theory behind DPO see this [blog post](https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841) Matthew Gunton; this [blog post](https://huggingface.co/blog/dpo-trl) from Hugging Face features some code and demonstrates how to make use of DPO in practice. Another [blog post](https://huggingface.co/blog/pref-tuning) from Hugging Face also discusses tradeoffs between a number of the DPO-flavored methods which have emerged in recent months.

## Context Scaling

Beyond task specification or alignment, another common goal of finetuning is to increase the effective context length of a model, either via additional training, adjusting parameters for positional encodings, or both. Even if adding more tokens to a model’s context can “type-check”, training on additional longer examples is generally necessary if the model may not have seen such long sequences during pretraining.

Resources:

*   [“Scaling Rotational Embeddings for Long-Context Language Models”](https://gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models) by Gradient AI
*   [“Extending the RoPE”](https://blog.eleuther.ai/yarn/) by Eleuther AI, introducing the YaRN method for increased context via attention temperature scaling
*   [“Everything About Long Context Fine-tuning”](https://huggingface.co/blog/wenbopan/long-context-fine-tuning) by Wenbo Pan

## Distillation and Merging

Here we’ll look at two very different methods of consolidating knowledge across LLMs — distillation and merging. Distillation was first popularized for BERT models, where the goal is to “distill” the knowledge and performance of a larger model into a smaller one (at least for some tasks) by having it serve as a “teacher” during the smaller model’s training, bypassing the need for large quantities of human-labeled data.

Some resources on distillation:

*   [“Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT”](https://medium.com/huggingface/distilbert-8cf3380435b5) from Hugging Face
*   [“LLM distillation demystified: a complete guide”](https://snorkel.ai/llm-distillation-demystified-a-complete-guide/) from Snorkel AI
*   [“Distilling Step by Step” blog](https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html) from Google Research

Merging, on the other hand, is much more of a “wild west” technique, largely used by open-source engineers who want to combine the strengths of multiple finetuning efforts. It’s kind of wild to me that it works at all, and perhaps grants some credence to “linear representation hypotheses” (which will appear in the next section when we discuss interpretability). The idea is basically to take two different finetunes of the same base model and just average their weights. No training required. Technically, it’s usually “spherical interpolation” (or “slerp”), but this is pretty much just fancy averaging with a normalization step. For more details, see the post [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models) by Maxime Labonne.

# Section V: LLM Evaluations and Applications

**Goal:** Survey how LLMs are used and evaluated in practice, beyond just "chatbots".

Here we’ll be looking at a handful of topics related to improving or modifying the performance of language models without additional training, as well as techniques for measuring and understanding their performance.

Before diving into the individual chapters, I’d recommend these two high-level overviews, which touch on many of the topics we’ll examine here:

*   [“Building LLM applications for production”](https://huyenchip.com/2023/04/11/llm-engineering.html) by Chip Huyen
*   [“What We Learned from a Year of Building with LLMs” Part 1](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) and [Part 2](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/) from O’Reilly (several authors)

These web courses also have a lot of relevant interactive materials:

*   [“Large Language Model Course”](https://github.com/mlabonne/llm-course) from Maxime Labonne
*   [“Generative AI for Beginners”](https://microsoft.github.io/generative-ai-for-beginners/) from Microsoft

## Benchmarking

Beyond the standard numerical performance measures used during LLM training like cross-entropy loss and [perplexity](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72), the true performance of frontier LLMs is more commonly judged according to a range of benchmarks, or “evals”. Common types of these are:

*   Human-evaluated outputs (e.g. [LMSYS Chatbot Arena](https://chat.lmsys.org/))
*   AI-evaluated outputs (as used in [RLAIF](https://argilla.io/blog/mantisnlp-rlhf-part-4/))
*   Challenge question sets (e.g. those in HuggingFace’s [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard))

See the [slides](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf) from Stanford’s CS224n for an overview. This [blog post](https://www.jasonwei.net/blog/evals) by Jason Wei and [this one](https://humanloop.com/blog/evaluating-llm-apps?utm_source=newsletter&utm_medium=sequence&utm_campaign=) by Peter Hayes do a nice job discussing the challenges and tradeoffs associated with designing good evaluations, and highlighting a number of the most prominent ones used today. The documentation for the open source framework [inspect-ai](https://ukgovernmentbeis.github.io/inspect_ai/) also features some useful discussion around designing benchmarks and reliable evaluation pipelines.

## Sampling and Structured Outputs

While typical LLM inference samples tokens one at a time, there are number of parameters controlling the token distribution (temperature, top_p, top_k) which can be modified to control the variety of responses, as well as non-greedy decoding strategies that allow some degree of “lookahead”. This [blog post](https://towardsdatascience.com/decoding-strategies-in-large-language-models-9733a8f70539) by Maxime Labonne does a nice job discussing several of them.

Sometimes we also want our outputs to follow a particular structure, particularly if we are using LLMs as a component of a larger system rather than as just a chat interface. Few-shot prompting works okay, but not all the time, particularly as output schemas become more complicated. For schema types like JSON, Pydantic and Outlines are popular tools for constraining the output structure from LLMs. Some useful resources:

*   [Pydantic Concepts](https://docs.pydantic.dev/latest/concepts/models/)
*   [Outlines for JSON](https://outlines-dev.github.io/outlines/reference/json/)
*   [Outlines review](https://michaelwornow.net/2023/12/29/outlines-demo) by Michael Wornow

## Prompting Techniques

There are many prompting techniques, and many more prompt engineering guides out there, featuring methods for coaxing more desirable outputs from LLMs. Some of the classics:

*   Few-Shot Examples
*   Chain-of-Thought
*   Retrieval-Augmented Generation (RAG)
*   ReAct

This [blog post](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) by Lilian Weng discusses several of the most dominant approaches, [guide](https://www.promptingguide.ai/techniques) has decent coverage and examples for a wider range of the prominent techniques used today. Keyword-searching on Twitter/X or LinkedIn will give you plenty more. We’ll also dig deeper into RAG and agent methods in later chapters.

## Vector Databases and Reranking

RAG systems require the ability to quickly retrieve relevant documents from large corpuses. Relevancy is typically determined by similarity measures for semantic [embedding](#embeddings-and-topic-modeling) vectors of both queries and documents, such as cosine similarity or Euclidean distance. If we have just a handful of documents, this can be computed between a query and each document, but this quickly becomes intractable when the number of documents grows large. This is the problem addressed by vector databases, which allow retrieval of the *approximate* top-K matches (significantly faster than checking all pairs) by maintaining high-dimensional indices over vectors which efficiently encode their geometric structure. These [docs](https://www.pinecone.io/learn/series/faiss/) from Pinecone do a nice job walking through a few different methods for vector storage, like Locality-Sensitive Hashing and Hierarchical Navigable Small Worlds, which can be implemented with the popular FAISS open-source library. This [talk](https://www.youtube.com/watch?v=W-i8bcxkXok) by Alexander Chatzizacharias gives a nice overview as well.

Another related application of vector retrieval is the “reranking” problem, wherein a model can optimize for other metrics beyond query similarity, such as diversity within retrieved results. See these [docs](https://www.pinecone.io/learn/series/rag/rerankers/) from Pinecone for an overview. We’ll see more about how retrieved results are actually used by LLMs in the next chapter.

## Retrieval-Augmented Generation

One of the most buzzed-about uses of LLMs over the past year, retrieval-augmented generation (RAG) is how you can “chat with a PDF” (if larger than a model’s context) and how applications like Perplexity and Arc Search can “ground” their outputs using web sources. This retrieval is generally powered by embedding each document for storage in a vector database + querying with the relevant section of a user’s input.

Some overviews:

*   [“Deconstructing RAG”](https://blog.langchain.dev/deconstructing-rag/) from Langchain
*   [“Building RAG with Open-Source and Custom AI Models”](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models) from Chaoyu Yang

The [Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction) video course from DeepLearning.AI may also be useful for exploring variants on the standard setup.

## Tool Use and "Agents"

The other big application buzzwords you’ve most likely encountered in some form are “tool use” and “agents”, or “agentic programming”. This typically starts with the ReAct framework we saw in the prompting section, then gets extended to elicit increasingly complex behaviors like software engineering (see the much-buzzed-about “Devin” system from Cognition, and several related open-source efforts like Devon/OpenDevin/SWE-Agent). There are many programming frameworks for building agent systems on top of LLMs, with Langchain and LlamaIndex being two of the most popular. There also seems to be some value in having LLMs rewrite their own prompts + evaluate their own partial outputs; this observation is at the heart of the DSPy framework (for “compiling” a program’s prompts, against a reference set of instructions or desired outputs) which has recently been seeing a lot of attention.

Resources:

*   [“LLM Powered Autonomous Agents” (post)](https://lilianweng.github.io/posts/2023-06-23-agent/) from Lilian Weng
*   [“A Guide to LLM Abstractions” (post)](https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/) from Two Sigma
*   [“DSPy Explained! (video)”](https://www.youtube.com/watch?v=41EfOY0Ldkc) by Connor Shorten

Also relevant are more narrowly-tailored (but perhaps more practical) applications related to databases — see these two [blog](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/) [posts](https://neo4j.com/blog/unifying-llm-knowledge-graph/) from Neo4J for discussion on applying LLMs to analyzing or constructing knowledge graphs, or this [blog post](https://numbersstation.ai/data-wrangling-with-fms-2/) from Numbers Station about applying LLMs to data wrangling tasks like entity matching.

## LLMs for Synthetic Data

An increasing number of applications are making use of LLM-generated data for training or evaluations, including distillation, dataset augmentation, AI-assisted evaluation and labeling, self-critique, and more. This [post](https://www.promptingguide.ai/applications/synthetic_rag) demonstrates how to construct such a synthetic dataset (in a RAG context), and this [post](https://argilla.io/blog/mantisnlp-rlhf-part-4/) from Argilla gives an overview of RLAIF, which is often a popular alternative to RLHF, given the challenges associated with gathering pairwise human preference data. AI-assisted feedback is also a central component of the “Constitutional AI” alignment method pioneered by Anthropic (see their [blog](https://www.anthropic.com/news/claudes-constitution) for an overview).

## Representation Engineering

Representation Engineering is a new and promising technique for fine-grained steering of language model outputs via “control vectors”. Somewhat similar to LoRA adapters, it has the effect of adding low-rank biases to the weights of a network which can elicit particular response styles (e.g. “humorous”, “verbose”, “creative”, “honest”), yet is much more computationally efficient and can be implemented without any training required. Instead, the method simply looks at differences in activations for pairs of inputs which vary along the axis of interest (e.g. honesty), which can be generated synthetically, and then performs dimensionality reduction.

See this short [blog post](https://www.safe.ai/blog/representation-engineering-a-new-way-of-understanding-models) from Center for AI Safety (who pioneered the method) for a brief overview, and this [post](https://vgel.me/posts/representation-engineering/) from Theia Vogel for a technical deep-dive with code examples. Theia also walks through the method in this [podcast episode](https://www.youtube.com/watch?v=PkA4DskA-6M).

## Mechanistic Interpretability

Mechanistic Interpretability (MI) is the dominant paradigm for understanding the inner workings of LLMs by identifying sparse representations of “features” or “circuits” encoded in model weights. Beyond enabling potential modification or explanation of LLM outputs, MI is often viewed as an important step towards potentially “aligning” increasingly powerful systems. Most of the references here will come from [Neel Nanda](https://www.neelnanda.io), a leading researcher in the field who’s created a large number of useful educational resources about MI across a range of formats:

*   [“A Comprehensive Mechanistic Interpretability Explainer & Glossary”](https://www.neelnanda.io/mechanistic-interpretability/glossary)
*   [“An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers”](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers)
*   [“Mechanistic Interpretability Quickstart Guide”](https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide) (Neel Nanda on LessWrong)
*   [“How useful is mechanistic interpretability?”](https://www.lesswrong.com/posts/tEPHGZAb63dfq2v8n/how-useful-is-mechanistic-interpretability) (Neel and others, discussion on LessWrong)
*   [“200 Concrete Problems In Interpretability”](https://docs.google.com/spreadsheets/d/1oOdrQ80jDK-aGn-EVdDt3dg65GhmzrvBWzJ6MUZB8n4/edit#gid=0) (Annotated spreadsheet of open problems from Neel)

Additionally, the articles [“Toy Models of Superposition”](https://transformer-circuits.pub/2022/toy_model/index.html) and [“Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet”](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) from Anthropic are on the longer side, but feature a number of great visualizations and demonstrations of these concepts.

## Linear Representation Hypotheses

An emerging theme from several lines of interpretability research has been the observation that internal representations of features in Transformers are often “linear” in high-dimensional space (a la Word2Vec). On one hand this may appear initially surprising, but it’s also essentially an implicit assumption for techniques like similarity-based retrieval, merging, and the key-value similarity scores used by attention. See this [blog post](https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/) by Beren Millidge, this [talk](https://www.youtube.com/watch?v=ko1xVcyDt8w) from Kiho Park, and perhaps at least skim the paper [“Language Models Represent Space and Time”](https://arxiv.org/pdf/2310.02207) for its figures.

# Section VI: Performance Optimizations for Efficient Inference

**Goal:** Survey architecture choices and lower-level techniques for improving resource utilization (time, compute, memory).

Here we’ll look at a handful of techniques for improving the speed and efficiency of inference from pre-trained Transformer language models, most of which are fairly widely used in practice. It’s worth first reading this short Nvidia [blog post](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) for a crash course in several of the topics we’ll look at (and a number of others).

## Parameter Quantization

With the rapid increase in parameter counts for leading LLMs and difficulties (both in cost and availability) in acquiring GPUs to run models on, there’s been a growing interest in quantizing LLM weights to use fewer bits each, which can often yield comparable output quality with a 50-75% (or more) reduction in required memory. Typically this shouldn’t be done naively; Tim Dettmers, one of the pioneers of several modern quantization methods (LLM.int8(), QLoRA, bitsandbytes) has a great [blog post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) for understanding quantization principles, and the need for mixed-precision quantization as it relates to emergent features in large-model training. Other popular methods and formats are GGUF (for llama.cpp), AWQ, HQQ, and GPTQ; see this [post](https://www.tensorops.ai/post/what-are-quantized-llms) from TensorOps for an overview, and this [post](https://www.maartengrootendorst.com/blog/quantization/) from Maarten Grootendorst for a discussion of their tradeoffs.

In addition to enabling inference on smaller machines, quantization is also popular for parameter-efficient training; in QLoRA, most weights are quantized to 4-bit precision and frozen, while active LoRA adapters are trained in 16-bit precision. See this [talk](https://www.youtube.com/watch?v=fQirE9N5q_Y) from Tim Dettmers, or this [blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) from Hugging Face for overviews. This [blog post](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html) from Answer.AI also shows how to combine QLoRA with FSDP for efficient finetuning of 70B+ parameter models on consumer GPUs.

## Speculative Decoding

The basic idea behind speculative decoding is to speed up inference from a larger model by primarily sampling tokens from a much smaller model and occasionally applying corrections (e.g. every *N* tokens) from the larger model whenever the output distributions diverge. These batched consistency checks tend to be much faster than sampling *N* tokens directly, and so there can be large overall speedups if the token sequences from smaller model only diverge periodically.

See this [blog post](https://jaykmody.com/blog/speculative-sampling/) from Jay Mody for a walkthrough of the original paper, and this PyTorch [article](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for some evaluation results. There’s a nice [video](https://www.youtube.com/watch?v=hm7VEgxhOvk) overview from Trelis Research as well.

## FlashAttention

Computing attention matrices tends to be a primary bottleneck in inference and training for Transformers, and FlashAttention has become one of the most widely-used techniques for speeding it up. In contrast to some of the techniques we’ll see in [Section 7](#s7) which *approximate* attention with a more concise representation (occurring some representation error as a result), FlashAttention is an *exact* representation whose speedup comes from hardware-aware impleemntation. It applies a few tricks — namely, tiling and recomputation — to decompose the expression of attention matrices, enabling significantly reduced memory I/O and faster wall-clock performance (even with slightly increasing the required FLOPS).

Resources:

*   [Talk](https://www.youtube.com/watch?v=gMOAud7hZg4) by Tri Dao (author of FlashAttention)
*   [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) by Aleksa Gordić

## Key-Value Caching and Paged Attention

As noted in the [NVIDIA blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) referenced above, key-value caching is fairly standard in Transformer implementation matrices to avoid redundant recomputation of attention. This enables a tradeoff between speed and resource utilization, as these matrices are kept in GPU VRAM. While managing this is fairly straightforward for a single “thread” of inference, a number of complexities arise when considering parallel inference or multiple users for a single hosted model instance. How can you avoid recomputing values for system prompts and few-shot examples? When should you evict cache elements for a user who may or may not want to continue a chat session? PagedAttention and its popular implementation [vLLM](https://docs.vllm.ai/en/stable/) addresses this by leveraging ideas from classical paging in operating systems, and has become a standard for self-hosted multi-user inference servers.

Resources:

*   [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) (video, Efficient NLP)
*   [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) (video, Anyscale)
*   vLLM [blog post](https://blog.vllm.ai/2023/06/20/vllm.html)

## CPU Offloading

The primary method used for running LLMs either partially or entirely on CPU (vs. GPU) is llama.cpp. See [here](https://www.datacamp.com/tutorial/llama-cpp-tutorial) for a high-level overview; llama.cpp serves as the backend for a number of popular self-hosted LLM tools/frameworks like LMStudio and Ollama. Here’s a [blog post](https://justine.lol/matmul/) with some technical details about CPU performance improvements.

# Section VII: Sub-Quadratic Context Scaling

**Goal:** Survey approaches for avoiding the "quadratic scaling problem" faced by self-attention in Transformers.

A major bottleneck in scaling both the size and context length of Transformers is the quadratic nature of attention, in which all pairs of token interactions are considered. Here we’ll look at a number of approaches for circumventing this, ranging from those which are currently widely used to those which are more exploratory (but promising) research directions.

## Sliding Window Attention

Introduced in the “Longformer” [paper](https://arxiv.org/abs/2004.05150), sliding window attention acts as a sub-quadratic drop-in replacement for standard attention which allows attending only to a sliding window (shocking, right?) of recent tokens/states rather than the entire context window, under the pretense that vectors for these states have already attended to earlier ones and thus have sufficient representational power to encode relevant pieces of early context. Due to its simplicity, it’s become one of the more widely adopted approaches towards sub-quadratic scaling, and is used in Mistral’s popular Mixtral-8x7B model (among others).

Resources:

*   [“What is Sliding Window Attention?”](https://klu.ai/glossary/sliding-window-attention) (blog post by Stephen M. Walker)
*   [“Sliding Window Attention”](https://medium.com/@manojkumal/sliding-window-attention-565f963a1ffd) (blog post by Manoj Kumal)
*   [“Longformer: The Long-Document Transformer”](https://www.youtube.com/watch?v=_8KNb5iqblE) (video by Yannic Kilcher)

## Ring Attention

Another modification to standard attention mechanisms, Ring Attention enables sub-quadratic full-context interaction via incremental computation with a “message-passing” structure, wherein “blocks” of context communicate with each other over a series of steps rather than all at once. Within each block, the technique is essentially classical attention. While largely a research direction rather than standard technique at least within the open-weights world, Google’s Gemini is [rumored](https://www.reddit.com/r/MachineLearning/comments/1arj2j8/d_gemini_1m10m_token_context_window_how/) to possibly be using Ring Attention in order to enable its million-plus-token context.

Resources:

*   [“Breaking the Boundaries: Understanding Context Window Limitations and the idea of Ring Attention”](https://medium.com/@tanuj22july/breaking-the-boundaries-understanding-context-window-limitations-and-the-idea-of-ring-attention-170e522d44b2) (blog post, Tanuj Sharma)
*   [“Understanding Ring Attention: Building Transformers With Near-Infinite Context”](https://www.e2enetworks.com/blog/understanding-ring-attention-building-transformers-with-near-infinite-context) (blog post, E2E Networks)
*   [“Ring Attention Explained”](https://www.youtube.com/watch?v=jTJcP8iyoOM) (video)

## Linear Attention (RWKV)

The Receptance-Weighted Key Value (RWKV) architecture is a return to the general structure of RNN models (e.g LSTMs), with modifications to enable increased scaling and a *linear* attention-style mechanism which supports recurrent “unrolling” of its representation (allowing constant computation per output token as context length scales).

Resources:

*   (Huggingface blog)
*   [“The RWKV language model: An RNN with the advantages of a transformer” - Pt. 1](https://johanwind.github.io/2023/03/23/rwkv_overview.html) (blog post, Johan Wind)
*   [“How the RWKV language model works” - Pt. 2](https://johanwind.github.io/2023/03/23/rwkv_details.html)
*   [“RWKV: Reinventing RNNs for the Transformer Era (Paper Explained)”](https://www.youtube.com/watch?v=x8pW19wKfXQ) (video, Yannic Kilcher)

## Structured State Space Models

Structured State Space Models (SSMs) have become one of the most popular alternatives to Transformers in terms of current research focus, with several notable variants (S4, Hyena, Mamba/S6, Jamba, Mamba-2), but are somewhat notorious for their complexity. The architecture draws inspiration from classical control theory and linear time-invariant systems, with a number of optimizations to translate from continuous to discrete time, and to avoid dense representations of large matrices. They support both recurrent and convolutional representations, which allows efficiency gains both for training and at inference, and many variants require carefully-conditioned “hidden state matrix” representations to support “memorization” of context without needing all-pairs attention. SSMs also seem to be becoming more practical at scale, and have recently resulted in breakthrough speed improvements for high-quality text to speech (via [Cartesia AI](https://www.cartesia.ai/), founded by the inventors of SSMs).

The best explainer out there is likely [“The Annotated S4”](https://srush.github.io/annotated-s4/), focused on the S4 paper from which SSMs originated. The post [“A Visual Guide to Mamba and State Space Models”](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) is great for intuitions and visuals with slightly less math, and Yannic Kilcher has a nice [video](https://www.youtube.com/watch?v=9dSkvxS2EB0) on SSMs as well.

Recently, the Mamba authors released their follow-up “Mamba 2” paper, and their accompanying series of blog posts discusses some newly-uncovered connections between SSM representations and linear attention which may be interesting:

*   [State Space Duality (Mamba-2) Part I - The Model](https://tridao.me/blog/2024/mamba2-part1-model/)
*   [State Space Duality (Mamba-2) Part II - The Theory](https://tridao.me/blog/2024/mamba2-part2-theory/)
*   [State Space Duality (Mamba-2) Part III - The Algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
*   [State Space Duality (Mamba-2) Part IV - The Systems](https://tridao.me/blog/2024/mamba2-part4-systems/)

## HyperAttention

Somewhat similar to RWKV and SSMs, HyperAttention is another proposal for achieving near-linear scaling for attention-like mechanisms, relying on locality-sensitive hashing (think vector DBs) rather than recurrent representations. I don’t see it discussed as much as the others, but it may be worth being aware of nonetheless.

For an overview, see this [blog post](https://medium.com/@yousra.aoudi/linear-time-magic-how-hyperattention-optimizes-large-language-models-b691c0e2c2b0) by Yousra Aoudi and short explainer [video](https://www.youtube.com/watch?v=uvix7XwAjOg) by Tony Shin.

# Section VIII: Generative Modeling Beyond Sequences

**Goal:** Survey topics building towards generation of non-sequential content like images, from GANs to diffusion models.

So far, everything we’ve looked has been focused on text and sequence prediction with language models, but many other “generative AI” techniques require learning distributions with less of a sequential structure (e.g. images). Here we’ll examine a number of non-Transformer architectures for generative modeling, starting from simple mixture models and culminating with diffusion.

## Distribution Modeling

Recalling our first glimpse of language models as simple bigram distributions, the most basic thing you can do in distributional modeling is just count co-occurrence probabilities in your dataset and repeat them as ground truth. This idea can be extended to conditional sampling or classification as “Naive Bayes” ([blog post](https://mitesh1612.github.io/blog/2020/08/30/naive-bayes) [video](https://www.youtube.com/watch?v=O2L2Uv9pdDA)), often one of the simplest algorithms covered in introductory machine learning courses.

The next generative model students are often taught is the Gaussian Mixture Model and its Expectation-Maximization algorithm; Gaussian Mixture Models + Expectation-Maximization algorithm. This [blog post](https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html) and this [video](https://www.youtube.com/watch?v=DODphRRL79c) give decent overviews; the core idea here is assuming that data distributions can be approximated as a mixture of multivariate Gaussian distributions. GMMs can also be used for clustering if individual groups can be assumed to be approximately Gaussian.

While these methods aren’t very effective at representing complex structures like images or language, related ideas will appear as components of some of the more advanced methods we’ll see.

## Variational Auto-Encoders

Auto-encoders and variational auto-encoders are widely used for learning compressed representations of data distributions, and can also be useful for “denoising” inputs, which will come into play when we discuss diffusion. Some nice resources:

*   [“Autoencoders”](https://www.deeplearningbook.org/contents/autoencoders.html) chapter in the “Deep Learning” book
*   [blog post]([https://lilianweng.github.io/posts/2018-08-12-vae/]) from Lilian Weng
*   [video](https://www.youtube.com/watch?v=9zKuYvjFFS8) from Arxiv Insights
*   [blog post](https://towardsdatascience.com/deep-generative-models-25ab2821afd3) from Prakash Pandey on both VAEs and GANs

## Generative Adversarial Nets

The basic idea behind Generative Adversarial Networks (GANs) is to simulate a “game” between two neural nets — the Generator wants to create samples which are indistinguishable from real data by the Discriminator, who wants to identify the generated samples, and both nets are trained continuously until an equilibrium (or desired sample quality) is reached. Following from von Neumann’s minimax theorem for zero-sum games, you basically get a “theorem” promising that GANs succeed at learning distributions, if you assume that gradient descent finds global minimizers and allow both networks to grow arbitrarily large. Granted, neither of these are literally true in practice, but GANs do tend to be quite effective (although they’ve fallen out of favor somewhat in recent years, partly due to the instabilities of simultaneous training).

Resources:

*   [“Complete Guide to Generative Adversarial Networks”](https://blog.paperspace.com/complete-guide-to-gans/) from Paperspace
*   [“Generative Adversarial Networks (GANs): End-to-End Introduction”](https://www.analyticsvidhya.com/blog/2021/10/an-end-to-end-introduction-to-generative-adversarial-networksgans/) by
*   [Deep Learning, Ch. 20 - Generative Models](https://www.deeplearningbook.org/contents/generative_models.html) (theory-focused)

## Conditional GANs

Conditional GANs are where we’ll start going from vanilla “distribution learning” to something which more closely resembles interactive generative tools like DALL-E and Midjourney, incorporating text-image multimodality. A key idea is to learn “representations” (in the sense of text embeddings or autoencoders) which are more abstract and can be applied to either text or image inputs. For example, you could imagine training a vanilla GAN on (image, caption) pairs by embedding the text and concatenating it with an image, which could then learn this joint distribution over images and captions. Note that this implicitly involves learning conditional distributions if part of the input (image or caption) is fixed, and this can be extended to enable automatic captioning (given an image) or image generation (given a caption). There a number of variants on this setup with differing bells and whistles. The VQGAN+CLIP architecture is worth knowing about, as it was a major popular source of early “AI art” generated from input text.

Resources:

*   [“Implementing Conditional Generative Adversarial Networks”](https://blog.paperspace.com/conditional-generative-adversarial-networks/) blog from Paperspace
*   [“Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs”](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8) by Saul Dobilas
*   [“The Illustrated VQGAN”](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) by LJ Miranda
*   [“Using Deep Learning to Generate Artwork with VQGAN-CLIP”](https://www.youtube.com/watch?v=Ih4qOakCZD4) talk from Paperspace

## Normalizing Flows

The aim of normalizing flows is to learn a series of invertible transformations between Gaussian noise and an output distribution, avoiding the need for “simultaneous training” in GANs, and have been popular for generative modeling in a number of domains. Here I’ll just recommend [“Flow-based Deep Generative Models”](https://lilianweng.github.io/posts/2018-10-13-flow-models/) from Lilian Weng as an overview — I haven’t personally gone very deep on normalizing flows, but they come up enough that they’re probably worth being aware of.

## Diffusion Models

One of the central ideas behind diffusion models (like StableDiffusion) is iterative guided application of denoising operations, refining random noise into something that increasingly resembles an image. Diffusion originates from the worlds of stochastic differential equations and statistical physics — relating to the “Schrodinger bridge” problem and optimal transport for probability distributions — and a fair amount of math is basically unavoidable if you want to understand the whole picture. For a relatively soft introduction, see [“A friendly Introduction to Denoising Diffusion Probabilistic Models”](https://medium.com/@gitau_am/a-friendly-introduction-to-denoising-diffusion-probabilistic-models-cc76b8abef25) by Antony Gitau. If you’re up for some more math, check out [“What are Diffusion Models?”](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for more of a deep dive. If you’re more interested in code and pictures (but still some math), see [“The Annotated Diffusion Model”](https://huggingface.co/blog/annotated-diffusion) from Hugging Face, as well as this Hugging Face [blog post](https://huggingface.co/blog/lora) on LoRA finetuning for diffusion models.

# Section IX: Multimodal Models

**Goal:** Survey how models can use multiple modalities of input and output (text, audio, images) simultaneously.

Note: This is the set of topics with which I’m the least familiar, but wanted to include for completeness. I’ll be lighter on commentary and recommendations here, and will return to add more when I think I have a tighter story to tell. The post [“Multimodality and Large Multimodal Models (LMMs)”](https://huyenchip.com/2023/10/10/multimodal.html) by Chip Huyen is a nice broad overview (or [“How Multimodal LLMs Work”](https://www.determined.ai/blog/multimodal-llms) by Kevin Musgrave for a more concise one).

## Tokenization Beyond Text

The idea of tokenization isn’t only relevant to text; audio, images, and video can also be “tokenized” for use in Transformer-style archictectures, and there a range of tradeoffs to consider between tokenization and other methods like convolution. The next two sections will look more into visual inputs; this [blog post](https://www.assemblyai.com/blog/recent-developments-in-generative-ai-for-audio/) from AssemblyAI touches on a number of relevant topics for audio tokenization and representation in sequence models, for applications like audio generation, text-to-speech, and speech-to-text.

## VQ-VAE

The VQ-VAE architecture has become quite popular for image generation in recent years, and underlies at least the earlier versions of DALL-E.

Resources:

*   [“Understanding VQ-VAE (DALL-E Explained Pt. 1)”](https://mlberkeley.substack.com/p/vq-vae) from the Machine Learning @ Berkeley blog
*   [“How is it so good ? (DALL-E Explained Pt. 2)”](https://mlberkeley.substack.com/p/dalle2)
*   [“Understanding Vector Quantized Variational Autoencoders (VQ-VAE)”](https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a) by Shashank Yadav

## Vision Transformers

Vision Transformers extend the Transformer architecture to domains like image and video, and have become popular for applications like self-driving cars as well as for multimodal LLMs. There’s a nice [section](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) in the d2l.ai book about how they work.

[“Generalized Visual Language Models”](https://lilianweng.github.io/posts/2022-06-09-vlm/) by Lilian Weng discusses a range of different approaches and for training multimodal Transformer-style models.

The post [“Guide to Vision Language Models”](https://encord.com/blog/vision-language-models-guide/) from Encord’s blog overviews several architectures for mixing text and vision.

If you’re interested in practical large-scale training advice for Vision Transformers, the MM1 [paper](https://arxiv.org/abs/2403.09611) from Apple examines several architecture and data tradeoffs with experimental evidence.

[“Multimodal Neurons in Artificial Neural Networks”](https://distill.pub/2021/multimodal-neurons/) from Distill.pub has some very fun visualizations of concept representations in multimodal networks.

##### Citation

If you’re making reference to any individual piece of content featured here, please just cite that directly. However, if you wish to cite this as a broad survey, you can use the BibTeX citation below.

```
@misc{Brown24GAIHB,
  author = {Brown, William},
  title = {Generative AI Handbook: A Roadmap for Learning Resources},
  year = 2024,
  url = {https://genai-handbook.github.io}
} 
```