# 1.6 构建和评估 LLM 应用程序

> [`boramorka.github.io/LLM-Book/en/CHAPTER-1/1.6%20Building%20and%20Evaluating%20LLM%20Applications/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/1.6%20Building%20and%20Evaluating%20LLM%20Applications/)

建立由大型语言模型（LLMs）驱动的应用程序需要不仅仅是干净的集成——它需要一个系统的质量评估，涵盖客观和主观方面。在实践中，你将准确性、召回率和 F1（当有黄金答案时）与用户评分和满意度指标（CSA）相结合，同时跟踪运营指标，如成本和延迟。这种组合揭示了弱点，为发布决策提供了信息，并指导了有针对性的改进。

典型的生产路径从简单的提示和少量数据集开始，以快速迭代；然后扩大覆盖范围，复杂化场景，细化指标和质量标准——记住，完美并不总是必要的。通常，在质量和预算约束内持续解决目标任务就足够了。在高风险场景（如医学、执法、金融）中，更严格的验证变得至关重要：随机抽样和保留测试、偏差和错误检查，以及关注伦理和法律问题——防止伤害，确保可解释性，并启用审计。

良好的工程风格强调模块化和快速迭代，自动化的回归测试和测量，与业务目标相一致的有意度量选择，以及定期的偏差/公平性分析。

为了使评估可重复，使用评分标准和评估协议：提前定义标准——与用户意图和上下文的相关性、事实准确性、完整性以及连贯性/流畅性——以及过程、量表和阈值。对于主观任务，使用多个独立的评分者和自动一致性检查。在可能的情况下，将答案与理想（专家）答案进行比较——一个“黄金标准”为更客观的判断提供了一个基准。以下是一个用于可重复实验和评估的小型环境框架和调用函数：

```py
import  os
from  openai  import OpenAI
from  dotenv  import load_dotenv

load_dotenv()
client = OpenAI()

def  fetch_llm_response(prompts, model="gpt-4o-mini", temperature=0, max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content 
```

接下来，正式化基于评分标准的评估，并分配权重以计算一个综合评分，并提供详细的反馈。以下是一个模板，其中模型根据给定的标准进行评估；解析是一个占位符，应替换为适合您模型输出格式的逻辑：

```py
def  evaluate_response_against_detailed_rubric(test_data, llm_response):
  """
 Evaluate the answer on accuracy, relevance, completeness, and coherence.
 Return an overall score and detailed feedback.
 """
    rubric_criteria = {
        'accuracy': {'weight': 3, 'score': None, 'feedback': ''},
        'relevance': {'weight': 2, 'score': None, 'feedback': ''},
        'completeness': {'weight': 3, 'score': None, 'feedback': ''},
        'coherence': {'weight': 2, 'score': None, 'feedback': ''}
    }
    total_weight = sum(c['weight'] for c in rubric_criteria.values())

    system_prompt = "Assess the support agent’s answer given the provided context."
    evaluation_prompt = f"""\
 [Question]: {test_data['customer_query']}
 [Context]: {test_data['context']}
 [Expected answers]: {test_data.get('expected_answers',  'N/A')}
 [LLM answer]: {llm_response}

 Evaluate the answer on accuracy, relevance, completeness, and coherence.
 Provide scores (0–10) for each criterion and specific feedback.
 """

    evaluation_results = fetch_llm_response([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": evaluation_prompt},
    ])

    # Parsing stub — replace with real parsing of your model’s output
    for k in rubric_criteria:
        rubric_criteria[k]['score'] = 8
        rubric_criteria[k]['feedback'] = "Good performance on this criterion."

    overall = sum(v['score'] * v['weight'] for v in rubric_criteria.values()) / total_weight
    detailed = {k: {"score": v['score'], "feedback": v['feedback']} for k, v in rubric_criteria.items()}
    return {"overall_score": overall, "detailed_scores": detailed} 
```

当你需要黄金标准比较时，明确地将模型的答案与理想的专家答案进行比较，并评分高优先级标准（事实准确性、一致性、完整性、连贯性）。以下是一个返回综合评分和原始比较文本的框架，用于审计：

```py
def  detailed_evaluation_against_ideal_answer(test_data, llm_response):
    criteria = {
        'factual_accuracy': {'weight': 4, 'score': None, 'feedback': ''},
        'alignment_with_ideal': {'weight': 3, 'score': None, 'feedback': ''},
        'completeness': {'weight': 3, 'score': None, 'feedback': ''},
        'coherence': {'weight': 2, 'score': None, 'feedback': ''}
    }
    total = sum(c['weight'] for c in criteria.values())

    system_prompt = "Compare the LLM answer to the ideal answer, focusing on factual content and alignment."
    comparison_prompt = f"""\
 [Question]: {test_data['customer_query']}
 [Ideal answer]: {test_data['ideal_answer']}
 [LLM answer]: {llm_response}
 """

    evaluation_text = fetch_llm_response([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comparison_prompt},
    ])

    # Parsing stub
    for k in criteria:
        criteria[k]['score'] = 8
        criteria[k]['feedback'] = "Good alignment with the gold answer."

    score = sum(v['score'] * v['weight'] for v in criteria.values()) / total
    return {"overall_score": score, "details": criteria, "raw": evaluation_text} 
```

在这些基础知识之上，增加高级技术：通过嵌入和相似度指标评估语义相似度（而不仅仅是表面重叠），引入独立审稿人进行群体评估，包括对一致性和逻辑的自动化检查，并构建适应您领域和任务类型的自适应评估框架。在生产中，持续评估至关重要：跟踪版本和指标历史；从用户反馈回到开发的闭环；包括多样化的案例、边缘案例和文化/语言差异；涉及专家（包括盲审以减少偏见）；与替代模型进行比较；并雇佣专门的“裁判”来检测矛盾和事实错误。严格的方法和持续的迭代——加上评分标准、黄金标准、专家评审和自动化检查——有助于您构建可靠和道德的系统。

## 理论问题

1.  为什么需要评估 LLM 答案，以及从哪些维度进行评估？

1.  提供指标示例并解释它们在开发中的作用。

1.  从开发到生产的迭代路径是什么样的？

1.  为什么高风险场景需要更严格的严谨性？请举例说明。

1.  列出启动、迭代和自动化测试的最佳实践。

1.  自动化测试如何帮助开发？

1.  为什么指标应该针对特定任务进行调整？

1.  如何构建评分标准和评估协议？

1.  哪些高级评估技术适用，以及为什么？

1.  持续评估和广泛的测试覆盖率如何提高可靠性？

## 实践任务

1.  编写一个函数，从环境中读取 API 密钥，查询 LLM，并测量运行时间和使用的令牌。
