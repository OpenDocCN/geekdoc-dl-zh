# 答案 1.6

> [`boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.6/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.6/)

## 理论

1.  评估 LLM 答案是理解有效性、与目标的一致性以及改进领域所必需的。评估准确性、相关性和完整性。

1.  关键指标：准确性、召回率、F1 和用户满意度评分。这些指导产品开发和发布决策。

1.  生产路径是迭代的：从快速原型开始，找出差距，逐渐增加复杂性和数据集覆盖范围。实用性比完美更重要。

1.  高风险场景（医学、法律、金融）需要更严格的验证、偏差检测/缓解和伦理审查。

1.  最佳实践：从小处着手，快速迭代，自动化测试和质量检查。

1.  自动化测试加速了黄金标准比较，揭示了错误，并提供了持续反馈。

1.  选择与应用目标风险相匹配的指标和严谨性；对于高风险情况使用更高的严谨性。

1.  一个完整的评估框架包括评分标准、协议（谁/什么/如何），以及必要时进行黄金标准比较。

1.  高级技术：语义相似性（嵌入）、群体评估、自动连贯性/逻辑检查，以及针对特定领域的自适应方案。

1.  持续评估和多样化的测试案例增加了在不同场景下的可靠性和相关性。

## 实践（草图）

1.  基于评分标准的评估函数：

    ```py
    def  evaluate_response(response: str, rubric: dict) -> dict:
        results = {}
        total_weight = sum(rubric[c]['weight'] for c in rubric)
        total_score = 0
        for criteria, details in rubric.items():
            score = details.get('weight', 1)  # stub — replace with real logic
            feedback = f"Stub feedback for {criteria}."
            results[criteria] = {'score': score, 'feedback': feedback}
            total_score += score * details['weight']
        results['overall'] = {
            'weighted_average_score': total_score / total_weight,
            'feedback': 'Overall feedback based on the rubric.'
        }
        return results 
    ```

1.  评分标准模板：

    ```py
    rubric = {
        'accuracy': {'weight': 3},
        'relevance': {'weight': 2},
        'completeness': {'weight': 3},
        'coherence': {'weight': 2},
    } 
    ```

1.  理想（黄金）答案作为加权评分和文本反馈的比较基准。
