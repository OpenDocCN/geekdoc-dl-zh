# 答案 1.3

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.3/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.3/)

## 理论

1.  集成 OpenAI 审查 API：获取 API 密钥，在后台添加客户端库，并将审查插入内容提交管道，以便在发布前对所有数据进行分析。

1.  定制化：调整敏感性，关注特定违规行为，并使用符合社区标准和合规要求的自己的允许/拒绝列表。

1.  扩展审查：除了文本之外，添加对图像和视频的检查（使用 OpenAI 工具或第三方解决方案）以提供全面保护。

1.  分隔符通过将用户输入与系统指令分离并保留命令完整性来降低提示注入风险。

1.  使用分隔符隔离命令，明确区分可执行指令和用户数据，防止恶意指令的注入。

1.  附加措施：严格的输入验证、最小权限设计、允许列表、正则表达式、监控和日志记录以检测异常。

1.  直接评估：要求模型将输入分类为注入尝试或非注入尝试——这减少了误报并提高了响应准确性。

1.  响应措施：通知和教育用户，要求他们重新措辞，隔离可疑内容供人工审查，并动态调整敏感性。

1.  直接评估的优缺点：准确性和适应性 versus 开发/维护复杂性以及需要在安全性与用户体验之间取得平衡。

1.  将 Moderation API 与反注入策略相结合，显著提高了 UGC 平台的安全性和完整性。

## 练习（草图）

1.  审查单个文本片段：

    ```py
    from  openai  import OpenAI

    client = OpenAI()

    def  moderate_content(content: str) -> bool:
        resp = client.moderations.create(model="omni-moderation-latest", input=content)
        return bool(resp.results[0].flagged) 
    ```

1.  从字符串中移除分隔符：

    ```py
    def  sanitize_delimiter(input_text: str, delimiter: str) -> str:
        return input_text.replace(delimiter, "") 
    ```

1.  检查输入长度：

    ```py
    def  validate_input_length(input_text: str, min_length=1, max_length=200) -> bool:
        return min_length <= len(input_text) <= max_length 
    ```

1.  基于简单启发式的用户会话：

    ```py
    class  UserSession:
        def  __init__(self, user_id: int):
            self.user_id = user_id
            self.trust_level = 0
            self.sensitivity_level = 5

        def  adjust_sensitivity(self):
            if self.trust_level > 5:
                self.sensitivity_level = max(1, self.sensitivity_level - 1)
            else:
                self.sensitivity_level = min(10, self.sensitivity_level + 1)

        def  evaluate_input(self, user_input: str) -> bool:
            dangerous_keywords = ["exec", "delete", "drop"]
            return any(k in user_input.lower() for k in dangerous_keywords)

        def  handle_input(self, user_input: str):
            if self.evaluate_input(user_input):
                if self.trust_level < 5:
                    print("Input flagged and sent for security review.")
                else:
                    print("The request looks suspicious. Please clarify or rephrase.")
            else:
                print("Input accepted. Thank you!")
            print("Remember: input should be clear and free of potentially dangerous commands.") 
    ```

1.  直接评估注入（存根逻辑）：

    ```py
    def  direct_evaluation_for_injection(user_input: str) -> str:
        if "ignore instructions" in user_input.lower() or "disregard previous guidelines" in user_input.lower():
            return 'Y'
        return 'N' 
    ```

1.  主循环中的示例集成：

    ```py
    if __name__ == "__main__":
        session = UserSession(user_id=1)
        while True:
            text = input("Enter text (or 'exit'): ")
            if text.lower() == 'exit':
                break

            text = sanitize_delimiter(text, "####")
            if not validate_input_length(text):
                print("Input too short/long.")
                continue

            if moderate_content(text):
                print("Content flagged as unacceptable. Please revise.")
                continue

            if direct_evaluation_for_injection(text) == 'Y':
                print("Potential injection detected. Please rephrase.")
                continue

            session.handle_input(text) 
    ```
