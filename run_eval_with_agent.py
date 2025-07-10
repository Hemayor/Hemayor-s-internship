import json
from agents2 import search_agent, qa_agent, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# 初始化模型
llm = OllamaLLM()

# 评价prompt
evaluation_prompt = PromptTemplate.from_template("""
你是一个专业的法律问答评估助手，请根据标准答案与模型输出，对模型的表现进行客观、准确的评估。

【问题】
{question}

【标准答案】
{expected_answer}

【模型回答】
{model_output}

请根据以下方面进行综合评价：
- 是否覆盖了标准答案中的核心要点？
- 是否存在错误或误导性内容？
- 是否结构清晰，有没有冗余重复？
- 是否缺少重要法律依据？

请用一句完整的话进行评价，语言简洁准确，不超过50字。必须用中文回答。

评价结果：
""")

# 构建 Langchain 评估链
evaluator_chain = evaluation_prompt | llm | StrOutputParser()

# 加载测试数据（178条）
with open("test_questions.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 保存结果为数组
results = []

print(f"🚀 正在处理 {len(test_data)} 条测试数据...\n")

for item in tqdm(test_data):
    question = item.get("question", "").strip()
    expected = item.get("expected_answer", "").strip()

    try:
        # 检索
        retrieved = search_agent(question, topk=5)
        retrieved_text = "\n\n".join([x[0] for x in retrieved])

        # 问答
        model_output = qa_agent(question, retrieved)

        # 评估
        evaluation_note = evaluator_chain.invoke({
            "question": question,
            "expected_answer": expected,
            "model_output": model_output
        }).strip()

        # 构造结果
        result = {
            "question": question,
            "expected_answer": expected,
            "retrieved_text": retrieved_text,
            "model_output": model_output.strip(),
            "evaluation_note": evaluation_note
        }

        results.append(result)

    except Exception as e:
        print(f"❌ 错误 - {question}: {e}")

# 保存为格式化 JSON，换行、美观、转义“\n”为实际换行
with open("test_result_pretty.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
