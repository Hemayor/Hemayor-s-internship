"""
基于Langchain的Agentic问答流程：检索Agent、问答Agent、总结Agent
"""
import requests
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import LLM
# from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 1. 加载模型和数据
print("🔍 正在加载模型和向量库...")
model_path = os.path.join(project_root, "pre-process/hf_models", "text2vec-base-chinese")
model = SentenceTransformer(model_path)
index_path = os.path.join(current_dir, "faiss.index")
index = faiss.read_index(index_path)

chunks_path = os.path.join(current_dir, "chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

laws_path = os.path.join(current_dir, "law_output.json")
with open(laws_path, "r", encoding="utf-8") as f:
    laws = json.load(f)

# 构建 id → 完整法条 映射
law_dict = {str(item["id"]): item for item in laws}

print(f"✅ 加载完成，索引 {len(chunks)} 段，完整法条 {len(law_dict)} 条。\n")

# 2. 自定义Ollama LLM类
class OllamaLLM(LLM):
    """自定义Ollama LLM类"""

    model_name: str = Field(default="llama3")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    num_predict: int = Field(default=512)
    repeat_penalty: float = Field(default=1.1)
    top_k: int = Field(default=40)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用Ollama API"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.num_predict,
                    "repeat_penalty": self.repeat_penalty,
                    "top_k": self.top_k
                }
            )
            return response.json().get("response", "")
        except Exception as e:
            return f"【系统错误】无法连接到本地大模型：{e}\n\n请检查Ollama服务是否正常运行。"

# 3. 自定义检索器
class LawRetriever:
    """自定义法律检索器"""

    def __init__(self, model, index, chunks, law_dict, top_k: int = 5):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.law_dict = law_dict
        self.top_k = top_k

    def preprocess_query(self, query: str) -> str:
        """预处理查询，提取关键信息"""
        # 移除标点符号，保留中文、英文、数字
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query)
        return query.strip()

    def search_law(self, query: str) -> List[Dict]:
        """搜索法律条文"""
        query_vec = self.model.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        scores, indices = self.index.search(query_vec.astype("float32"), self.top_k * 2)

        results = []
        seen_ids = set()

        for score, idx in zip(scores[0], indices[0]):
            chunk_item = self.chunks[idx]
            law_id = str(chunk_item["id"])

            if law_id in seen_ids:
                continue

            full_law = self.law_dict.get(law_id, {})

            if full_law.get("content"):
                results.append({
                    "id": law_id,
                    "category": full_law.get("category", "未知"),
                    "title": full_law.get("title", "未知"),
                    "content": full_law.get("content", "（未找到完整法条）"),
                    "score": float(score)
                })
                seen_ids.add(law_id)

                if len(results) >= self.top_k:
                    break

        return results

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        processed_query = self.preprocess_query(query)
        results = self.search_law(processed_query)

        documents = []
        for result in results:
            # 构建文档内容
            content = f"法条ID: {result['id']}\n分类: {result['category']}\n标题: {result['title']}\n内容: {result['content']}"

            doc = Document(
                page_content=content,
                metadata={
                    "id": result["id"],
                    "category": result["category"],
                    "title": result["title"],
                    "score": result["score"]
                }
            )
            documents.append(doc)

        return documents

# 4. 创建Langchain组件
# 初始化LLM
llm = OllamaLLM()

# 初始化检索器
retriever = LawRetriever(model, index, chunks, law_dict, top_k=8)

# 5. 问答Agent的Prompt模板
qa_prompt_template = """你是一位资深的中国法律专家，具有丰富的法律实务经验。请根据提供的相关法条，为用户提供专业、准确的法律咨询。

【相关法条】
{context}

【历史对话】
{history}

【用户问题】
{question}

【回答要求】
1. 语言：必须使用中文回答，避免使用英文
2. 结构：按照"法律依据→具体分析→实务建议"的结构回答
3. 引用：明确引用相关法条的具体条款
4. 实务：提供具体的操作建议和注意事项
5. 准确：确保法律依据准确，避免误导

【回答格式】
根据《XXX法》第X条规定，...

具体分析：
...

实务建议：
1. ...
2. ...

回答："""

qa_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=qa_prompt_template
)

# 6. 总结Agent的Prompt模板
summary_prompt_template = """你是一位专业的法律顾问，请根据用户的具体问题，对以下法律问答内容进行专业、准确的中文总结。

【用户问题】
{question}

【问答内容】
{content}

【总结要求】
请严格按照以下结构进行总结，必须包含这四个部分，每个要点都要单独一行：

📋 问题总结
• 针对用户问题的确切回答：[根据用户的具体问题，用一句话准确回答用户最关心的核心问题]

📋 核心要点
• 主要法律问题：[具体内容]
• 关键法律依据：[具体内容]
• 核心争议点：[具体内容]

⚖️ 法律条款
• 相关法条名称：[具体内容]
• 具体条款内容：[具体内容]
• 适用条件：[具体内容]

💡 实务建议
• 操作步骤：[具体内容]
• 注意事项：[具体内容]
• 风险提示：[具体内容]
• 维权途径：[具体内容]

要求：
- 必须使用中文
- 严格按照上述四个部分结构
- 每个要点都要单独一行，使用"•"符号
- 每个部分都要有具体内容
- 语言专业准确
- 结构清晰明了
- 重点突出
- 避免使用英文
- 每个要点之间要有换行
- 开头必须根据用户的具体问题给出准确的回答

请按照以下格式输出：
📋 问题总结
• [根据用户的具体问题，用一句话准确回答用户最关心的核心问题]

📋 核心要点
• [具体内容]
• [具体内容]
• [具体内容]

⚖️ 法律条款
• [具体内容]
• [具体内容]
• [具体内容]

💡 实务建议
• [具体内容]
• [具体内容]
• [具体内容]

总结："""

summary_prompt = PromptTemplate(
    input_variables=["question", "content"],
    template=summary_prompt_template
)

# 7. 构建Langchain链
# 问答链
qa_chain = qa_prompt | llm | StrOutputParser()

# 总结链
summary_chain = summary_prompt | llm | StrOutputParser()


# 8. Agent函数
def search_agent(query: str, topk: int = 10) -> List[tuple]:
    """检索Agent - 使用Langchain检索器"""
    # 直接调用检索器的搜索方法
    processed_query = retriever.preprocess_query(query)
    results = retriever.search_law(processed_query)

    retrieved_texts = []
    for result in results:
        # 构建文档内容
        content = f"法条ID: {result['id']}\n分类: {result['category']}\n标题: {result['title']}\n内容: {result['content']}"
        retrieved_texts.append((
            content,
            result["score"],
            result["id"]
        ))

    return retrieved_texts

def qa_agent(query: str, retrieved_texts: List[tuple], history: Optional[List[Dict]] = None) -> str:
    """问答Agent - 使用Langchain链"""
    if not retrieved_texts:
        return "抱歉，未能找到相关的法律条文。请尝试用不同的关键词重新提问。"

    # 构建上下文
    context_parts = []
    for i, (text, score, chunk_id) in enumerate(retrieved_texts, 1):
        context_parts.append(f"[法条{i}] 相关度:{score:.3f}\n{text}")

    context = '\n\n'.join(context_parts)

    # 构建历史对话上下文
    history_text = ""
    if history:
        history_contexts = []
        for h in history[-2:]:  # 只保留最近2轮对话
            history_contexts.append(f"用户：{h['user']}\n助手：{h['assistant']}")
        history_text = "\n\n历史对话：\n" + "\n".join(history_contexts)

    # 调用Langchain链
    result = qa_chain.invoke({
        "context": context,
        "history": history_text,
        "question": query
    })

    return result

def summary_agent(answers: List[str], user_question: Optional[str] = None) -> str:
    """总结Agent - 使用Langchain链"""
    if not answers:
        return "暂无内容可总结"

    # 将所有回答合并
    content = "\n".join(answers)

    # 调用Langchain链
    result = summary_chain.invoke({
        "question": user_question or "未知问题",
        "content": content
    })

    return f"【智能总结】\n{result}"

# 主程序：用户交互逻辑
if __name__ == '__main__':
    history: List[Dict[str, str]] = []
    print("欢迎使用基于Langchain的多轮法律问答系统，输入 exit 退出。")

    while True:
        query = input('请输入你的问题（按回车检索，输入 exit 退出）：')
        if query.strip().lower() == 'exit':
            break

        retrieved = search_agent(query, topk=8)
        print('\n【检索Agent结果】')
        for idx, (text, score, chunk_id) in enumerate(retrieved, 1):
            # 从law_dict获取完整信息
            full_law = law_dict.get(str(chunk_id), {})
            law_name = full_law.get('category', '未知法律')
            law_id = chunk_id
            law_content = full_law.get('content', '（未找到完整法条）')

            print(f'{idx}. 《{law_name}》')
            print(f'   ID: {law_id}')
            print(f'   相似度: {score:.4f}')
            print(f'   完整条文: {law_content}')
            print('-' * 50)

        answer = qa_agent(query, retrieved, history)
        print('\n【问答Agent结果】')
        print(answer)

        summary = summary_agent([answer], user_question=query)
        print('\n【总结Agent结果】')
        print(summary)

        print('-' * 40)
        # 记录历史问答
        history.append({"user": query, "assistant": answer})
