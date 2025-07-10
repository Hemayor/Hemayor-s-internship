import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载模型和数据
print("正在加载模型和向量库...")
model = SentenceTransformer('hf_models/text2vec-base-chinese')
index = faiss.read_index("faiss.index")

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("law_output.json", "r", encoding="utf-8") as f:
    laws = json.load(f)

# 构建 id → 完整法条 映射
law_dict = {str(item["id"]): item for item in laws}

print(f"加载完成，索引 {len(chunks)} 段，完整法条 {len(law_dict)} 条。\n")


# 2. 查询函数
def search_law(query, top_k=5):
    query_vec = model.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    scores, indices = index.search(query_vec.astype("float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk_item = chunks[idx]
        law_id = str(chunk_item["id"])  # 注意转成字符串匹配 law_dict
        full_law = law_dict.get(law_id, {})

        results.append({
            "id": law_id,
            "category": full_law.get("category", "未知"),
            "title": full_law.get("title", "未知"),
            "content": full_law.get("content", "（未找到完整法条）"),
            "score": float(score)
        })

    return results


# 3. 终端交互
if __name__ == "__main__":
    print("请输入法律相关问题（输入 q 退出）：")
    while True:
        query = input("\n你的问题：")
        if query.strip().lower() == 'q':
            print("再见！")
            break

        results = search_law(query)
        print("\n检索结果（按相似度排序）：")
        for i, item in enumerate(results, 1):
            print(f"\n🔹 结果 {i}")
            print(f"id：{item['id']}")
            print(f"法文名称：{item['category']}")
            print(f"条文标题：{item['title']}")
            print(f"完整条文内容：\n{item['content']}")
            print(f"相似度得分：{item['score']:.4f}（1.0 为最相关）")
