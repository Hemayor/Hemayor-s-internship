import re
import json
from collections import Counter


def clean_content(text):
    """
    清洗内容：保留换行符，去除回车符、全角空格、行内多余空格
    """
    text = text.replace("\r", "")  # 去除回车符
    text = re.sub(r'^[ \t\u3000]+', '', text, flags=re.MULTILINE)  # 每行开头的空格或全角空格
    text = re.sub(r'[ \t\u3000]+$', '', text, flags=re.MULTILINE)  # 每行结尾的空格或全角空格
    text = re.sub(r'[ \t\u3000]{2,}', ' ', text)  # 行内多个空格合并为1个
    return text.strip()


def extract_articles(text):
    """
    提取单部法律中的条文列表 [(title, content)]
    只识别顶格出现的“第X条”作为新法条起点
    """
    articles = []
    # 加 MULTILINE 模式：^ 表示行首，确保是段落的开头
    pattern = re.compile(r'^第[一二三四五六七八九十百千万零]+条', re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        start = match.start()
        title = match.group()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start + len(title):end].strip()
        content = clean_content(content)  # 调用清洗函数
        articles.append((title, content))

    return articles


def split_laws_by_title(raw_text):
    """
    将整个输入拆分为多个法文，返回 [(法文名, 法文内容文本)]
    """
    blocks = raw_text.split("=" * 30)
    laws = []

    for i in range(1, len(blocks) - 1, 2):
        title = blocks[i].strip()
        content = blocks[i + 1].strip()
        if title and content:
            laws.append((title, content))

    return laws

def parse_all_laws(raw_text):
    laws = split_laws_by_title(raw_text)
    all_results = []
    global_id = 1  # 全局 ID 计数器

    for law_title, law_text in laws:
        articles = extract_articles(law_text)
        for title, content in articles:
            all_results.append({
                "id": global_id,
                "category": law_title,
                "title": title,
                "content": content
            })
            global_id += 1  # 每次递增
    return all_results


# 主函数
if __name__ == "__main__":
    with open("legal_clauses.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    parsed = parse_all_laws(raw_text)

    with open("law_output.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print("多部法文处理完成，已输出至 law_output.json")
