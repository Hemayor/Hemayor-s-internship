from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import time

def read_links(filename):
    urls = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"(https?://[^\s]+)", line)
            if match:
                urls.append(match.group(1))
    return urls

def clean_title(h2_tag):
    """提取法律标题，不包括嵌套标签"""
    return ''.join(h2_tag.find_all(string=True, recursive=False)).strip()

def remove_redundant_article(article_title, paragraph):
    """去除段落中重复的“第x条”"""
    if paragraph.startswith(article_title):
        return paragraph[len(article_title):].strip()
    return paragraph

def parse_law_page_selenium(driver, url):
    try:
        driver.get(url)
        time.sleep(0.2)  # 等待 JS 渲染完成，可改成 WebDriverWait 更稳
        soup = BeautifulSoup(driver.page_source, "html.parser")
    except Exception as e:
        print(f"页面加载失败：{url}\n原因：{e}")
        return None, []

    # 提取法律标题
    h2 = soup.find("h2", class_="title")
    title = clean_title(h2) if h2 else "未知法文"

    articles = []

    for tiao_wrap in soup.find_all("div", class_="tiao-wrap"):
        article_title_tag = tiao_wrap.find("span", class_="navtiao")
        article_title = article_title_tag.get_text(strip=True) if article_title_tag else "未知条"

        paragraphs = []

        # 遍历所有 kuan-wrap
        for kuan_wrap in tiao_wrap.find_all("div", class_="kuan-wrap"):
            for elem in kuan_wrap.children:
                if getattr(elem, 'name', None) == 'div':
                    class_list = elem.get("class", [])
                    if "kuan-content" in class_list:
                        text = elem.get_text(strip=True)
                        text = remove_redundant_article(article_title, text)
                        if text:
                            paragraphs.append(text)
                    elif "xiang-wrap" in class_list:
                        xiang_elem = elem.find("div", class_="xiang-content")
                        if xiang_elem:
                            text = xiang_elem.get_text(strip=True)
                            if text:
                                paragraphs.append(text)

        articles.append({
            "article": article_title,
            "paragraphs": paragraphs
        })

    return title, articles



def save_all_to_txt(all_data, filename="legal_clauses_部门规章.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for title, articles in all_data:
            f.write("=" * 30 + "\n")
            f.write(title + "\n")
            f.write("=" * 30 + "\n")
            for art in articles:
                f.write(art["article"] + "\n")
                for para in art["paragraphs"]:
                    if para.strip():
                        f.write("　　" + para.strip() + "\n")
                f.write("\n")



def main():
    url_list = read_links("法条详情链接_部门规章（全部）.txt")
    print(f"共读取 {len(url_list)} 个链接，开始爬取...\n")

    # 设置 Selenium 选项
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=chrome_options)

    all_data = []

    for i, url in enumerate(url_list, start=1):
        print(f"正在爬取第 {i} 个链接：{url}")
        title, articles = parse_law_page_selenium(driver, url)
        if title and articles:
            all_data.append((title, articles))
            print(f"完成：{title}，共 {len(articles)} 条法条\n")
        else:
            print("未能提取有效内容。\n")
        time.sleep(1)

    driver.quit()
    save_all_to_txt(all_data)
    print("所有爬取任务已完成，内容已保存至 legal_clauses_部门规章.txt。")

if __name__ == "__main__":
    main()
