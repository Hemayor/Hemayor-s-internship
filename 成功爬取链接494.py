from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import os

# -------------------- 配置 --------------------
BASE_URL = "https://www.pkulaw.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'Referer': BASE_URL
}

# -------------------- 提取链接函数 --------------------
def get_law_detail_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    detail_urls = []
    for a in soup.select('h4 a[href*="/chl/"]'):
        href = a['href']
        clean_url = href.split('?')[0]
        full_url = urljoin(BASE_URL, clean_url)
        detail_urls.append(full_url)
    return detail_urls

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    options = Options()
    # options.add_argument("--headless")  # 如需隐藏浏览器窗口取消注释
    driver = webdriver.Chrome(options=options)

    driver.get(BASE_URL)
    print("👉 请在浏览器中完成登录、筛选、滑块验证等操作。")
    input("✅ 完成后按回车继续...")

    wait = WebDriverWait(driver, 30)
    visited_urls = set()
    all_urls = []
    prev_first_url = ""

    for page in range(1, 6):
        print(f"\n📄 正在处理第 {page} 页...")

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.accompanying-wrap h4 a")))
        except:
            input("⚠️ 页面加载失败，可能需要滑块验证，请手动处理后按回车...")

        page_source = driver.page_source
        detail_urls = get_law_detail_urls(page_source)

        # 判断页面是否重复（首个链接相同）
        if detail_urls and detail_urls[0] == prev_first_url:
            print("⚠️ 当前页与上一页内容重复，可能页面未刷新或遇到滑块验证")
            input("👉 请在浏览器完成验证或手动点击“下一页”，完成后按回车...")

            # 再次尝试获取页面
            page_source = driver.page_source
            detail_urls = get_law_detail_urls(page_source)
            if detail_urls and detail_urls[0] == prev_first_url:
                print("❌ 页面仍然重复，跳过此页。")
                continue

        # 保存当前页的首链接
        if detail_urls:
            prev_first_url = detail_urls[0]

        print(f"🔗 获取到 {len(detail_urls)} 个链接")
        for url in detail_urls:
            if url not in visited_urls:
                visited_urls.add(url)
                all_urls.append(url)

        # 点击“下一页”
        if page < 5:
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "下一页")))
                next_btn.click()
                time.sleep(4)  # 等待加载
            except Exception as e:
                print(f"⚠️ 无法点击下一页：{e}")
                input("👉 请手动点击“下一页”并处理验证码，然后按回车继续...")

    # 保存链接到文件
    with open("详情页链接.txt", "w", encoding="utf-8") as f:
        for i, url in enumerate(all_urls, 1):
            f.write(f"{i}. {url}\n")

    print(f"\n✅ 共获取 {len(all_urls)} 条链接，已保存到 详情页链接.txt")
    driver.quit()
