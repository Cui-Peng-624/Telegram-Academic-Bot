import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

from urllib.parse import urljoin

def fetch_google_scholar_results(inquiry, as_ylo, start, hl="zh-CN"):
    base_url = "https://scholar.google.com/scholar"
    query_params = {
        "q": inquiry,
        "hl": hl,
        "as_sdt": "0,5",
        "as_ylo": as_ylo,
        "start": start
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    results = []

    response = requests.get(base_url, params=query_params, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        divs = soup.find_all("div", class_="gs_r gs_or gs_scl")

        for div in divs:
            result = {}
            h3 = div.find("h3")
            if h3:
                a = h3.find("a")
                if a:
                    title = a.get_text()
                    result["Title"] = title
                    href = a.get('href')
                    result["href"] = href
                    
            gs_a = div.find("div", class_="gs_a")
            if gs_a:
                basic_info = gs_a.get_text()
                result["basic_info"] = basic_info

            gs_rs = div.find("div", class_="gs_rs")
            if gs_rs:
                abstract = gs_rs.get_text()
                result["concise_abstract"] = abstract

            gs_or_nvi = div.find("a", class_="gs_or_nvi")
            if gs_or_nvi:
                snapshot = gs_or_nvi.get('href')
                if snapshot and snapshot != "javascript:void(0)":
                    try:
                        second_response = requests.get(snapshot)
                        second_response.raise_for_status()  # Raise an exception for bad status codes
                    except requests.exceptions.InvalidSchema:
                        # 如果是无效的URL架构，可能是相对路径，将其转换为绝对路径
                        snapshot_url = urljoin("https://scholar.google.com", snapshot)
                        try:
                            second_response = requests.get(snapshot_url)
                            second_response.raise_for_status()
                        except requests.exceptions.RequestException as e:
                            print(f"Error occurred while fetching secondary URL (after conversion): {e}")
                            continue
                    except requests.exceptions.RequestException as e:
                        print(f"Error occurred while fetching secondary URL: {e}")
                        continue
                    
                    if second_response.status_code == 200:
                        second_soup = BeautifulSoup(second_response.content, "html.parser")
                        article_abstract_div = second_soup.find("div", id="articleAbstract")
                        if article_abstract_div is not None: 
                            articleAbstract = article_abstract_div.get_text()
                            result["full_abstract"] = articleAbstract

            results.append(result)
    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 1))
        print(f"Too many requests. Retrying after {retry_after} seconds...")
    else:
        print("Error", response.status_code)
    return results

def check_and_initialize_persist_file(persist_file):
    if not os.path.exists(persist_file):  # If the persistence file does not exist, create an empty DataFrame and save it
        df_persist = pd.DataFrame(columns=['Title', 'href', 'basic_info', 'concise_abstract', 'full_abstract'])
        df_persist.to_csv(persist_file, index=False)
    else:  # If the persistence file exists, read the file content
        df_persist = pd.read_csv(persist_file)
    return df_persist

# 函数，用于将爬取结果保存到临时文件夹
def fetch_and_save_results_temp(temp_save_path, q, as_ylo, start):
    all_results = []
    for i in range(0, 10 * start, 10):
        results = fetch_google_scholar_results(q, as_ylo, i)
        all_results.extend(results)
        time.sleep(5)
    
    if all_results:
        df_new = pd.DataFrame(all_results)
        df_new = df_new.dropna(subset=['Title'])
        
        df_new.to_csv(temp_save_path, index=False)  # 覆写临时文件
        # print(f"成功保存{len(all_results)}条结果到{temp_save_path}")
    else:
        # print("没有新的结果")
        return df_new

# 函数，用于根据用户反馈将数据从临时文件夹保存到永久文件夹
def confirm_and_save_results(temp_save_path, permanent_save_path, user_confirmation):
    if user_confirmation:
        df_temp = check_and_initialize_persist_file(temp_save_path)
        df_permanent = check_and_initialize_persist_file(permanent_save_path)
        
        df_permanent = pd.concat([df_permanent, df_temp], ignore_index=True)
        df_permanent.to_csv(permanent_save_path, index=False)
        print(f"成功将数据从{temp_save_path}保存到{permanent_save_path}")
    else:
        print("用户未确认保存，数据未转移")

# 示例调用
# fetch_and_save_results_temp("temp_scholar_results.csv", "machine+learning", 2020, 10)
# user_confirmation = input("是否将数据保存到永久文件夹？(y/n): ").lower() == 'y'
# confirm_and_save_results("temp_scholar_results.csv", "permanent_scholar_results.csv", user_confirmation)

