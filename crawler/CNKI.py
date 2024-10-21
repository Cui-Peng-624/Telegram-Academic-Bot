import os
import requests
from lxml import etree
from bs4 import BeautifulSoup
import pandas as pd  # 新增
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}

def get_page_text(url, search_word, page_num):
    data = {
        'searchType': 'MulityTermsSearch', 'ArticleType': '', 'ReSearch': '', 'ParamIsNullOrEmpty': 'false', 'Islegal': 'false',
        'Content': search_word,
        'Theme': '', 'Title': '', 'KeyWd': '', 'Author': '', 'SearchFund': '', 'Originate': '', 'Summary': '', 'PublishTimeBegin': '', 'PublishTimeEnd': '', 
        'MapNumber': '', 'Name': '', 'Issn': '', 'Cn': '', 'Unit': '', 'Public': '', 'Boss': '', 'FirstBoss': '', 'Catalog': '', 'Reference': '', 'Speciality': '', 
        'Type': '', 'Subject': '', 'SpecialityCode': '', 'UnitCode': '', 'Year': '', 'AcefuthorFilter': '', 'BossCode': '', 'Fund': '', 'Level': '', 'Elite': '', 
        'Organization': '', 'Order': '1',
        'Page': str(page_num),
        'PageIndex': '', 'ExcludeField': '', 'ZtCode': '', 'Smarts': '',
    }
    response = requests.post(url=url, headers=headers, data=data)
    page_text = response.text
    return page_text

def list_to_str(my_list):
    my_str = "".join(my_list)
    return my_str

def get_abstract(url):
    response = requests.get(url=url, headers=headers)
    page_text = response.text
    tree = etree.HTML(page_text)
    abstract = tree.xpath('//div[@class="xx_font"]//text()')
    return abstract

def parse_page_text(page_text, top_n, temp_save_path):
    tree = etree.HTML(page_text)
    item_list = tree.xpath('//div[@class="list-item"]')
    page_info = []
    
    for item in item_list[0:top_n]:  # 只解析前top_n篇
        # 标题
        title = list_to_str(item.xpath('./p[@class="tit clearfix"]/a[@class="left"]/@title'))
        # 链接
        link = 'https:' + list_to_str(item.xpath('./p[@class="tit clearfix"]/a[@class="left"]/@href'))
        # 作者
        author = list_to_str(item.xpath('./p[@class="source"]/span[1]/@title'))
        # 出版日期
        date = list_to_str(item.xpath('./p[@class="source"]/span[last()-1]/text() | ./p[@class="source"]/a[2]/span[1]/text()'))
        # 关键词
        keywords = list_to_str(item.xpath('./div[@class="info"]/p[@class="info_left left"]/a[1]/@data-key'))
        # 摘要
        abstract = list_to_str(get_abstract(url=link))
        # 文献来源
        paper_source = list_to_str(item.xpath('./p[@class="source"]/span[last()-2]/text() | ./p[@class="source"]/a[1]/span[1]/text()'))
        # 文献类型
        paper_type = list_to_str(item.xpath('./p[@class="source"]/span[last()]/text()'))
        # 下载量
        download = list_to_str(item.xpath('./div[@class="info"]/p[@class="info_right right"]/span[@class="time1"]/text()'))
        # 被引量
        refer = list_to_str(item.xpath('./div[@class="info"]/p[@class="info_right right"]/span[@class="time2"]/text()'))

        item_info = [i.strip() for i in [title, author, paper_source, paper_type, date, abstract, keywords, download, refer, link]]
        page_info.append(item_info)
    
    # 将爬取的数据转换为 DataFrame
    df = pd.DataFrame(page_info, columns=['Title', 'Author', 'Source', 'Type', 'Date', 'Abstract', 'Keywords', 'Download', 'Citations', 'href'])
    
    # 删除 Title 为空的行
    df = df[df['Title'] != '']
    # 删除 Abstract 为空的行
    df = df[df['Abstract'] != '']

    df.to_csv(temp_save_path, index=False)  # 覆写临时文件
    print(f"成功保存{len(df)}条结果到{temp_save_path}")
    
    return df


