# coding=utf-8 #
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import quote
import re
import ssl
import json
import multiprocessing as mp
import urllib
from urllib.error import URLError
import time
import validators

ssl._create_default_https_context = ssl._create_unverified_context
BASE_URL = "https://cn.nytimes.com"
QUERY_URL = "https://www.nytimes.com/search?query="
# CATEGORIES = ['world', 'china', 'business', 'technology', 'science', 'health', 'education', 'culture']
CATEGORIES = ["op-ed", "lens", "style", "travel"]

LOG_FILE = open("./LOG/Analysis.txt", 'a', encoding='utf-8')  # 记录爬取的条数
UNOPENED_NEWS_URL = open("./LOG/Unopened_news.txt", 'a', encoding='utf-8')  # 记录没能成功打开的新闻页面链接
UNOPENED_NAVI_URL = open("./LOG/Unopened_navi.txt", 'a', encoding='utf-8')  # 记录没能成功打开的导航页面链接


# 三个跳出条件：1.导航列表页不再存在下一页时跳出 2.不存在当前新闻页面的双语页面时 Continue 3. 不存在当前新闻页面的英文页面时 Continue
# urllib.error.HTTPError: HTTP Error 404: Not Found   for 1.
# urllib.error.HTTPError: HTTP Error 404: Not Found   for 2.
# len(soup.find_all('link', {"rel": "alternate", "hreflang": "en-us"}))   for 3.

def parse_news_page(
        url: str,
        news_type=None
) -> dict or None:
    """
    解析一个新闻页面并返回含有中英双语的原文和摘要的字典
    """
    html = None
    url = url.replace(" ", "")
    valid = validators.url(url)
    if valid != True:
        print(f"URL: {url} 不合法!")
        return
    time.sleep(5)
    try:
        html = urlopen(
            url=url
        ).read().decode("utf-8")

    except (OSError, URLError, urllib.error.HTTPError) as e:
        # 捕捉网络错误和HTTP错误，继续尝试
        try_time = 20
        flag = False
        if "404" in str(e):
            # 如果是404错误，说明页面不存在，直接返回
            print(f"新闻 {url} 没有双语版本,跳过此条")
            return
        if "403" in str(e):
            # 如果报403，说明被ban，脚本挂起5分钟
            print("ip被ban，挂起5分钟")
            time.sleep(300)
        time.sleep(5)
        print(f"新闻页 {url} 打开失败，正在尝试重新打开...")
        for i in range(try_time):
            try:
                time.sleep(5)
                html = urlopen(url).read().decode('utf-8')
            except urllib.error.HTTPError as ee:
                # 报404了，记录报404的次数，如果超过最大10次就说明这个网页不存在
                if "404" in str(ee):
                    # 如果是404错误，说明页面不存在，直接返回
                    print(f"新闻 {url} 没有双语版本,跳过此条")
                    return
                elif "403" in str(ee):
                    # 如果报403，说明被ban，脚本挂起5分钟
                    print("ip被ban，挂起5分钟")
                    time.sleep(300)
                else:
                    time.sleep(5)
                    print(f"重试第{i + 1}次...")
            except (OSError, URLError):
                time.sleep(5)
                print(f"重试第{i + 1}次...")
            else:
                print("重新打开新闻页URL成功!")
                flag = True
                break
        if not flag:
            print("重新打开新闻页URL失败")
            return
    if not html:
        print(f"新闻页 {url} 没读出东西，咋回事儿家人们？赶紧排查!")
        print(url, file=UNOPENED_NEWS_URL)
        return
    soup = BeautifulSoup(html, features='lxml')
    try:
        chinese_title = soup.find_all('meta', {'property': "og:title"})[0]['content']
        english_title = soup.find_all('h1', {"class": "en-title"})[0].get_text()
        content = soup.find_all('div', {'class': "article-paragraph"})
        chinese_summary = soup.find_all('meta', {'name': "description"})[0]['content']
    except Exception as e:
        print(f"获取标题、中文摘要和双语内容时发生错误，错误信息:{e}")
        return
    english_summary = None
    content_list = []

    sent_dict = {}
    for i, c in enumerate(content):
        string = c.get_text()
        if i % 2 == 0:
            sent_dict['english'] = string
        else:
            sent_dict['chinese'] = string

        if (i + 1) % 2 == 0:
            sent_dict['index'] = i // 2
            content_list.append(sent_dict)
            sent_dict = {}

    eng_page_url = soup.find_all('link', {"rel": "alternate", "hreflang": "en-us"})
    if not eng_page_url:
        print(f"新闻 {url} 没有英语版本跳转链接,正在尝试搜索...")
        # todo 如果没有英语版本跳转链接，那么直接到纽约时报英文网去搜索这篇新闻的英文标题从而获得摘要
        search_page = None
        query = quote(english_title)
        search_url = QUERY_URL + query
        try:
            search_page = urlopen(
                url=search_url
            ).read().decode('utf-8')
        except (OSError, URLError, urllib.error.HTTPError) as e:
            try_time = 20
            flag = False
            if "404" in str(e):
                # 如果是404错误，说明页面不存在，直接返回
                print(f"新闻 {url} 在纽约时报英文网上搜索页面报404错误！")
                return
            if "403" in str(e):
                # 如果报403，说明被ban，脚本挂起5分钟
                print("ip被ban，挂起5分钟")
                time.sleep(300)
            time.sleep(5)
            print(f"新闻 {url} 在纽约时报英文网上的搜索页连接失败，正在重新尝试...")
            for i in range(try_time):
                try:
                    time.sleep(5)
                    search_page = urlopen(
                        url=search_url
                    ).read().decode('utf-8')
                except urllib.error.HTTPError as ee:
                    if "404" in str(ee):
                        # 如果是404错误，说明页面不存在，直接返回
                        print(f"新闻 {url} 在纽约时报英文网上搜索页面报404错误！")
                        return
                    elif "403" in str(ee):
                        # 如果报403，说明被ban，脚本挂起5分钟
                        print("ip被ban，挂起5分钟")
                        time.sleep(300)
                    else:
                        time.sleep(5)
                        print(f"重试第{i + 1}次...")
                except (OSError, URLError):
                    print(f"重试第{i + 1}次...")
                    time.sleep(5)
                else:
                    print(f"重新打开新闻{url}在纽约时报英文网上的搜索页成功!")
                    flag = True
                    break
            if not flag:
                print(f"重新打开新闻{url}在纽约时报英文网上的搜索页失败!")
                return

        if search_page is not None:
            search_soup = BeautifulSoup(
                search_page,
                features='lxml'
            )
            headlines = search_soup.find_all("h4", {"class": "css-2fgx4k"})
            summaries = search_soup.find_all("p", {"class": "css-16nhkrn"})
            if not headlines or not summaries:
                print(f"搜索页显示无相关结果！")
                return
            headline_html, summary_html = headlines[0], summaries[0]
            headline = headline_html.get_text()
            summary = summary_html.get_text()
            if headline.strip() == english_title.strip():
                english_summary = summary
                print(f"在纽约时报英文网通过搜索标题找到了新闻 {url} 的英文摘要!")
            else:
                print(f"在纽约时报英文网通过搜索标题没有找到新闻 {url} 的英文摘要!")
                return

    else:
        time.sleep(5)
        eng_page = None
        try:
            eng_page = urlopen(
                url=eng_page_url[0]['href']
            ).read().decode('utf-8')
        except (OSError, URLError, urllib.error.HTTPError) as e:
            try_time = 20
            flag = False
            if "404" in str(e):
                # 如果是404错误，说明页面不存在，直接返回
                print(f"新闻 {url} 没有英语版本,跳过此条")
                return
            if "403" in str(e):
                # 如果报403，说明被ban，脚本挂起5分钟
                print("ip被ban，挂起5分钟")
                time.sleep(300)
            time.sleep(5)
            print(f"英文新闻页面 {eng_page_url[0]['href']} 打开失败，正在尝试重新打开... ")

            for i in range(try_time):
                try:
                    time.sleep(5)
                    eng_page = urlopen(
                        url=eng_page_url[0]['href']
                    ).read().decode('utf-8')
                except urllib.error.HTTPError as ee:
                    if "404" in str(ee):
                        # 如果是404错误，说明页面不存在，直接返回
                        print(f"新闻 {url} 没有英语版本,跳过此条")
                        return
                    elif "403" in str(ee):
                        # 如果报403，说明被ban，脚本挂起5分钟
                        print("ip被ban，挂起5分钟")
                        time.sleep(300)
                    else:
                        time.sleep(5)
                        print(f"重试第{i + 1}次...")
                except (OSError, URLError):
                    print(f"重试第{i + 1}次...")
                    time.sleep(5)
                else:
                    print("重新打开新闻页URL成功!")
                    flag = True
                    break
            if not flag:
                print(f"英文新闻页面 {eng_page_url[0]['href']} 重新打开失败")
                return
        if eng_page is None:
            return
        eng_soup = BeautifulSoup(eng_page, features='lxml')
        english_summary = eng_soup.find_all('meta', {'name': "description"})[0]['content']

    data = dict(
        Category=news_type,
        Title={
            "english": english_title,
            "chinese": chinese_title
        },
        Content=content_list,
        Summary={
            "english": english_summary,
            "chinese": chinese_summary
        }
    )
    return data


def parse_navigation_page(
        url: str,
        news_type: str,
        fp,
) -> tuple or None:
    """
    解析新闻列表导航页，获取每个新闻的URL传到'parse_news_page'方法中进行解析
    :param url: 导航页网址
    :param news_type: 新闻类别
    :param fp: 存储数据的文件指针
    :return: (数据dict,有效的数据条数) 或 None（如果为None的话说明爬完了）
    """
    html = None
    url = url.replace(" ", "")
    valid = validators.url(url)
    if valid != True:
        print(f"URL: {url} 不合法!")
        return [], 0
    time.sleep(5)
    try:
        html = urlopen(
            url=url
        ).read().decode('utf-8')

    except (OSError, URLError, urllib.error.HTTPError) as e:
        # 网络错误，继续尝试
        try_time = 20
        flag = False
        if "404" in str(e):
            # 如果是404错误，说明页面不存在，直接返回
            print(f"导航页 {url} 不存在")
            return
        if "403" in str(e):
            # 如果报403，说明被ban，脚本挂起5分钟
            print("ip被ban，挂起5分钟")
            time.sleep(300)

        time.sleep(5)
        print(f"导航列表页 {url} 打开失败，正在尝试重新打开...")

        for i in range(try_time):
            try:
                time.sleep(5)
                html = urlopen(
                    url=url
                ).read().decode('utf-8')
            except urllib.error.HTTPError as ee:
                # 报404,403了
                if "404" in str(ee):
                    # 如果是404错误，说明页面不存在，直接返回
                    print(f"新闻 {url} 没有双语版本,跳过此条")
                    return
                elif "403" in str(ee):
                    # 如果报403，说明被ban，脚本挂起5分钟
                    print("ip被ban，挂起5分钟")
                    time.sleep(300)
                else:
                    time.sleep(5)
                    print(f"重试第{i + 1}次...")
            except (OSError, URLError):
                time.sleep(5)
                print(f"重试第{i + 1}次...")
            else:
                print(f"重新打开导航列表页成功!")
                flag = True
                break
        if not flag:
            print(f"所有对重新打开导航列表页 {url} 的尝试都以失败告终 T_T ")
            print(url, file=UNOPENED_NAVI_URL)
            return [], 0

    if not html:
        print(f"导航列表页 {url} 读出个卵蛋，排查")
        return [], 0

    soup = BeautifulSoup(html, features='lxml')
    news_pages = soup.find_all('a',
                               {'target': '_blank',
                                'href': re.compile("/([a-z])*/(\d){8}/.+/$"),
                                'title': re.compile(".+$")})
    count = 0
    # res = []
    for page in news_pages:
        page_url = BASE_URL + page['href'] + 'dual/'
        time.sleep(5)
        data = parse_news_page(
            url=page_url,
            news_type=news_type
        )
        if data:
            count += 1
            # res.append(data)
            try:
                print(json.dumps(data, ensure_ascii=False), file=fp)
            except Exception as eee:
                print(f"写入失败,错误信息:{eee}")

    return count


def pipeline(
        news_type: str,
        file_path: str,
):
    """
    处理爬虫的总流程
    """
    page_num = 1
    fp = None
    try:
        fp = open(file_path, 'a', encoding='utf-8')
    except (FileExistsError, FileNotFoundError):
        print("打开文件失败，检查路径合法性")

    if not fp: return
    total = 0
    while True:
        print(f"开始处理{news_type}类新闻的第{page_num}页...")
        navi_url = BASE_URL + '/' + news_type + '/' + str(page_num) + '/'
        time.sleep(5)
        news_processed = parse_navigation_page(
            url=navi_url,
            news_type=news_type,
            fp=fp)
        if news_processed is None: break  # 说明该类别已经爬完了
        total += news_processed
        # items.extend(news_processed[0])
        page_num += 1

    print(f"{news_type}类型新闻爬取结束，共计爬取{total}条")
    print(f"{news_type}类型新闻爬取结束，共计爬取{total}条", file=LOG_FILE)
    fp.close()


for category in CATEGORIES:
    p = mp.Process(target=pipeline, args=(category, f"./Data/{category}.json"))
    p.start()

LOG_FILE.close()
UNOPENED_NAVI_URL.close()
UNOPENED_NEWS_URL.close()
