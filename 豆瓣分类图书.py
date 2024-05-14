import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# In[ ]:

# 包装爬虫函数

# 获取书籍列表和id
def get_postlist_and_ids(url, url_xpath, file_path):
    driver = webdriver.Chrome()
    driver.get(url) # 打开该类图书网页
    time.sleep(2)
    postlist = driver.find_elements(By.XPATH, url_xpath)
    print(len(postlist)) # 20
    result_id = {} # 用于存储该类图书的书名和id
    for post in postlist:
        title = post.text.strip().replace('\n', '')
        link = post.get_attribute('href')
        id = link.split('/')[-2]
        result_id[title] = id
    driver.quit()
    return result_id

# 获取每本书的评论，保证该类别至少1000条
def get_comments(id_type, comment_xpath,file_path):
    comments = []
    for title, id in id_type.items():
        # 获取每本图书的评论网址
        comment_link = r"https://book.douban.com/subject/"+id+r"/comments/"
        current_page = 0
        driver = webdriver.Chrome()
        # 翻页3次，每页20条评论
        for _ in range(4):
            try:
                url = comment_link + "?start=" + str(current_page) + "&limit=20&status=P&sort=score"
                driver.get(url)
                time.sleep(2)
                comment_elements = driver.find_elements(By.XPATH, comment_xpath)
                for comment_element in comment_elements:
                    comment_text = comment_element.text.replace('\n', '')
                    comments.append(comment_text)
                    # 直接写入文件
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(comment_text + "\n")
                current_page += 20
                time.sleep(4)
            except:
                print("爬取网页" + url + "失败")
        driver.quit()
    print(f"这类图书共爬取 {len(comments)} 条评论")
    return comments

# 输出评论txt函数

def write_comments_to_file(comments_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for comment in comments_list:
            f.write(comment + "\n")


# In[ ]:

# 分类爬取

# 推理
url_tuili = r'https://book.douban.com/tag/%E6%8E%A8%E7%90%86'
url_xpath_tuili = r"/html/body/div[3]/div[1]/div/div[1]/div/ul/li/div[2]/h2/a"
id_tuili = get_postlist_and_ids(url_tuili, url_xpath_tuili)
comment_xpath_tuili = r"/html/body/div[3]/div[1]/div/div[1]/div/div[4]/div[1]/ul/li/div[2]/p/span" 
file_path_tuili = r"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_推理.txt"
comments_tuili = get_comments(id_tuili, comment_xpath_tuili, file_path_tuili)

# 历史
url_lishi = r'https://book.douban.com/tag/%E5%8E%86%E5%8F%B2'
url_xpath_lishi = r"/html/body/div[3]/div[1]/div/div[1]/div/ul/li/div[2]/h2/a"
id_lishi = get_postlist_and_ids(url_lishi, url_xpath_lishi)
comment_xpath_lishi = r"/html/body/div[3]/div[1]/div/div[1]/div/div[4]/div[1]/ul/li/div[2]/p/span" 
file_path_lishi = r"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_历史.txt"
comments_lishi = get_comments(id_lishi, comment_xpath_lishi, file_path_lishi)

# 心理学
url_xinli = r'https://book.douban.com/tag/%E5%BF%83%E7%90%86%E5%AD%A6'
url_xpath_xinli = r"/html/body/div[3]/div[1]/div/div[1]/div/ul/li/div[2]/h2/a"
id_xinli = get_postlist_and_ids(url_xinli, url_xpath_xinli)
comment_xpath_xinli = r"/html/body/div[3]/div[1]/div/div[1]/div/div[4]/div[1]/ul/li/div[2]/p/span" 
file_path_xinli = r"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_心理.txt"
 

# 哲学
url_zhexue = r'https://book.douban.com/tag/%E5%93%B2%E5%AD%A6'
url_xpath_zhexue = r"/html/body/div[3]/div[1]/div/div[1]/div/ul/li/div[2]/h2/a"
id_zhexue = get_postlist_and_ids(url_zhexue, url_xpath_zhexue)
comment_xpath_zhexue = r"/html/body/div[3]/div[1]/div/div[1]/div/div[3]/div[1]/ul/li/div[2]/p/span" 
file_path_zhexue = r"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_哲学.txt"
comments_zhexue = get_comments(id_zhexue, comment_xpath_zhexue, file_path_zhexue)

# 科幻
url_kehuan = r'https://book.douban.com/tag/%E7%A7%91%E5%B9%BB'
url_xpath_kehuan = r"/html/body/div[3]/div[1]/div/div[1]/div/ul/li/div[2]/h2/a"
id_kehuan = get_postlist_and_ids(url_kehuan, url_xpath_kehuan)
comment_xpath_kehuan = r"/html/body/div[3]/div[1]/div/div[1]/div/div[4]/div[1]/ul/li/div[2]/p/span" 
file_path_kehuan = r"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_科幻.txt"
comments_kehuan = get_comments(id_kehuan, comment_xpath_kehuan, file_path_kehuan)

# In[ ]:

# 词云绘制

import numpy as np
import jieba
import jieba.analyse
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image

def generate_word_cloud(text_file, image_file, save_path):
    # 读取文本,分词
    text = open(text_file, encoding="utf-8").read()
    text = ' '.join(jieba.cut(text))

    # 提取关键词与权重
    freq = jieba.analyse.extract_tags(text, topK=200, withWeight=True)
    freq = {i[0]: i[1] for i in freq}

    # 生成对象
    pic = np.array(Image.open(image_file))
    wc = WordCloud(
        font_path='C:\Windows\Fonts\FZQTJW.TTF', 
        mask=pic,
        background_color="white",
        mode='RGBA',
        ).generate_from_frequencies(freq)
    image_colors = ImageColorGenerator(pic)
    wc.recolor(color_func=image_colors)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # 保存文件
    wc.to_file(save_path)

# 生成五个主题的词云
types = ['历史', '哲学', '推理', '心理', '科幻']
for type_ in types:
    text_file = rf"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\comments_{type_}.txt"
    image_file = rf"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\{type_}背景.jpg"
    save_path = rf"D:\Desktop\本科\大三\python编程与数据分析\作业\爬虫作业\{type_}词云1.png"
    generate_word_cloud(text_file, image_file, save_path)