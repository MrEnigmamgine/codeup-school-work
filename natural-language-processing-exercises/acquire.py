# Core libraries
import os
import re

# Data Science libraries
import pandas as pd

# Webkit libraries
import requests
from bs4 import BeautifulSoup

SEED = 8
CACHE_DIR = './cache/'
CSV='./codeup-articles.csv'
CSV2 = './news-articles.csv'


def parse_codeup_article(url):
    headers = {'User-Agent': 'Codeup Data Science'}
    anything = re.compile('.+')
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup( response.content, 'html.parser') 
    article = soup.find('article', {'class': re.compile(r'post.+\s')})
    title = article.find('h1').text
    content = article.find('div', {'class':'entry-content'})
    return {
        'title': title,
        'published': article.find(anything, {'class':'published'}).text,
        'category': article.find(anything,{'rel':'category'}).text,
        'content': content.text
    }

def get_codeup_blog_page(url):
    headers = {'User-Agent': 'Codeup Data Science'}
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content)

def get_all_articles(blog_home, debug=False):
    page = get_codeup_blog_page(blog_home)
    article_urls = []
    page_n = 0
    next_page = True
    while next_page:
        page_n += 1
        if debug:
            article_urls.append(page_n)
        next_page = page.find('div', class_='alignleft').find('a')
        for article in page.find_all('article'):
            article_urls.append( article.find(class_='entry-title').find('a')['href'] )
        if next_page:
            page = get_codeup_blog_page(next_page['href'])
    return article_urls

def blog_new_data():
    blog_home = 'https://codeup.com/blog/'
    all_articles = get_all_articles(blog_home=blog_home)
    data = []
    for url in all_articles:
        try:
            article_dict = parse_codeup_article(url)
            data.append(article_dict)
        except:
            print(f'Failed to parse {url} ... Skipping')
        finally:
            pass
    df = pd.DataFrame(data)
    return df

def blog_get_data(refresh=False):
    path = CACHE_DIR+CSV
    cached = os.path.isfile(path)
    if refresh or not cached:
        df = blog_new_data()
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df

def get_soup(url, headers = {'User-Agent': 'Codeup Data Science'}):
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content)

def news_new_data():
    base_url = 'https://inshorts.com/en/read/'
    categories = [
        'business',
        'sports',
        'technology',
        'entertainment',
        ]
    data = []
    for category in categories:
        soup = get_soup(base_url+category)
        cards = soup.find_all(class_='news-card' )
        for card in cards:
            article_dict = {
                'title': card.find(itemprop='headline').string,
                'author': card.find(class_='author').string,
                'published': card.find(class_='time')['content'],
                'category': category,
                # 'source': card.find(class_='source')['href'],
                'content': card.find(itemprop='articleBody').text
            }
            data.append(article_dict)
    df = pd.DataFrame(data)
    return df

def news_get_data(refresh=False):
    path = CACHE_DIR+CSV2
    cached = os.path.isfile(path)
    if refresh or not cached:
        df = news_new_data()
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df