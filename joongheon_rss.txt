In [26]:
!pip install feedparser
!pip install newspaper3k
!pip install konlpy

import feedparser
from newspaper import Article
from konlpy.tag import Okt
from collections import Counter
from operator import eq
import numpy as np

urls = ("http://rss.etnews.com/Section901.xml"
	, None)
#	, "http://rss.etnews.com/Section902.xml"
#	, "http://rss.etnews.com/Section903.xml"

def get_tags(text, ntags=50):
	num_unique_words = 0
	num_most_freq = 0
	ranking = 0
	spliter = Okt()
	nouns = spliter.nouns(text)
	count = Counter(nouns)
	return_list = []
	for n, c in count.most_common(ntags):
		ranking = ranking + 1		
		temp = {'tag': n, 'count': c, 'ranking': ranking}
		return_list.append(temp)
		num_unique_words = num_unique_words + 1		
		if num_unique_words == 1:
			num_most_freq = c
	return num_unique_words, num_most_freq, return_list

def TF(request, most_freq, tag):
	return 0.1 + 0.9*Howmanywords(request, tag)/most_freq

def Howmanywords(request, tag):
	nWords = 0
	nRanking = 0
	for n in tag:
		noun = n['tag']
		count = n['count']
		rank = n['ranking']
		if eq(noun, request):
			nWords = count
			nRanking = rank
	return nWords, nRanking

def crawl_rss(urls):
	arr_rss = []
	for url in urls:
		print("[Crawl RSS] ",url)
		parse_rss = feedparser.parse(url)
		for p in parse_rss.entries:
			arr_rss.append({'title':p.title, 'link':p.link})
	return arr_rss
	
def crawl_article(url, language='ko'):
	print("[Crawl Article] ",url)
	var_article = Article(url, language=language)
	var_article.download()
	var_article.parse()
	return var_article.title, var_article.text
	
def main():
	spliter = Okt()
	article_list = crawl_rss(urls)
	print(article_list)
	for article in article_list:
		_, text = crawl_article(article['link'])
		article['text'] = text
	print(article_list)
		
	print('[Parsing Text]')
	for article in article_list:
		num_unique_words, num_most_freq, tags = get_tags(article['text'], 1000)
		article['tags'] = tags
		article['num_unique_words'] = num_unique_words
		article['num_most_freq'] = num_most_freq

	print('[Query]')
	query = input() # query
	for article in article_list:
		n, _ = Howmanywords(query, article['tags'])
		if n != 0:
			print("TF: ", n, article['title'], article['link'])
		
	print('[Parsing Title]')
	for article in article_list:
		article['title_noun'] = spliter.nouns(article['title'])
		print("(중복제거전)")
		print(article['title_noun'])
		print("(중복제거후)")
		print(list(set(article['title_noun'])))
		print("")

	print('[Filtering Fishing Articles]')
	# 기사마다 제목의 각각 명사들이 본문에 몇 번 나오는지 세고 합산결과 출력
	for article in article_list: # 매 기사마다
		n = 0
		for t in list(set(article['title_noun'])): # 매 기사의 제목의 명사마다
			nW, _ = Howmanywords(t, article['tags'])
			n = n + nW
		n = n / len(article['title_noun'])
		print("TF (title): ", format(n, ".2f"), article['title'], article['link'])

	# 문제1) 제목에서 쿼리 검색
	print("[문제1]")
	for article in article_list: # 매 기사마다
		flag = 0
		for t in list(set(article['title_noun'])): # 매 기사의 제목의 명사마다
			if eq(query, t):
				print(article['title'], article['link'])
	
	# 문제2) 본문에서의 쿼리 검색 후, 쿼리의 (TF, TF등수, 제목, URL) 출력
	print("[문제2]")
	for article in article_list:
		n, r = Howmanywords(query, article['tags'])
		if n != 0:
			print("TF: ", n, "TF ranking: ", r, article['title'], article['link'])
		
if __name__ == "__main__":
	main()
