# -*- coding: utf-8 -*-
from scrapy import Spider
from selenium import webdriver
from scrapy.selector import Selector
from scrapy.http import Request
from time import sleep
from selenium.common.exceptions import NoSuchElementException

class RespapSpider(Spider):
	name = 'respap'
	allowed_domains = ['semanticscholar.org']

	def start_requests(self):
		self.driver = webdriver.Chrome(executable_path='C:/Users/abhinav/Downloads/chromedriver.exe')
		self.driver.get('https://www.semanticscholar.org/search?year%5B0%5D=2014&year%5B1%5D=2019&publicationType%5B0%5D=JournalArticle&q=blockchain&sort=relevance&page=105')

		sleep(20)
		sel= Selector(text=self.driver.page_source)
		papers= sel.xpath('//*[@class="search-result"]/*[@class="search-result-header"]/*[@class="search-result-title"]/a/@href').extract()
		for paper in papers:
			url='https://semanticscholar.org' + paper                             
			yield Request(url, callback=self.parse_paper)

	def parse_paper(self, response):
		title=response.xpath('//h1/text()').extract()
		authors=response.xpath('//*[@class="subhead"]/li/*[@class="author-list"]/*[@data-heap-id=""]/*[@class="author-list__link author-list__author-name"]/*[@class=""]/span/text()').extract()
		journal=response.xpath('//*[@data-selenium-selector="venue-metadata"]/*[@class=""]/span/text()').extract()
		year_of_publication=response.xpath('//*[@data-selenium-selector="paper-year"]/*[@class=""]/span/text()').extract()
		abstract=response.xpath('//*[@class="text-truncator abstract__text text--preline"]/text()').extract()
		no_of_citations=response.xpath('//*[@class="scorecard_stat__headline__dark"]/text()').extract()
		highly_influenced_papers=response.xpath('//*[@class="scorecard__description__highlight"]/*[@class="scorecard__description__number"]/text()').extract()
		cite_background=response.xpath('//*[@data-heap-key="background"]/*[@class="scorecard__description__number"]/text()').extract()
		cite_method=response.xpath('//*[@data-heap-key="methodology"]/*[@class="scorecard__description__number"]/text()').extract()
		cite_result=response.xpath('//*[@data-heap-key="result"]/*[@class="scorecard__description__number"]/text()').extract()
		twitter_mention=response.xpath('//*[@class="scorecard_stat__headline"]/text()').extract()
		citation_titles=response.xpath('//*[@id="citing-papers"]/*[@class="card-content card-content__children-own-layout"]/*[@class="card-content-main"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__title"]/*[@data-selenium-selector="title-link"]/*[@class=""]/span/text()').extract()
		citation_journals = response.xpath('//*[@id="citing-papers"]/*[@class="card-content card-content__children-own-layout"]/*[@class="card-content-main"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__meta"]/*[@class="citation__meta__publication"]/*[@data-selenium-selector="venue-metadata"]/text()').extract()
		citation_dates = response.xpath('//*[@id="citing-papers"]/*[@class="card-content card-content__children-own-layout"]/*[@class="card-content-main"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__meta"]/*[@class="citation__meta__publication"]/*[@data-selenium-selector="paper-year"]/text()').extract()
		reference_titles= response.xpath('//*[@id="references"]/*[@class="card-content"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__title"]/*[@data-selenium-selector="title-link"]/*[@class=""]/span/text()').extract()
		reference_journals= response.xpath('//*[@id="references"]/*[@class="card-content"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__meta"]/*[@class="citation__meta__publication"]/*[@data-selenium-selector="venue-metadata"]/text()').extract()
		reference_dates= response.xpath('//*[@id="references"]/*[@class="card-content"]/*[@class="paper-detail-content-card"]/*[@class="citation-list__citations"]/*[@class="paper-citation"]/*[@class="citation__body"]/*[@class="citation__meta"]/*[@class="citation__meta__publication"]/*[@data-selenium-selector="paper-year"]/text()').extract()
		yield {'Title of the Article':title,'Authors':authors,'Published Journal':journal,"Year of Publication":year_of_publication,"Abstract of the article":abstract,"Number of Citations":no_of_citations,"Highly Influenced Papers":highly_influenced_papers,"Cite Background":cite_background,"Cite Methods":cite_method,"Cite Results":cite_result,"Twitter Mentions":twitter_mention,"Citations Titles":citation_titles,"Citations Journals":citation_journals,"Citations Dates":citation_dates,"Reference Titles":reference_titles,"Reference Journals":reference_journals,"Reference Dates":reference_dates}

