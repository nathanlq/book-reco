import scrapy
import logging
import re


class FuretSpider(scrapy.Spider):
    name = 'furet'

    start_urls = [
        'https://www.furet.com/livres/litterature/romans/meilleures-ventes.html',
        'https://www.furet.com/livres/litterature/policiers-et-thrillers/meilleures-ventes.html',
        'https://www.furet.com/livres/litterature/romances/meilleures-ventes.html',
        'https://www.furet.com/livres/litterature/livres-de-poche/meilleures-ventes-poche.html',
        'https://www.furet.com/livres/litterature/science-fiction-et-fantasy.html',
        'https://www.furet.com/livres/litterature/theatre-et-poesie/meilleures-ventes.html',
        'https://www.furet.com/livres/litterature/lettres-superieures/meilleures-ventes.html'
        'https://www.furet.com/livres/litterature/recits-de-voyages.html',
        'https://www.furet.com/livres/litterature/recits-de-voyages.html',
        'https://www.furet.com/livres/litterature/pleiades.html',
    ]

    unique_products = set()

    explored_urls = set()

    def parse(self, response):
        logging.debug(f"Parsing URL: {response.url}")

        links = response.css('a::attr(href)').getall()

        product_url_pattern = re.compile(
            r'https://www\.furet\.com/livres/.*-\d+\.html$')

        for link in links:
            absolute_link = response.urljoin(link)
            if product_url_pattern.match(absolute_link) and absolute_link not in self.explored_urls:
                self.explored_urls.add(absolute_link)
                yield scrapy.Request(absolute_link, callback=self.parse_product)

        pagination_links = response.css(
            'a.dct-btn-pager.box.page-item::attr(href)').getall()
        for pagination_link in pagination_links:
            absolute_pagination_link = response.urljoin(pagination_link)
            if absolute_pagination_link not in self.explored_urls:
                self.explored_urls.add(absolute_pagination_link)
                yield scrapy.Request(absolute_pagination_link, callback=self.parse)

    def parse_product(self, response):
        logging.debug(f"Parsing product URL: {response.url}")

        product_title = response.css('h1.product-title::text').get()
        author = response.css('a.author::text').get()
        resume = response.css('div#resume div.content::text').get()

        if product_title and author:
            product_title = product_title.strip()
            author = author.strip()
            resume = resume.strip() if resume else ''
            logging.debug(f"Product title: {product_title}")

            product_id = f"{product_title.lower()}-{author.lower()}"

            if product_id not in self.unique_products:
                self.unique_products.add(product_id)

                labels = response.css('ul li a span::text').getall()

                information = {}
                info_items = response.css(
                    'ul.informations-container li.information')
                for item in info_items:
                    name = item.css('div.name span::text').get()
                    value_text = item.css('div.value::text').get()
                    value_link = item.css('div.value a::text').get()
                    value = value_link if value_link else value_text
                    if name and value:
                        information[name.strip()] = value.strip()
                image_url = response.css('meta[property="og:image"]::attr(content)').get()

                yield {
                    'product_title': product_title,
                    'author': author,
                    'resume': resume,
                    'labels': labels,
                    'information': information,
                    'image_url': image_url
                }
