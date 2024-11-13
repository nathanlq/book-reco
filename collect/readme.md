# Collect Module Documentation

For now the `collect` is only using base scrapy with a custom spyder, `FuretSpider`, the code can be found at `collect/furet_scraper/spiders/furet_spider.py`

The `FuretSpider` is a Scrapy spider designed to scrape book information from the "Furet" bookstore website, specifically focusing on bestsellers and popular books across various genres. This spider collects data such as the book's title, author, description, labels, and other metadata.

## Code Overview

The `FuretSpider` class inherits from `scrapy.Spider` and defines methods for extracting book details and handling pagination.

### Spider Details

- **Name**: `furet`
- **Start URLs**:  
  A list of URLs for different book categories on the Furet website, including literature, thrillers, romances, science fiction, and more. These URLs serve as entry points for the spider to begin its crawl.

### Key Attributes

- **`unique_products`**:  
  A set used to track unique books by creating a unique identifier based on each book's title and author. This prevents duplicate books from being scraped.

- **`explored_urls`**:  
  A set used to store URLs that have already been visited by the spider, which helps avoid re-fetching and re-processing the same pages.

## Methods

### `parse`

- **Purpose**:  
  This is the main parsing method for each category page, responsible for finding links to individual book pages and pagination links to other category pages.

- **Process**:
  - **Product Links**:
    Extracts all links on the page and filters them using a regular expression (`product_url_pattern`) to find links that match individual product pages. If a product URL hasn’t been visited, it’s added to `explored_urls` and passed to `parse_product`.
  
  - **Pagination Links**:
    Finds pagination links on the category page and recursively requests each page until all relevant book listings are collected. Each unique pagination link is added to `explored_urls` to avoid duplicate requests.

- **Logging**:
  Logs the URL of the page being parsed for tracking purposes.

### `parse_product`

- **Purpose**:  
  Parses individual product pages to extract specific details about each book, including title, author, summary, labels, additional information, and image URL.

- **Process**:
  - **Extracted Fields**:
    - `product_title`: The title of the book.
    - `author`: The author of the book.
    - `resume`: The book's summary or description.
    - `labels`: Category labels associated with the book (e.g., genre or tags).
    - `information`: Additional information about the book, such as ISBN, editor, and dimensions, extracted from name-value pairs on the page.
    - `image_url`: URL to the book's cover image.
  
  - **Deduplication**:
    - A unique identifier (`product_id`) is created for each book based on a combination of the title and author (converted to lowercase). If this identifier is not already in `unique_products`, it is added and the book details are yielded for further processing.

- **Logging**:  
  Logs the product URL and the product title for debugging and tracking purposes.

### Yielded Data

Each book is represented as a dictionary with the following structure:

```python
{
    'product_title': product_title,   # Book title as a string
    'author': author,                 # Author name as a string
    'resume': resume,                 # Book summary as a string
    'labels': labels,                 # List of category labels
    'information': information,       # Dictionary of additional info
    'image_url': image_url            # URL to book cover image
}
```

## Notes
- **Pagination Handling**:
The spider recursively handles pagination, ensuring that all relevant pages within each category are crawled.

- **Duplicate Avoidance**:
The spider efficiently avoids duplicate requests and duplicate items through `explored_urls` and `unique_products`, ensuring that each book is scraped only once.

- **Logging**:
Debug logs are included to track the URLs being parsed, product URLs visited, and product titles processed.