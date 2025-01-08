from scrapy.dupefilters import BaseDupeFilter
from scrapy.utils.request import request_fingerprint

class CustomDupeFilter(BaseDupeFilter):
    def __init__(self):
        self.visited = set()

    def request_seen(self, request):
        fp = request_fingerprint(request)
        if fp in self.visited:
            return True
        self.visited.add(fp)
        return False
