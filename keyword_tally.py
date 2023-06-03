# ./keyword_tally.py

class KeywordTally:
    def __init__(self):
        self.keyword_tally_count = {}
        self.total_keywords = 0
        self.most_common_count = 0

    def tally(self, keywords):
        for keyword in keywords:
            self.total_keywords += 1
            if keyword in self.keyword_tally_count:
                self.keyword_tally_count[keyword] += 1
            else:
                self.keyword_tally_count[keyword] = 1

            if self.keyword_tally_count[keyword] > self.most_common_count:
                self.most_common_count = self.keyword_tally_count[keyword]

    def get_significance(self, keywords):
        significance = 0
        for keyword in keywords:
            if keyword in self.keyword_tally_count:
                ratio = self.keyword_tally_count[keyword] / self.most_common_count
                significance += 1 - ratio
        return significance / len(keywords)
    
    def exportKeywordTally(self):
        return self.keyword_tally_count

    def importKeywordTally(self, keyword_tally_data):
        self.keyword_tally_count = keyword_tally_data
        self.total_keywords = sum(keyword_tally_data.values())
        self.most_common_count = max(keyword_tally_data.values())