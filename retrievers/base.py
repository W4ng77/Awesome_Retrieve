class BaseRetriever:
    def __init__(self):
        pass

    def build_index(self, corpus):
        raise NotImplementedError

    def retrieve(self, query, top_k=5):
        raise NotImplementedError
