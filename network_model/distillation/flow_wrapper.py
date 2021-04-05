class FlowForDistillation(object):
    """
    こちらの記事を参考に実装しました
    https://tripdancer0916.hatenablog.com/entry/2018/10/14/keras%E3%81%A7%E7%9F%A5%E8%AD%98%E3%81%AE%E8%92%B8%E7%95%99_1
    """

    def __init__(self, base_iterator):
        self.__iterator = base_iterator

    def __iter__(self):
        return self

    def __next__(self):
        base_result = next(self.__iterator)
        return (base_result[0], base_result[1]), base_result[1]
