from numba import jit


class NeighborRecorder(object):

    def __init__(self, record_max_num: int):
        self.__record_max_num = record_max_num
        self.__record_distances = None
        self.__record_indexes = None

    @property
    def data_num(self) -> int:
        if self.__record_distances is None:
            return 0
        return len(self.__record_distances)

    @property
    def count_num(self) -> int:
        return self.data_num if self.data_num < self.__record_max_num else self.__record_max_num

    def append_distances(self, distances):
        if self.__record_distances is None:
            self.__record_distances = distances
            return self
        self.__record_distances = self.__record_distances + distances
        return self

    def append_indexes(self, indexes):
        if self.__record_indexes is None:
            self.__record_indexes = indexes
            return self
        self.__record_indexes = self.__record_distances + indexes
        return self

    def get_predicted_index(self):
        if self.data_num == 0:
            return -1
        result = sum(self.__record_indexes)//self.count_num
        return result

    def record_distances(self, distances, class_indexes):
        for distance, class_index in zip(distances, class_indexes):
            self.record(distance, class_index)

    def record(self, distance, class_index):
        if self.data_num == 0:
            self.__record_distances = [distance]
            self.__record_indexes = [class_index]
            return self
        if self.__record_distances[0] > distance:
            self.__record_distances.insert(0, distance)
            self.__record_indexes.insert(0, class_index)
            return self

        for index in range(self.count_num - 1):
            if self.__record_distances[index] > distance > self.__record_distances[index + 1]:
                self.__record_distances.insert(index+1, distance)
                self.__record_indexes.insert(index+1, class_index)
                if self.data_num > self.count_num:
                    self.__record_distances = self.__record_distances[:self.count_num]
                    self.__record_indexes = self.__record_indexes[:self.count_num]
                return self
        return self

