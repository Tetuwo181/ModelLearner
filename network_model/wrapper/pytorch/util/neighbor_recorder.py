
class NeighborRecorder(object):

    def __init__(self, record_max_num:int):
        self.__record_max_num = record_max_num
        self.__record_distances = []
        self.__record_indexes = []

    @property
    def data_num(self) -> int:
        return len(self.__record_distances)

    @property
    def count_num(self) -> int:
        return self.data_num if self.data_num < self.__record_max_num else self.__record_max_num

    def get_predicted_index(self):
        if self.data_num == 0:
            return -1
        return sum(self.__record_indexes)//self.count_num

    def record(self, distance, class_index):
        if self.data_num == 0:
            self.__record_distances.append(distance)
            self.__record_indexes.append(class_index)
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

