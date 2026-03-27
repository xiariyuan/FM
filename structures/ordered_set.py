# Copyright (c) Ruopeng Gao. All Rights Reserved.

from collections import OrderedDict


class OrderedSet:
    def __init__(self):
        self.dict = OrderedDict()

    def add(self, value):
        if value in self.dict:
            del self.dict[value]
        self.dict[value] = None

    def discard(self, value):
        if value in self.dict:
            del self.dict[value]

    def remove(self, value):
        if value not in self.dict:
            raise KeyError(value)
        del self.dict[value]

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict.keys())

    def __contains__(self, value):
        return value in self.dict


if __name__ == '__main__':
    ordered_deque = OrderedSet()
    ordered_deque.add(1)
    ordered_deque.add(3)
    ordered_deque.add(2)
    ordered_deque.add(1)
    print(list(ordered_deque))
    print(4 in ordered_deque)
