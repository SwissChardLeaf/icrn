from abc import ABC, abstractmethod


class OrderedSymbol(ABC):
    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __gt__(self, other):
        pass

    @abstractmethod
    def __le__(self, other):
        pass

    @abstractmethod
    def __ge__(self, other):
        pass


class BaseSymbol(ABC):
    pass


class IndexedSymbol(ABC):
    pass


# class
