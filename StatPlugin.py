from abc import ABCMeta, abstractmethod

class StatPlugin(object):
    __metaclass__ = ABCMeta
    ENABLED = True

    # required_data returns a list() of strings that provide the names of the data that this stat wants to process
    @abstractmethod
    def required_data(self):
        return list()

    # return_stats returns a dict() of the stat identifier as a string (key) that maps to the stat value
    @abstractmethod
    def return_stats(self, data):
        return dict()