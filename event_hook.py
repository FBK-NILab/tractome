# -*- coding: utf-8 -*-

"""

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

class EventHook(object):
    """
    Class that allows simulating the events-delegate functionality.
    Check later if there is a simplest way to do it in a more
    Pythonian way.
    """
    def __init__(self):
        self.__handlers = []
 
    def __iadd__(self, handler):
        self.__handlers.append(handler)
        return self
 
    def __isub__(self, handler):
        self.__handlers.remove(handler)
        return self
         
    def fire(self, *args, **keywargs):
        for handler in self.__handlers:
            handler(*args, **keywargs)
         
    def clearObjectHandlers(self, inObject):
        for theHandler in self.__handlers:
            if theHandler.im_self == inObject:
                self -= theHandler
