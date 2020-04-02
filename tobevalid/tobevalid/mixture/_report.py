# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:13:31 2019

@author: KavehB
"""


class Element:
    def __init__(self, parent=None):
        self.__parent = parent

    def parent(self):
        return self.__parent


class Nest(Element):
    def __init__(self, parent=None):
        Element.__init__(self, parent)
        self.__children = list()
        if parent == None:
            self.__depth = 1
        else:
            self.__depth = parent.depth() + 1

    def depth(self):
        return self.__depth

    def addchild(self, element):
        self.__children.append(element)

    def children(self):
        return self.__children


class VTable(Element):
    def __init__(self, parent, columns, head=""):
        Element.__init__(self, parent)
        self.__columns = columns
        self.__rows = list()
        self.__length = 0
        self.__head = head

    def addrow(self, row):
        self.__rows.append(row)
        self.__length += 1

    def length(self):
        return self.__length

    def columns(self):
        return self.__columns.copy()

    def data(self):
        return self.__rows.copy()


class HTable(Element):
    def __init__(self, parent, columns, head=""):
        Element.__init__(self, parent)
        self.__columns = columns
        self.__rows = dict()
        self.__length = 0
        self.__head = head

    def addrow(self, key, row):
        self.__rows[key] = row

    def columns(self):
        return iter(self.__columns)

    def data(self):
        return self.__rows.copy()


class Plot(Element):
    def __init__(self, parent, head, figure, func, title):
        Element.__init__(self, parent)
        self.__head = head
        self.__figure = figure
        self.__func = func
        self.__title = title

    def save(self, path):
        self.__figure.savefig(path + "/" + self.head + ".png")

    def figure(self):
        return self.__figure

    def head(self):
        return self.__head

    def func(self):
        return self.__func
    
    def title(self):
        return self.__title               


class Head(Nest):
    def __init__(self, head, parent=None):
        Nest.__init__(self, parent)
        self.__head = head

    def head(self):
        return self.__head


class Text(Element):
    def __init__(self, parent, text, name=None):
        Element.__init__(self, parent)
        self.__text = text
        self.__name = name

    def text(self):
        return self.__text

    def name(self):
        return self.__name


class Lines:
    def __init__(self, count=1):
        self.__count = count

    def count(self):
        return self.__count


class Report:
    def __init__(self, title):
        self.__title = title
        self.__elements = list()

    def title(self):
        return self.__title

    def head(self, string, child=False):
        if child:
            self.__last().addchild(Head(string, self.__last()))
        else:
            self.__elements.append(Head(string))
        return self

    def image(self, figure, func, name, title):
        self.__last().addchild(Plot(self.__last(), name, figure, func, title))
        return self

    def text(self, string, name=None):
        self.__last().addchild(Text(self.__last(), string, name))
        return self

    def vtable(self, columns, data, name=""):
        table = VTable(self.__last(), columns, name)
        for row in data:
            table.addrow(row)
        self.__last().addchild(table)
        return self

    def htable(self, columns, data, name=""):
        table = HTable(self.__last(), columns, name)
        for key, row in data.items():
            table.addrow(key, row)
        self.__last().addchild(table)
        return self

    def items(self):
        return iter(self.__elements)

    def __last(self):
        return self.__elements[-1]


class ReportGenerator:

    def __init__(self):
        self._extension = ""

    def save(self, report, path, name):
        self._open()
        self._dir = path

        self._title(report.title())
        for element in report.items():
            self._write(element)

        self._close()
        self._save(name + self._extension)

    def _write(self, element):
        if isinstance(element, Head):
            return self._head(element)

        elif isinstance(element, HTable):
            return self._htable(element)

        elif isinstance(element, VTable):
            return self._vtable(element)

        elif isinstance(element, Plot):
            return self._image(element)

    def __open(self):
        pass

    def __init(self):
        pass

    def _title(self, string):
        pass

    def _head(self, head):
        pass

    def _image(self, plot):
        pass

    def _vtable(self, table):
        pass

    def _htable(self, table):
        pass

    def _close(self):
        pass

    def _save(self, file):
        pass
