# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:00:08 2019

@author: KavehB
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:50:40 2019

@author: KavehB
"""

from ._report import ReportGenerator
import json

class JSONReport(ReportGenerator):
    
    def __init__(self):
        ReportGenerator.__init__(self)
        self._extension = ".json"
    
    def _open(self):
        self.__json = dict()

    def _title(self, string):
        self.__json["title"] = string
        return self
                  
    def _head(self, head):
        children = []        
        for child in head.children():
            children.append(self._write(child))
        
        if head.parent() == None:
            self.__json[head.head()] = children
        return {head.head():children}

    def _image(self, plot):
        pyplot = plot.figure()
        file = plot.head() + self._extension +  ".png"
        pyplot.savefig(self._dir + "/" + file)
        return [self._dir + "/" + file]

    def _vtable(self, table):
        
        columns = table.columns()
        data = table.data()
        
  
        result = []     
        for row in data:
            d = dict()
            for i in range(min(len(row), len(columns))):
                d[columns[i]] = row[i]
            result.append(d)                    
            
        return result
        
    def _htable(self, table):                
        return table.data()  
                                          
    def _close(self):
        return self
    
    def _save(self, file):
        f = open(self._dir + "/" + file, "w")
        f.write(json.dumps(self.__json))
        f.close()
        return self 
    
    
    
    
    