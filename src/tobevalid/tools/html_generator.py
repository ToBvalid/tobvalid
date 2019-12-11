# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:50:40 2019

@author: KavehB
"""

from .report import Report, HTable, VTable, Text, Lines, Plot, Head

class HTMLReport:
    
    def save(self, report, path, name):
        self.__open(path)
        for element in report.items():
            if isinstance(element, Head):
                if element.one():
                    self.__head1(element.head())
                else:
                    self.__head2(element.head())
                    
            elif isinstance(element, HTable):
                self.__htable(element.columns(), element.data())
          
            elif isinstance(element, VTable):
                    self.__vtable(element.columns(), element.data())
                    
            elif isinstance(element, Plot):
                self.__image(element.figure(), element.head() + ".png")
                
        self.__close().__save(path + "/" + name + ".html")
       
    def __open(self, path):
        self.__dir = path
        self.__closed = False
        self.__html = '''<HTML>
                            <HEAD>
                                <meta HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=windows-1255" />
                            </HEAD>
                            
                            <BODY>
                            
                            <style>
                                body {
                                    background-color: #FFFFFF;
                                }
                        
                                .HDR {
                                    background-color: #AACCFF;
                                    text-align: center;
                                    border: 1px solid black;
                                }
                        
                                .ROW0 {
                                    background-color: #F0FFFF;
                                }
                        
                                .ROW1 {
                                    background-color: #F0FAFF;
                                }
                        
                                td {
                                    padding-left: 5px;
                                    padding-right: 5px;
                                    border-bottom: 1px solid #CCCCCC;
                                }
                        
                                table {
                                    border-collapse: collapse;
                                }
                            </style>
                      '''                      
    def __head1(self, string):
        if self.__closed:
            return self
        
        self.__html = self.__html + "<h1>" + string + "</h1>"
        return self
    
    def __head2(self, string):
        if self.__closed:
            return self
        self.__html = self.__html + "<h2>" + string + "</h2>"
        return self
    
    def __image(self, pyplot, file):
        if self.__closed:
            return self
        pyplot.savefig(self.__dir + "/" + file)
        self.__html = self.__html + '<br><img src="'+ file +'"><br>'
        return self
    
    def __vtable(self, columns, data):
        if self.__closed:
            return self
        self.__table_init()
        self.__table_columns(columns)
             
        for row in data:
            self.__html = self.__html + '<TR height="20" class="ROW0">\n'
            for cell in row:
                 self.__html = self.__html + '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self.__html = self.__html +  '</TR>\n'
            
        self.__table_close()
        return self
        
        
        return self
    def __htable(self, columns, data):
        if self.__closed:
            return self
        self.__table_init()
        self.__table_columns(columns)
             

       
        for key, row in data.items():
            self.__html = self.__html + '<TR height="20" class="ROW0">\n'
            self.__html = self.__html + '<TD NOWRAP="" class="HDR">' + key + '</TD>\n'
            for cell in row:
                 self.__html = self.__html + '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self.__html = self.__html +  '</TR>\n'
            
        self.__table_close()
        return self
    
    def __table_columns(self, columns):
        if self.__closed:
            return
        for column in columns:
             self.__html = self.__html + '<TD class="HDR">' + str(column) + '</TD>\n'
             
    def __table_init(self):
        if self.__closed:
            return
        self.__html = self.__html + '''<TABLE>
                                        <COL />
                                            <COL span="28" style="text-align: center;" />
                                            <TBODY>'''
    def __table_close(self):
        if self.__closed:
            return
        self.__html = self.__html + '''</TBODY>
                                        </TABLE>'''                                       
    def __close(self):
        if self.__closed:
            return self
        self.__closed = True
        self.__html = self.__html +'''</BODY>
    
                         </HTML>
                      '''
        return self
    
    def __save(self, file):
        if not self.__closed:
            return self
        f = open(self.__dir + "/" + file, "w")
        f.write(self.__html)
        f.close()
        return self
    
    
    
    