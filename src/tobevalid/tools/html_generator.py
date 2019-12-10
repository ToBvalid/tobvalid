# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:50:40 2019

@author: KavehB
"""

class HTMLReport:
    
       
    def open(self, path):
        self.dir = path
        self._closed = False
        self._html = '''<HTML>
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
    def head1(self, string):
        if self._closed:
            return self
        
        self._html = self._html + "<h1>" + string + "</h1>"
        return self
    
    def head2(self, string):
        if self._closed:
            return self
        self._html = self._html + "<h2>" + string + "</h2>"
        return self
    
    def image(self, pyplot, file):
        if self._closed:
            return self
        pyplot.savefig(self.dir + "/" + file)
        self._html = self._html + '<br><img src="'+ file +'"><br>'
        return self
    
    def vtable(self, columns, data):
        if self._closed:
            return self
        self._table_init()
        self._table_columns(columns)
             
        for row in data:
            self._html = self._html + '<TR height="20" class="ROW0">\n'
            for cell in row:
                 self._html = self._html + '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self._html = self._html +  '</TR>\n'
            
        self._table_close()
        
        
        return self
    def htable(self, columns, data):
        if self._closed:
            return self
        self._table_init()
        self._table_columns(columns)
             

       
        for key, row in data.items():
            self._html = self._html + '<TR height="20" class="ROW0">\n'
            self._html = self._html + '<TD NOWRAP="" class="HDR">' + key + '</TD>\n'
            for cell in row:
                 self._html = self._html + '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self._html = self._html +  '</TR>\n'
            
        self._table_close()
        return self
    
    def _table_columns(self, columns):
        if self._closed:
            return
        for column in columns:
             self._html = self._html + '<TD class="HDR">' + str(column) + '</TD>\n'
             
    def _table_init(self):
        if self._closed:
            return
        self._html = self._html + '''<TABLE>
                                        <COL />
                                            <COL span="28" style="text-align: center;" />
                                            <TBODY>'''
    def _table_close(self):
        if self._closed:
            return
        self._html = self._html + '''</TBODY>
                                        </TABLE>'''                                       
    def close(self):
        if self._closed:
            return self
        self._closed = True
        self._html = self._html +'''</BODY>
    
                         </HTML>
                      '''
        return self
    
    def save(self, file):
        if not self._closed:
            return self
        f = open(self.dir + "/" + file, "w")
        f.write(self._html)
        f.close()
        return self
    
    
    
    