"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from ._report import ReportGenerator, Report, Head, Text, Lines, VTable, HTable, Plot


class HTMLReport(ReportGenerator):

    def __init__(self, dpi=None):
        ReportGenerator.__init__(self, dpi)
        self._extension = ".html"

    def _open(self):
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

    def _title(self, string):
        self.__html = self.__html + "<h1>" + string + "</h1>"
        return self

    def _head(self, head):

        tag = min(head.depth() + 1, 6)
        self.__html = self.__html + "<h" + \
            str(tag) + ">" + head.head() + "</h" + str(tag) + ">"

        for child in head.children():
            self._write(child)
        return self

    def _image(self, plot, dpi=None):
        pyplot = plot.figure()
        plot.func()(pyplot, plot.title())
        file = plot.head() + self._extension + ".png"
        pyplot.savefig(self._dir + "/" + file, dpi=dpi)
        self.__html = self.__html + '<br><img src="' + file + '"><br>'
        return self

    def _vtable(self, table):
        columns = table.columns()
        data = table.data()
        self.__table_init()
        self.__table_columns(columns)

        for row in data:
            self.__html = self.__html + '<TR height="20" class="ROW0">\n'
            for cell in row:
                self.__html = self.__html + \
                    '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self.__html = self.__html + '</TR>\n'

        self.__table_close()
        return self

    def _htable(self, table):
        columns = table.columns()
        data = table.data()

        self.__table_init()
        self.__table_columns(columns)

        for key, row in data.items():
            self.__html = self.__html + '<TR height="20" class="ROW0">\n'
            self.__html = self.__html + '<TD NOWRAP="" class="HDR">' + key + '</TD>\n'
            for cell in row:
                self.__html = self.__html + \
                    '<TD NOWRAP="" class="DATASTR">' + str(cell) + '</TD>\n'
            self.__html = self.__html + '</TR>\n'

        self.__table_close()
        return self

    def _close(self):
        self.__html = self.__html + '''</BODY>
    
                         </HTML>
                      '''
        return self

    def _save(self, file):
        f = open(self._dir + "/" + file, "w")
        f.write(self.__html)
        f.close()
        return self

    def __table_columns(self, columns):
        for column in columns:
            self.__html = self.__html + \
                '<TD class="HDR">' + str(column) + '</TD>\n'

    def __table_init(self):
        self.__html = self.__html + '''<TABLE>
                                        <COL />
                                            <COL span="28" style="text-align: center;" />
                                            <TBODY>'''

    def __table_close(self):
        self.__html = self.__html + '''</TBODY>
                                        </TABLE>'''
