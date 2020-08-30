"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from typing import Dict, List
from .report import ReportGenerator
import panel as pn
from bokeh.resources import INLINE

css = '''#  body{
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
                }'''

pn.extension(raw_css=[css])


class HTMLReport(ReportGenerator):

    def __init__(self, dpi=None):
        ReportGenerator.__init__(self, dpi)
        self._extension = ".html"

    def save_reports(self, reports, path, name):
        
        panel = pn.Tabs()
        self._dir = path  
        if isinstance(reports, List):
            self._save_report_list(reports, panel)

        if isinstance(reports, Dict):
            for key in reports:
                pnl = pn.Tabs()
                self._save_report_list(reports[key], pnl)
                panel.append((key, pnl))

                 
        panel.save(self._dir + "/" + name + self._extension, resources=INLINE, title="{} Report".format(name), embed=True)
    
    def _save_report_list(self, reports, panel):
         for report in reports:
            html = HTMLReport(dpi=self._dpi)
            html._prepare(report, self._dir)
            panel.append((report.title(), html.__panel))

    def _open(self):
        self.__panel = pn.Column(sizing_mode='stretch_width')

    def _title(self, string):
        self.__panel.append("# " + string)
        return self

    def _head(self, head):
        self.__panel.append("".join(["#"]*head.depth()) + " " + head.head())
        for child in head.children():
            self._write(child)
        return self

    def _image(self, plot, dpi=None):
        
        pyplot = plot.figure()
        plot.func()(pyplot, plot.title())
        file = plot.head() + self._extension + ".png"
        pyplot.savefig(self._dir + "/" + file, dpi=dpi)
        pyplot.clf()
        pyplot.close()
        self.__panel.append(pn.pane.HTML('<br><img src="' + file + ' " width="600" height="400"><br>'))
        return self

    def _vtable(self, table):
        self.__html = ""
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
        self.__panel.append(pn.pane.HTML(self.__html, style={
                            'HDR': {'background-color': '#AACCFF'}}))
        return self

    def _htable(self, table):
        self.__html = ""
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
        self.__panel.append(pn.pane.HTML(self.__html, style={
                            'HDR': {'background-color': '#AACCFF'}}))
        return self

    def _text(self, text):
        self.__panel.append(pn.pane.HTML(text.text(), style={"padding-left":"{}px".format(30*text.indent()), "text-align": "left"}))
        return self

    def _texts(self, texts):
        res = ""
        for text in texts.texts():
            res = res + '<div style="text-indent:{}px;text-align: left">'.format(30*text.indent()) + text.text() + '</div>'
        self.__panel.append(pn.pane.HTML(res))
        return self    

    def _close(self):
        return self

    def _save(self, file):
        self.__panel.save(self._dir + "/" + file, resources=INLINE, title="Report")
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
