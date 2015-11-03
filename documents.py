import pandas
from util import CsvOption, ProcessOption
import util
import os
import csv

class Documents(object):
    def __init__(self, dir_name, csv_option=CsvOption(), process_option=ProcessOption()):
        self.dir_name = dir_name
        self.csv_option = csv_option
        self.doc_num = self.get_num_doc()
        self.process_option = process_option

    def __iter__(self):
        for document in self.field_iterator(csv_option.review_name):
            yield util.process(document, process_option)

    def field_iterator(self, field):
        for file_name in os.listdir(self.dir_name):
            if file_name.endswith(".csv"):
                documents = pandas.read_csv(os.path.join(self.dir_name, file_name), \
                    delimiter=self.csv_option.deli, names=self.csv_option.title, \
                    iterator=True, chunksize=self.csv_option.chunksize)
                for chunk in documents:
                    for sentiment in chunk[field]:
                        yield sentiment

    def get_num_doc(self):
        num = 0
        for file_name in os.listdir(self.dir_name):
            if file_name.endswith(".csv"):
                num += self.row_count(os.path.join(self.dir_name, file_name))
        return num

    def row_count(self, file_name):
        count = 0
        with open(file_name) as csv_file:
            count += sum(1 for row in csv.reader(csv_file))
        return count