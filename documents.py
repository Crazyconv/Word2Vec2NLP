import pandas
import os
import csv

import util
import setting

class Documents(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.doc_num = self.get_num_doc()

    def __iter__(self):
        for document in self.field_iterator(setting.csv_option.review_name):
            yield util.process(document)

    def field_iterator(self, field):
        for file_name in os.listdir(self.dir_name):
            documents = pandas.read_csv(os.path.join(self.dir_name, file_name), \
                delimiter=setting.csv_option.deli, names=setting.csv_option.title, \
                iterator=True, chunksize=setting.csv_option.chunksize)
            for chunk in documents:
                for content in chunk[field]:
                    yield content

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