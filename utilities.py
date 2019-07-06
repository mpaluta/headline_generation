import csv

class data_setup():

    @staticmethod
    def get_paths(set="train"):
        with open(set+"_IDs.csv") as file_path:
            reader = csv.reader(file_path)
            paths = list(reader)
        if [] in paths:
            paths = paths[::2]
        return paths
