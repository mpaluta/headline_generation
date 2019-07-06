import csv

class data_setup():

    # @staticmethod
    # def load_one_article(set="train", offset=0):
    #     with open(set+"_IDs.csv", newline='') as f:
    #         reader = csv.reader(f)
    #         row1 = next(reader)[offset]
    #
    #     with open(os.path.join('data', row1), encoding='utf-8') as f:
    #         article = NYTArticle.from_file(f)
    #
    #     return article

    @staticmethod
    def get_paths(set="train"):
        with open(set+"_IDs.csv") as file_path:
            reader = csv.reader(file_path)
            paths = list(reader)

        return paths
