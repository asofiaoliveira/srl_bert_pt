import logging
import os
import random
from overrides import overrides
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
import sys
from my_reader import SimpleReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class createFolds():
    """
    This class creates the folds for a specific data set.

    # Parameters

    file_path : `str`
        Path to data set.
    k : `int`
        Number of folds to create
    folds_dir : `str`
        Where to store the created folds.
    seed : `int`, (default = 13)
        The seed for the random shuffling of the data.
    remove_c : `bool`, (default = False)
        Wether to remove instances that contain continuation roles from the data.
    """
    def __init__(self, file_path: str, k: int, folds_dir: str, seed = 13, remove_c: bool = False) -> None:
        self.file_path = file_path
        self.k = k
        self.folds_dir = folds_dir
        self.seed = seed
        self.remove_c = remove_c

    def read_data(self):
        dataset_reader = SimpleReader(remove_c = self.remove_c)
        logger.info("Reading complete data set from %s", self.file_path)
        self.data = dataset_reader.read(self.file_path).instances

    def get_data(self):
        return self.data
    
    def get_folds(self):
        return self.folds
        
    def create_folds_dfalci(self):
        self.read_data()
        random.seed(self.seed)
        random.shuffle(self.data, random.random)
        length = len(self.data)
        self.folds = []
        for i in range(self.k-1):
            self.folds.append(self.data[int(i*length/self.k) : int((i+1)*length/self.k)])
        self.folds.append(self.data[int((self.k-1)*length/self.k):])
    
    def _util(self, data):
        y = data["tags"]
        a = pd.DataFrame({"tags":[list(filter(lambda a: a not in ["O"] and a[0]!="I", x)) for x in y]})
        a = pd.get_dummies(a.tags.apply(pd.Series).stack()).sum(level=0)
        a.pop("B-V")
        return data, a

    def create_folds(self):
        self.read_data()
        self.data = pd.DataFrame(self.data, columns=["words","tags"])
        data_all, a = self._util(self.data)
        self.folds=[]
        self.val = []
        self.train = []
        mskf = MultilabelStratifiedKFold(n_splits=10, random_state=13, shuffle = True)
        for train, test in mskf.split(data_all, a):
            self.folds.append(self.data.iloc[test])
            mskf2 = MultilabelStratifiedKFold(n_splits=9)
            a1,a2 = self._util(self.data.iloc[train])
            t, val = next(mskf2.split(a1,a2))
            self.val.append(self.data.iloc[train].iloc[val])
            self.train.append(self.data.iloc[train].iloc[t])


    def write_folds(self):
        self.create_folds()
        if not os.path.exists(self.folds_dir):
            os.mkdir(self.folds_dir)
        for fold_ind, fold in enumerate(self.folds):
            train = open(self.folds_dir + "/train" + str(fold_ind) + ".txt", "w", encoding="UTF-8")
            val = open(self.folds_dir + "/val" + str(fold_ind) + ".txt", "w", encoding="UTF-8")
            test = open(self.folds_dir + "/test" + str(fold_ind) + ".txt", "w", encoding="UTF-8")

            test.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in fold.values.tolist())
            train.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in self.train[fold_ind].values.tolist())
            val.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in self.val[fold_ind].values.tolist())
            
            print(len(self.train[fold_ind]), len(self.val[fold_ind]), len(fold))


    def write_folds_dfalci(self, test: bool = True):
        self.create_folds_dfalci()
        if not os.path.exists(self.folds_dir):
            os.mkdir(self.folds_dir)
        for fold_ind, fold in enumerate(self.folds):
            train = open(self.folds_dir + "/train" + str(fold_ind) + ".txt", "w", encoding="UTF-8")
            val = open(self.folds_dir + "/val" + str(fold_ind) + ".txt", "w", encoding="UTF-8")
            if test:
                test = open(self.folds_dir + "/test" + str(fold_ind) + ".txt", "w", encoding="UTF-8")

            train_data = self.folds[:fold_ind] + self.folds[fold_ind+1:]
            train_data = [sentence for folds in train_data for sentence in folds]

            if test:
                indices = [indi for indi in range(len(train_data))]
                random.shuffle(indices)
                validation_data = [train_data[indi] for indi in indices[:len(fold)]]
                train_data = [train_data[indi] for indi in indices[len(fold):]]
                test.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in fold)

            else:
                validation_data = fold
                
            train.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in train_data)
            val.writelines("%s\t%s\n" % (' '.join(tokens), ' '.join(tags)) for (tokens, tags) in validation_data)
            print(len(train_data), len(validation_data), len(fold))


if __name__ == "__main__":
    createFolds("./data/conll_data/PropBankBr_v1.1_Const.conll.txt", k = 20, folds_dir = "data/folds_dfalci_20", seed = 27, remove_c = True).write_folds_dfalci(test = False)

    createFolds("./data/conll_data/train.conll", k = 10, folds_dir = "data/folds_10").write_folds()
    createFolds("./data/conll_data/buscape.conll", k = 1, folds_dir = "data/buscape").write_folds_dfalci()
