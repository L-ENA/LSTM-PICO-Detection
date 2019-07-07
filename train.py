from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse

parser = argparse.ArgumentParser()

def main():
    # create instance of config
    config = Config(parser)

    # build model
    model = HANNModel(config)
    model.build()
    ###############################################comment this if model is trained from scratch
    config.restore = True
    if config.restore:
        model.restore_session("/home/lena/Dokumente/Master/dissertation/Data/output/model.weights") # optional, restore weights
    model.reinitialize_weights("proj")#reinitialise for this scope
    #####################################################################

    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    test  = Dataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

    # evaluate model
    model.evaluate(test)

if __name__ == "__main__":
    main()
