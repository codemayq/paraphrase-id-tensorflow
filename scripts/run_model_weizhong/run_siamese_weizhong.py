# coding=UTF-8
import argparse
import sys
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import json
import six

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from duplicate_questions.data.data_manager import DataManager
from duplicate_questions.data.embedding_manager import EmbeddingManager
from duplicate_questions.data.instances.weizhong_instance import WeizhongInstance
from duplicate_questions.models.siamese_bilstm.siamese_bilstm import SiameseBiLSTM
from duplicate_questions.data.data_indexer import DataIndexer

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def main():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # Parse config arguments
    argparser = argparse.ArgumentParser(
        description=("Run a baseline Siamese BiLSTM model "
                     "for paraphrase identification."))
    argparser.add_argument("mode", type=str,
                           choices=["train", "predict"],
                           help=("One of {train|predict}, to "
                                 "indicate what you want the model to do. "
                                 "If you pick \"predict\", then you must also "
                                 "supply the path to a pretrained model and "
                                 "DataIndexer to load."))
    argparser.add_argument("--model_load_dir", type=str,
                           default=os.path.join(project_dir,
                                                "models/"),
                           help=("The path to a directory with checkpoints to "
                                 "load for evaluation or prediction. The "
                                 "latest checkpoint will be loaded."))
    argparser.add_argument("--dataindexer_load_path", type=str,
                           default=os.path.join(project_dir,
                                                "models/"),
                           help=("The path to the dataindexer fit on the "
                                 "train data, so we can properly index the "
                                 "test data for evaluation or prediction."))
    argparser.add_argument("--train_file", type=str,
                           default=os.path.join(project_dir,
                                                "data/processed/weizhong/"
                                                "weizhong_train_train_split.txt"),
                           help="Path to a file to train on.")
    argparser.add_argument("--val_file", type=str,
                           default=os.path.join(project_dir,
                                                "data/processed/weizhong/"
                                                "weizhong_train_val_split.txt"),
                           help="Path to a file to monitor validation acc. on.")
    argparser.add_argument("--test_file", type=str,
                           default=os.path.join(project_dir,
                                                "data/raw/"
                                                "weizhong_dev.txt"))
    argparser.add_argument("--batch_size", type=int, default=512,
                           help="Number of instances per batch.")
    argparser.add_argument("--num_epochs", type=int, default=200,
                           help=("Number of epochs to perform in "
                                 "training."))
    argparser.add_argument("--early_stopping_patience", type=int, default=100,
                           help=("number of epochs with no validation "
                                 "accuracy improvement after which training "
                                 "will be stopped"))
    argparser.add_argument("--num_sentence_words", type=int, default=25,
                           help=("The maximum length of a sentence. Longer "
                                 "sentences will be truncated, and shorter "
                                 "ones will be padded."))
    argparser.add_argument("--word_embedding_dim", type=int, default=300,
                           help="Dimensionality of the word embedding layer")
    argparser.add_argument("--pretrained_embeddings_file_path", type=str,
                           help="Path to a file with pretrained embeddings.",
                           default=os.path.join(project_dir,
                                                "data/external/",
                                                "word2vec_finance_train_use.txt"))
    argparser.add_argument("--fine_tune_embeddings", action="store_true",
                           help=("Whether to train the embedding layer "
                                 "(if True), or keep it fixed (False)."))
    argparser.add_argument("--rnn_hidden_size", type=int, default=256,
                           help=("The output dimension of the RNN."))
    argparser.add_argument("--share_encoder_weights", action="store_true",
                           help=("Whether to use the same encoder on both "
                                 "input sentences (thus sharing weights), "
                                 "or a different one for each sentence"))
    argparser.add_argument("--rnn_output_mode", type=str, default="last",
                           choices=["mean_pool", "last"],
                           help=("How to calculate the final sentence "
                                 "representation from the RNN outputs. "
                                 "\"mean_pool\" indicates that the outputs "
                                 "will be averaged (with respect to padding), "
                                 "and \"last\" indicates that the last "
                                 "relevant output will be used as the "
                                 "sentence representation."))
    argparser.add_argument("--output_keep_prob", type=float, default=0.9,
                           help=("The proportion of RNN outputs to keep, "
                                 "where the rest are dropped out."))
    argparser.add_argument("--log_period", type=int, default=10,
                           help=("Number of steps between each summary "
                                 "op evaluation."))
    argparser.add_argument("--val_period", type=int, default=250,
                           help=("Number of steps between each evaluation of "
                                 "validation performance."))
    argparser.add_argument("--log_dir", type=str,
                           default=os.path.join(project_dir,
                                                "logs/"),
                           help=("Directory to save logs to."))
    argparser.add_argument("--save_period", type=int, default=250,
                           help=("Number of steps between each "
                                 "model checkpoint"))
    argparser.add_argument("--save_dir", type=str,
                           default=os.path.join(project_dir,
                                                "models/"),
                           help=("Directory to save model checkpoints to."))
    argparser.add_argument("--run_id", type=str, required=True,
                           help=("Identifying run ID for this run. If "
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--model_name", type=str, required=True,
                           help=("Identifying model name for this run. If"
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--reweight_predictions_for_kaggle", action="store_true",
                           help=("Only relevant when predicting. Whether to "
                                 "reweight the prediction probabilities to "
                                 "account for class proportion discrepancy "
                                 "between train and test."))

    config = argparser.parse_args()

    model_name = config.model_name
    run_id = config.run_id
    mode = config.mode

    # Get the data.
    batch_size = config.batch_size
    if mode == "train":
        # Read the train data from a file, and use it to index the validation data
        data_manager = DataManager(WeizhongInstance)
        num_sentence_words = config.num_sentence_words
        get_train_data_gen, train_data_size = data_manager.get_train_data_from_file(
            [config.train_file], min_count=2,
            max_lengths={"num_sentence_words": num_sentence_words})
        get_val_data_gen, val_data_size = data_manager.get_validation_data_from_file(
            [config.val_file], max_lengths={"num_sentence_words": num_sentence_words})
    else:
        # Load the fitted DataManager, and use it to index the test data
        logger.info("Loading pickled DataManager "
                    "from {}".format(config.dataindexer_load_path))

        config.dataindexer_load_path = os.path.join(config.dataindexer_load_path, config.model_name,
                                                    config.run_id.zfill(2), config.model_name + "-" + config.run_id.zfill(2) + "-" + "DataManager.pkl")

        data_manager = pickle.load(open(config.dataindexer_load_path, "rb"))
        test_data_gen, test_data_size = data_manager.get_test_data_from_file(
            [config.test_file])

    vars(config)["word_vocab_size"] = data_manager.data_indexer.get_vocab_size()

    # Log the run parameters.
    log_dir = config.log_dir
    log_path = os.path.join(log_dir, model_name, run_id.zfill(2))
    logger.info("Writing logs to {}".format(log_path))
    if not os.path.exists(log_path):
        logger.info("log path {} does not exist, "
                    "creating it".format(log_path))
        os.makedirs(log_path)
    params_path = os.path.join(log_path, mode + "params.json")
    logger.info("Writing params to {}".format(params_path))
    with open(params_path, 'w') as params_file:
        json.dump(vars(config), params_file, indent=4)

    # Get the embeddings.
    embedding_manager = EmbeddingManager(data_manager.data_indexer)
    embedding_matrix = embedding_manager.get_embedding_matrix(
        config.word_embedding_dim,
        config.pretrained_embeddings_file_path)
    vars(config)["word_embedding_matrix"] = embedding_matrix

    # Initialize the model.
    model = SiameseBiLSTM(vars(config))
    model.build_graph()

    if mode == "train":
        # Train the model.
        num_epochs = config.num_epochs
        num_train_steps_per_epoch = int(math.ceil(train_data_size / batch_size))
        num_val_steps = int(math.ceil(val_data_size / batch_size))
        log_period = config.log_period
        val_period = config.val_period

        save_period = config.save_period
        save_dir = os.path.join(config.save_dir, model_name, run_id.zfill(2) + "/")
        save_path = os.path.join(save_dir, model_name + "-" + run_id.zfill(2))

        logger.info("Checkpoints will be written to {}".format(save_dir))
        if not os.path.exists(save_dir):
            logger.info("save path {} does not exist, "
                        "creating it".format(save_dir))
            os.makedirs(save_dir)

        logger.info("Saving fitted DataManager to {}".format(save_dir))
        data_manager_pickle_name = "{}-{}-DataManager.pkl".format(model_name,
                                                                  run_id.zfill(2))

        pickle.dump(data_manager,
                    open(os.path.join(save_dir, data_manager_pickle_name), "wb"))

        patience = config.early_stopping_patience

        model.train(get_train_instance_generator=get_train_data_gen,
                    get_val_instance_generator=get_val_data_gen,
                    batch_size=batch_size,
                    num_train_steps_per_epoch=num_train_steps_per_epoch,
                    num_epochs=num_epochs,
                    num_val_steps=num_val_steps,
                    save_path=save_path,
                    log_path=log_path,
                    log_period=log_period,
                    val_period=val_period,
                    save_period=save_period,
                    patience=patience)
    else:
        # Predict with the model
        model_load_dir = os.path.join(config.model_load_dir, config.model_name, config.run_id.zfill(2))

        num_test_steps = int(math.ceil(test_data_size / batch_size))
        # Numpy array of shape (num_test_examples, 2)
        raw_predictions, lineids = model.predict(get_test_instance_generator=test_data_gen,
                                        model_load_dir=model_load_dir,
                                        batch_size=batch_size,
                                        num_test_steps=num_test_steps)
        # Remove the first column, so we're left with just the probabilities
        # that a question is a duplicate.
        # Write the predictions to an output submission file
        output_predictions_path = os.path.join(log_path, model_name + "-" +
                                               run_id.zfill(2) +
                                               "-output_predictions.csv")

        with open(output_predictions_path, "w") as output_file:
            output_file.write("test_id,result\n")
            for index, lineid in enumerate(lineids):
                output_file.write(str(lineid) + "," + str(raw_predictions[index])+ "\n")

        logger.info("Writing predictions to {}".format(output_predictions_path))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
