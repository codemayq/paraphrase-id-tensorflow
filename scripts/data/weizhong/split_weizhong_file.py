import argparse
import csv
import logging
import os
import random
from io import open
import six

import sys
import codecs

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Split a file from the Kaggle Quora dataset "
                     "into train and validation files, given a validation "
                     "proportion"))
    argparser.add_argument("validation_proportion", type=float,
                           help=("Proportion of data in the input file "
                                 "to randomly split into a separate "
                                 "validation file."))
    argparser.add_argument("dataset_input_path", type=str,
                           help=("The path to the cleaned Quora "
                                 "dataset file to split."))
    argparser.add_argument("dataset_output_path", type=str,
                           help=("The *folder* to write the "
                                 "split files to. The name will just have "
                                 "_{split}_split appended to it, before "
                                 "the extension"))
    config = argparser.parse_args()

    # Get the data
    logger.info("Reading csv at {}".format(config.dataset_input_path))
    csv_rows = []
    with codecs.open(config.dataset_input_path, encoding="UTF-8") as f:
        for line in f:
            csv_rows.append(line)

    logger.info("Shuffling input csv.")
    # For reproducibility
    random.seed(0)
    # Shuffle csv_rows deterministically in place
    random.shuffle(csv_rows)

    num_validation_lines = int(len(csv_rows) * config.validation_proportion)

    input_filename_full = os.path.basename(config.dataset_input_path)
    input_filename, input_ext = os.path.splitext(input_filename_full)
    train_out_path = os.path.join(config.dataset_output_path,
                                  input_filename + "_train_split" + input_ext)
    val_out_path = os.path.join(config.dataset_output_path,
                                input_filename + "_val_split" + input_ext)

    logger.info("Writing train split output to {}".format(train_out_path))
    with codecs.open(train_out_path, "w", encoding="utf-8") as f:
        for line in csv_rows[num_validation_lines:]:
            f.write(line)

    logger.info("Writing validation split output to {}".format(val_out_path))
    with codecs.open(val_out_path, "w") as f:
        for line in csv_rows[:num_validation_lines]:
            f.write(line)

if __name__ == "__main__":
    logging.basicConfig(format=("%(asctime)s - %(levelname)s - "
                                "%(name)s - %(message)s"),
                        level=logging.INFO)
    main()
