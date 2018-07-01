import os
import codecs


def check_label_ratio(train_file_name):
    train_file = codecs.open(train_file_name, encoding="UTF-8")
    count_pos = 0
    count_neg = 0
    for line in train_file:
        label = line.strip().split("\t")[-1]
        if label == "1":
            count_pos += 1
        else:
            count_neg += 1
    print(count_pos)
    print(count_neg)
    print(count_neg * 1.0 / (count_neg + count_pos))


if __name__ == '__main__':
    train_file_name = "../../data/raw/weizhong_train.txt"
    check_label_ratio(train_file_name)
