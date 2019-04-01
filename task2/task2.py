# imports, globals etc.
import argparse, csv, functools, math, numpy, pprint, sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

DEFAULT_PATH = "./test.classifier.txt"
ATTR_INDEX_TOTAL_ROWS_KEY = "total_rows"

# read from csv file and return list of json objects
# table_format=True  - return array of arrays e.g. first array is attr headers, the rest are records
# table_format=False - return list of json objs, key val pairs for each attr
def read_csv(filepath, table_format=True):
    with open(filepath, "r",) as csvfile:
        rdr = [row for row in csv.reader(csvfile)]
        if table_format:
            return rdr
        else:
            records = []
            attrs = rdr[0]
            for row in rdr[1:]:
                i = 0
                record = dict()
                while i < len(attrs):
                    record[attrs[i]] = row[i]
                    i += 1
                records.append(record)
            return records

def read_classifier(classifier_filepath, input_row):
    return None

def entropy(attr_index: dict):
    total_rows = attr_index[ATTR_INDEX_TOTAL_ROWS_KEY]
    # e.g. - (P(a) x log2(P(a))) - (P(b) x log2(P(b))) - (P(c) x log2(P(c))) - (P(d) x log2(P(d))) etc...
    return functools.reduce((lambda x, y: x + y), map(lambda x: (x[0], ((x[1] / total_rows) * math.log2(x[1] / total_rows))) * -1, attr_index.items()))

def build_attribute_index(rows, attribute):
    attr_index = dict()
    for row in rows:
        if row[attribute] not in attr_index.keys():
            attr_index[row[attribute]] = 0
        attr_index[row[attribute]] += 1
    attr_index[ATTR_INDEX_TOTAL_ROWS_KEY] = len(rows)
    return attr_index

def generate_classifier(csv_filepath, classifier_out, attr_to_predict):
    table = read_csv(csv_filepath)
    #pprint.pprint(table)
    #general_attribute_index = build_attribute_index(table, attr_to_predict)
    #general_entropy = entropy(general_attribute_index)

    attributes = table[0]
    dataset = pd.read_csv(csv_filepath, header=None, names=attributes)
    features = [attribute for attribute in attributes if attribute not in [attr_to_predict]]

    return None

def use_classifier(classifier, attributes, attr_to_predict):
    return None

def parse_cli():
    parser = argparse.ArgumentParser(description="Generate a classifier from a csv, or use a pre-existing classifier to predict a certain value.")

    # add args
    parser.add_argument("action", metavar="generate | predict", action="store", type=str, help="generate a classifier or predict from a list of attributes using --classifier")
    parser.add_argument("--classifier-in", action="store", type=str, required=False, help="The path to load a classifier from.")
    parser.add_argument("--classifier-out", action="store", type=str, required=False, help="The path to save a new classifier to.")
    parser.add_argument("--csv", action="store", type=str, required=False, help="The path of a CSV file to train a new classifier.")
    parser.add_argument("--input-row", action="store", type=str, required=False, help="The path of a CSV file to run on a classifier.")
    parser.add_argument("--attribute", action="store", type=str, required=True, help="The attribute to predict using the classifier.")

    args = parser.parse_args()

    # error checking
    err = False

    if args.action != "generate" and args.action != "predict":
        print("Unknown action '{0}'".format(args.action))
        err = True

    elif args.attribute is None or args.attribute == "":
        print("No attribute to predict!")
        err = True

    elif args.action == "generate":
        if args.csv is None or args.csv == "":
            print("No CSV file provided!")
            err = True
        if args.classifier_out is None or args.classifier_out == "":
            args.classifier_out = DEFAULT_PATH
            print("WARNING: No output file path provided. Using {0}".format(DEFAULT_PATH))

    elif args.action == "predict":
        if args.input_row is None or args.input_row == "":
            print("No test file provided!")
            err = True
        if args.classifier_in is None or args.classifier_in == "":
            print("No classifier specified!")
            err = True

    # return error code 1 + help text
    if err:
        parser.print_help()
        sys.exit(1)
    else:
        # calls generate or use classifier based on 'action' from CLI
        generate_classifier(args.csv, args.classifier_out, args.attribute) if args.action == "generate" else use_classifier(args.classifier_in, args.input_row, args.attribute)


if __name__ == "__main__":
    parse_cli()

