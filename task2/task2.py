# imports, globals etc.
import argparse, csv, functools, math, numpy, pickle, pprint, sys
# import ptvsd # for debugging in vs code
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing

DEFAULT_SAVE_PATH = "./test.classifier"

def read_classifier(classifier_filepath, input_row):
    return None

"""
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
"""

def read_csv(filepath):
    rows = []
    with open(filepath, "r") as csvfile:
        rdr = csv.reader(csvfile, delimiter=",")
        for row in rdr:
            found_nan = False
            for item in row:
                # prevent null values from getting through - e.g. trials.csv cell J644...
                if type(item) != str and math.isnan(item):
                    found_nan = True
            if found_nan is False:
                rows.append(row)
    return rows

def read_csv_headers(filepath):
    with open(filepath, 'r') as csvfile:
        return csv.DictReader(csvfile).fieldnames

def _data_of_types(data, types=[]):
    for item in data:
        if type(item) not in types:
            return False
    return True

def _data_is_numeric(data):
    return _data_of_types(data, [int, float])

def get_a_b_split(csv_filepath, attr_to_predict):
    raw = read_csv(csv_filepath)
    attributes = raw[0]
    dataset = pd.DataFrame(columns=attributes, data=raw[1:])
    features = numpy.array([attribute for attribute in attributes if attribute != attr_to_predict])
    
    # label encode values if there are mixed data types
    for attribute in features:
        if _data_is_numeric(dataset[attribute]) is False:
            le = preprocessing.LabelEncoder()
            le.fit(dataset[attribute])
            dataset[attribute] = le.transform(dataset[attribute])

    return dataset[features], dataset[attr_to_predict]

# this function was taken from a Gist post on Github
# author:   rola93
# url:      https://gist.github.com/zachguo/10296432#gistcomment-2638135-permalink
# created:  04/07/2018
# accessed: 04/04/2019
def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def generate_classifier(csv_filepath, classifier_out, attr_to_predict):
    attr_split, b_split = get_a_b_split(csv_filepath, attr_to_predict)

    # generate data splits for training and testing
    # a_train -> training data using attribute rows
    # a_test  -> test data using attribute rows
    # b_train -> training data using attribute to predict
    # b_test  -> test data using attribute to predict
    a_train, a_test, b_train, b_test = train_test_split(attr_split, b_split, test_size=0.3, random_state=1)

    # train the classifier
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=0.3, min_samples_leaf=0.4, max_features=3)
    decision_tree.fit(a_train, b_train)

    prediction = decision_tree.predict(a_test)

    print_confusion_matrix(metrics.confusion_matrix(b_test, prediction), ["true", "false"])
    print("Accuracy: {0:.2f}".format(metrics.accuracy_score(b_test, prediction) * 100))

    # write resources to file for later study
    a_train.to_csv("{0}_a_train.csv".format(classifier_out))
    b_train.to_csv("{0}_b_train.csv".format(classifier_out))
    a_test.to_csv("{0}_a_test.csv".format(classifier_out))
    b_test.to_csv("{0}_b_test.csv".format(classifier_out))

    # write to file using pickle library
    with open(classifier_out + ".pkl", 'wb') as classifier_file:
        pickle.dump(decision_tree, classifier_file)

    sys.exit(0)

def use_classifier(classifier_file, attributes_file, attr_to_predict):
    loaded_classifier = DecisionTreeClassifier()
    with open(classifier_file, 'rb') as clff:
        loaded_classifier = pickle.load(clff)

    attr_split, b_split = get_a_b_split(attributes_file, attr_to_predict)

    # generate data splits for training and testing
    # a_train -> training data using attribute rows
    # a_test  -> test data using attribute rows
    # b_train -> training data using attribute to predict
    # b_test  -> test data using attribute to predict
    a_train, a_test, b_train, b_test = train_test_split(attr_split, b_split, test_size=0, random_state=1)

    prediction = loaded_classifier.predict(attr_split)

    print(prediction)

    print_confusion_matrix(metrics.confusion_matrix(b_split, prediction), ["true", "false"])
    print("Accuracy: {0:.2f}".format(metrics.accuracy_score(b_test, prediction) * 100))

    sys.exit(0)

def parse_cli():
    parser = argparse.ArgumentParser(description="Generate a classifier from a csv, or use a pre-existing classifier to predict a certain value.")

    # add args
    parser.add_argument("action", metavar="generate | predict", action="store", type=str, help="generate a classifier or predict from a list of attributes using --classifier")
    parser.add_argument("--classifier-in", action="store", type=str, required=False, help="The path to load a classifier from.")
    parser.add_argument("--classifier-out", action="store", type=str, required=False, help="The path to save a new classifier to (not including file extension).")
    parser.add_argument("--csv", action="store", type=str, required=False, help="The path of a CSV file to train a new classifier.")
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
            args.classifier_out = DEFAULT_SAVE_PATH
            print("WARNING: No output file path provided. Using {0}".format(DEFAULT_SAVE_PATH))

    elif args.action == "predict":
        if args.csv is None or args.csv == "":
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
        generate_classifier(args.csv, args.classifier_out, args.attribute) if args.action == "generate" else use_classifier(args.classifier_in, args.csv, args.attribute)


if __name__ == "__main__":
    #address = ("localhost", 5678)
    #ptvsd.enable_attach(address=address)
    #ptvsd.wait_for_attach()
    parse_cli()

