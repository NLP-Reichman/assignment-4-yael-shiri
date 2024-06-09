'''
Main script for testing the assignment.
Runs the tests on the results json file.
'''

import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description='Language Modeling')
    parser.add_argument('test', type=str, help='The test to perform.')
    return parser.parse_args()

def test_read_data(results):
    lengths = tuple(results["lengths"])

    if not lengths == (1750, 250, 500):
        return f"Lengths are {lengths}, expected (1750, 250, 500)"
    return 1

def test_prepare_data(results):
    if not results["texts_type"] == dict:
        return f"Texts type is {results['texts_type']}, expected dict"
    if not tuple(results["train_labels_shape"]) == (1750, 77):
        return f"Train Labels shape is {results['labels_shape']}, expected (1750, 77)"
    return 1

def test_model(results):
    f1 = results["f1"]
    f1_wo_o = results["f1_wo_o"]

    # Min value to pass
    if f1 < 0.93:
        return f"F1 is {f1}, expected at least 0.80"
    if f1_wo_o < 0.90:
        return f"F1 without O is {f1_wo_o}, expected at least 0.60"
    
    # Values to partially pass
    if f1 < 0.97:
        return 2
    if f1_wo_o < 0.93:
        return 2

    # Pass with full marks
    return 1

def main():
    # Get command line arguments
    args = get_args()

    # Read results.json
    with open('results.json', 'r') as f:
        results = json.load(f)

    # Initialize the result variable
    result = None

    # Switch between the tests
    match args.test:
        case 'test_read_data':
            result = test_read_data(results["test_read_data"])
        case 'test_prepare_data':
            result = test_prepare_data(results["test_prepare_data"])
        case 'test_model':
            result = test_model(results["test_model"])
        case _:
            print('Invalid test.')

    # Print the result for the autograder to capture
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
