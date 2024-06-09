####################
# PLACE TESTS HERE #
train = read_data("data/train.txt")
dev = read_data("data/dev.txt")
test = read_data("data/test.txt")
def test_read_data():
    result = {
        'lengths': (len(train), len(dev), len(test)),
    }
    return result

train_sequences = prepare_data(train, tag2id)
dev_sequences = prepare_data(dev, tag2id)
test_sequences = prepare_data(test, tag2id)

def test_prepare_data():
    result = {
        'texts_type': type(train_sequences["texts"]),
        'train_labels_shape': (train_sequences["labels"].shape),
    }
    return result

train_ds = NERDataset(train_sequences)
dev_ds = NERDataset(dev_sequences)
test_ds = NERDataset(test_sequences)

def test_model():
    # Create model
    model = load_model(model_name, tag2id)

    # Train model and evaluate
    trainer = train(best_model, N_EPOCHS, BATCH_SIZE, train_ds, dev_ds)

    results = evaluate(trainer, "Evaluation on Test Set", test_ds, tag2id)

    return {
        'f1': results['F1'],
        'f1_wo_o': results['F1_WO_O'],
    }

TESTS = [
    test_read_data,
    test_prepare_data,
    test_model,
]

# Run tests and save results
res = {}
for test in TESTS:
    try:
        cur_res = test()
        res.update({test.__name__: cur_res})
    except Exception as e:
        res.update({test.__name__: repr(e)})

with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)

# Download the results.json file
files.download('results.json')

####################
