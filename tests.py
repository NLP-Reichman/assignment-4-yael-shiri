####################
# PLACE TESTS HERE #
train_raw = read_data("data/train.txt")
dev_raw = read_data("data/dev.txt")
test_raw = read_data("data/test.txt")
def test_read_data():
    result = {
        'lengths': (len(train_raw["texts"]), len(dev_raw["texts"]), len(test_raw["texts"])),
    }
    return result

train_sequences = prepare_data(train_raw, tag2id)
dev_sequences = prepare_data(dev_raw, tag2id)
test_sequences = prepare_data(test_raw, tag2id)

def test_prepare_data():
    result = {
        'dev_texts_shape': dev_sequences["texts"]["input_ids"].shape,
        'train_labels_shape': train_sequences["labels"].shape,
    }
    return result

train_ds = NERDataset(train_sequences)
dev_ds = NERDataset(dev_sequences)
test_ds = NERDataset(test_sequences)

N_EPOCHS = 5
def test_model():
    # Create model
    model = load_model(model_name, tag2id)

    # Train model and evaluate
    trainer = train_model(model, N_EPOCHS, BATCH_SIZE, train_ds, dev_ds)

    results_eval = evaluate(trainer, "Evaluation on Test Set", test_ds, tag2id)

    return {
        'f1': results_eval['F1'],
        'f1_wo_o': results_eval['F1_WO_O'],
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
