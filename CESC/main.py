from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import classification_report
import logging
import csv
import numpy as np

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Define metric
def clf_report(labels, preds):
    return classification_report(labels, preds, output_dict=True)


# evaluate on test set
test_df = pd.read_csv('test.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
test_df.columns = ['claim_labels', 'text_a', 'text_b', 'id', 'labels']
test_df = test_df[['text_a', 'text_b', 'labels']]
model = ClassificationModel('bert', 'outputs/best_model/')
result, model_outputs, wrong_predictions = model.eval_model(test_df, clf_report=clf_report)

preds = list(np.argmax(model_outputs, axis=-1))
label_map = {0: 0, 1: 1, 2: -1}
preds = [label_map[x] for x in preds]

with open('outputs/claims_stance_result.txt', 'w') as f:
	for x in preds:
	    f.write(x+'\n')






