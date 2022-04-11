from sklearn.metrics import classification_report
# from bcutils4r.eval_model import eval_model # A simple wrapper around scikit-learn and scikit-plot fucntions for evaluating binary classification


def evaluate_model(true_labels=[], pred_labels=[], class_names=[]):
    print(classification_report(true_labels, pred_labels, target_names=class_names))