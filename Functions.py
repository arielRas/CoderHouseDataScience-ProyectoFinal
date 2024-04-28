import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

class Metrics():
    def __init__(self) -> None:
        pass

    def plot_matrix_confusion(self, y_true, y_pred):
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, fmt='.0f', annot=True, cmap='inferno')
        plt.title('Matriz de confusion')
        plt.xlabel('Predict data')
        #plt.xticks(ticks=[0.5,1.5], labels=[label_0, label_1])
        plt.ylabel('True data')
        #plt.yticks(ticks=[0.5,1.5], labels=[label_0, label_1])
        plt.show()

    def plot_roc_curve(y_true , y_prob):
        mpl.style.use('ggplot')
        false_positive_rate, true_positive_rate, _threshold = roc_curve(y_true, y_prob)
        sns.lineplot(x=false_positive_rate, y=true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.title('Curva ROC')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.tight_layout()
        plt.show()
        mpl.style.use('default')
