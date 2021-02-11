import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt

from src.utils.logging import get_logger


def plot_lift_curve(y_true: Iterable, y_pred: Iterable, step: float = 0.01) -> None:
    """
        Function that plots a Lift Curve using the real label values of a dataset and
    the probability predictions of a Machine Learning Algorithm/model.
    Args:
        y_true {Iterable}: real labels of the data
        y_pred {Iterable}: probability predictions for such data
        step {float}: how big we want the steps in the percentiles to be
    Reference: https://towardsdatascience.com/the-lift-curve-unveiled-998851147871
    """

    # Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    # Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_true
    aux_lift['predicted'] = y_pred
    # Order the values for the predicted probability column:
    aux_lift.sort_values('predicted', ascending=False, inplace=True)

    # Create the values that will go into the X axis of our plot
    x_val = np.arange(step, 1 + step, step)
    # Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    # Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []

    # Calculate for each x value its correspondent y value
    for x in x_val:
        # The ceil function returns the closest integer bigger than our number
        num_data = int(np.ceil(x * len(aux_lift)))
        data_here = aux_lift.iloc[:num_data, :]  # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)

    # Plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40, 40)
    axis.plot(x_val, y_v, 'g-', linewidth=3, markersize=5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title('Lift Curve')
    plt.show()


def precision_at_k_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """
    Precision@k metric.
    Args:
        actual {Iterable}: actual labels of the data
        predicted {Iterable}: predicted labels of the data
        predicted_probas {Iterable}: probability predictions for such data
        k {int}: top k value on which score will be calculated
    Returns:
        float: Precision@k value
    """

    df = pd.DataFrame({'actual': actual, 'predicted': predicted, 'probas': predicted_probas})
    df = df.sort_values(by=['probas'], ascending=False).reset_index(drop=True)
    df = df[:k]

    return precision_score(df['actual'], df['predicted'])


def recall_at_k_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """
    Recall@k metric.
    Args:
        actual {Iterable}: actual labels of the data
        predicted {Iterable}: predicted labels of the data
        predicted_probas {Iterable}: probability predictions for such data
        k {int}: top k value on which score will be calculated
    Returns:
       float: Recall@k value
    """

    df = pd.DataFrame({'actual': actual, 'predicted': predicted, 'probas': predicted_probas})
    df = df.sort_values(by=['probas'], ascending=False).reset_index(drop=True)
    df = df[:k]

    return recall_score(df['actual'], df['predicted'])


def lift_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """
    Lift@k metric.
    Args:
        actual {Iterable}: actual labels of the data
        predicted {Iterable}: predicted labels of the data
        predicted_probas {Iterable}: probability predictions for such data
        k {int}: top k value on which score will be calculated
    Returns:
        float: Lift@k value
    """

    logger = get_logger(__name__, logging.INFO)

    numerator = recall_at_k_score(actual, predicted, predicted_probas, k)
    denominator = np.mean(actual)

    lift = numerator / denominator
    logger.info(f'Lift: {numerator} / {denominator} = {lift}')

    return lift

