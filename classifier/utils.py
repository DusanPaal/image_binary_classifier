import numpy as np
from keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.data import Dataset
from keras import layers
import itertools
import os

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, roc_auc_score,
    f1_score
)

def read_image(img_path: str, raw=False) -> np.ndarray:
    '''Reads an image from the specified 
    file path and processes it.
    
    Args:
        img_path (str): The file path to the image.
        raw (bool): If True, returns the raw image without processing. Default is False.

    Returns:
        np.ndarray: The processed image as a NumPy array. If raw is True, returns the raw image.
    '''
    
    img = image.load_img(img_path, target_size=(150, 150))

    if raw:
        return img
        
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    
    return img_tensor

def plot_confusion_matrix(
    y_true, y_pred, class_labels, normalize=False,
    flip=False, ax=None, title=None, **kwargs):
    '''Create a confusion matrix heatmap 
    to evaluate classification.

    Parameters:
    -----------
    y_true : array-like
        True labels of the data.
    y_pred : array-like
        Predicted labels of the data.
    class_labels : list
        Labels for the classes in the data.
    normalize : bool, default=False
        If True, normalize the confusion matrix.
    flip : bool, default=False
        If True, flip the class labels.
    ax : matplotlib Axes, default=None
        The axes to plot the heatmap.
    title : str, default=None
        The title of the plot.
    **kwargs : dict
        Additional keyword arguments to 
        pass to `sns.heatmap`.
    '''
    
    mat = confusion_matrix(y_true, y_pred)
    
    if normalize:
        fmt, mat = '.2%', mat / mat.sum()
    else:
        fmt = 'd'

    if flip:
        class_labels = class_labels[::-1]
        mat = np.flip(mat)

    axes = sns.heatmap(
        mat.T, square=True, annot=True, fmt=fmt,
        cbar=True, cmap=plt.cm.Blues, ax=ax, **kwargs
    )
    
    axes.set(xlabel='Actual', ylabel='Model Prediction')
    tick_marks = np.arange(len(class_labels)) + 0.5
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(class_labels)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(class_labels, rotation=0)
    axes.set_title(title or 'Confusion Matrix')

    plt.show()

def evaluate_testing_performance(model, y_test, y_pred, x_test):
    '''Evaluates the performance of a classification model.

    Metrics:
    --------
    Accuracy:
    The ratio of correctly predicted instances to total instances.

    Precision:
    The proportion of true positive predictions among all positive predictions.
    High precision means fewer false positives.

    Recall:
    The proportion of true positives identified out of actual positives. 
    High recall indicates fewer false negatives.

    F1 Score:
    The harmonic mean of precision and recall, useful when there's
    an imbalance between classes.

    AUC-ROC:
    Measures the area under the ROC curve, indicating how well the model
    distinguishes between classes.

    Parameters:
    -----------
    model : object
        A classification model object.

    y_test : array
        True labels of the test set.

    y_pred : array
        Predicted labels of the test set.

    x_test : array
        Test set features.
    '''

    # Probability scores for AUC
    y_pred_prob = model.predict_proba(x_test)[:, 1]  
    
    accuracy = accuracy_score(y_test, y_pred)   
    precision = precision_score(y_test, y_pred) 
    recall = recall_score(y_test, y_pred)       
    f1 = f1_score(y_test, y_pred)               
    auc = roc_auc_score(y_test, y_pred_prob)   
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc:.4f}')

def plot_performance_metrics(
    n_epochs: int, subplot_rows: int,
    subplot_cols: int, figsize: tuple, 
    formatters: dict=None,
    colors: list=None, **metrics
    ) -> None:
    """
    Plot the training performance of the model in terms of loss and accuracy.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    subplot_rows : int
        Number of subplot rows.
    subplot_cols : int
        Number of subplot columns.
    figsize : tuple
        Size of the figure (width, height).
    **metrics : dict
        Keyword arguments containing metric histories to plot. Example:
        history_loss=[...], history_val_loss=[...], etc.

    Raises
    ------
    ValueError
        If `history_metrics` is empty or `n_epochs` is not a positive integer.
    """
    if len(metrics) == 0:
        raise ValueError('Metrics to plot missing!')

    if n_epochs <= 0:
        raise ValueError('Number of epochs must be a positive integer!')

    epochs = range(1, n_epochs + 1)
    colors = [
        'blue', 'green', 'yellow', 'magenta', 'red', 'gray', 
        'cyan', 'black','white', 'orange', 'pink', 'brown'
    ]

    max_colors = len(colors)

    # Create a plot for visualizing the training parameters
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
    axes = axes.flatten() if subplot_rows > 1 or subplot_cols > 1 else [axes]  # Flatten for easy indexing

    indices = {}
    param_colors = {}

    for key, val in metrics.items():
        
        # Record the index of the new ax
        data_parameter = key.split('_')[-1]
        if data_parameter not in indices:
            indices[data_parameter] = len(indices)

        if data_parameter not in param_colors:
            param_colors[data_parameter] = 0
        else:
            param_colors[data_parameter] += 1

        # Identify label name based on the dataset type
        if key.startswith('validation') or key.startswith('val'):
            label = 'Validation'
        elif key.startswith('training') or key.startswith('train'):
            label = 'Training'
        elif key.startswith('testing') or key.startswith('test'):
            label = 'Testing'
        else:
            label = 'Metric'

        # Next available color from the palette
        color = colors[param_colors[data_parameter] % max_colors]  # Cycle colors if exceeded

        # Generate labels of the plot
        title = data_parameter.title()  # Plot main title        
        y_label = data_parameter.title()  # Plot y-axis label   
        x_label = 'Epochs'  # Plot x-axis label (constant)

        if formatters is not None and data_parameter in formatters:
            fmt = formatters[data_parameter]
            title = title.replace(data_parameter.title(), fmt(data_parameter))

        # Identify ax to which the values should be plotted
        idx = indices.get(data_parameter, 0)

        # Plot the values to the corresponding ax
        ax = axes[idx]
        ax.plot(epochs, val, color, label=label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

    plt.tight_layout()
    plt.show()
def create_batch_generators(
    img_shape: tuple, train_dir: str, 
    valid_dir: str, test_dir: str, 
    batch_size: int, shuffle:bool=True,
    include_validation: bool=True,
    ) -> tuple:
    '''Creates batch generators for 
    training, validation, and testing
    datasets.

    Parameters:
    -----------
    img_shape : tuple
        The shape of the images.

    train_dir : str
        The directory containing the training dataset.

    valid_dir : str
        The directory containing the validation dataset.

    test_dir : str
        The directory containing the testing dataset.

    batch_size : int
        The number of samples per batch.

    shuffle : bool, default=True
        If True, shuffle the dataset.

    include_validation : bool, default=True
        If True, include the validation dataset.
        If False, concatenate the training and 
        validation datasets.

    Returns:
    --------
    tuple: 
    A tuple containing the training, validation, 
    and testing dataset generators.
    '''

    if len(img_shape) != 2:
        raise ValueError('The image shape must be a tuple of two integers!')

    img_width, img_height = img_shape

    if img_width != img_height:
        raise ValueError('The image shape must be a square!')
    
    if img_height < 1 or img_width < 1:
        raise ValueError('The image shape must be greater than 0!')

    if batch_size < 1:
        raise ValueError('The batch size must be greater than 0!')
    
    if batch_size == 1:
        print('Warning: A batch size of 1 may cause the model to train slowly!')
        
    train_gen= image_dataset_from_directory(
        train_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch
    
    validation_gen = image_dataset_from_directory(
        valid_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch
    
    test_gen = image_dataset_from_directory(
        test_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch

    rescale = lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl)
    train_gen = train_gen.map(rescale)
    validation_gen = validation_gen.map(rescale)
    test_gen = test_gen.map(rescale)

    if not include_validation:
        # Concatenate training and validation datasets
        train_valid_gen = Dataset.concatenate(train_gen, validation_gen)
        return (train_valid_gen, test_gen)
    
    return (train_gen, validation_gen, test_gen)

def create_final_batch_generators(
    img_shape: tuple, train_dir: str, 
    valid_dir: str, test_dir: str, 
    batch_size: int, shuffle:bool=True) -> tuple:
    '''Creates batch generators for 
    training, validation, and testing
    datasets.

    Parameters:
    -----------
    img_width : int
        The width of the images.
        
    img_height : int
        The height of the images.
        
    train_dir : str
        The directory containing the training dataset.
        
    valid_dir : str
        The directory containing the validation dataset.
        
    test_dir : str
        The directory containing the testing dataset.
        
    batch_size : int
        The number of samples per batch.

    Returns:
    --------
    tuple: 
    A tuple containing the training, validation, 
    and testing dataset generators.
    '''

    if len(img_shape) != 2:
        raise ValueError('The image shape must be a tuple of two integers!')

    img_width, img_height = img_shape

    if img_width != img_height:
        raise ValueError('The image shape must be a square!')
    
    if img_height < 1 or img_width < 1:
        raise ValueError('The image shape must be greater than 0!')

    if batch_size < 1:
        raise ValueError('The batch size must be greater than 0!')
    
    if batch_size == 1:
        print('Warning: A batch size of 1 may cause the model to train slowly!')
        
    train_gen= image_dataset_from_directory(
        train_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch
    
    validation_gen = image_dataset_from_directory(
        valid_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch
    
    test_gen = image_dataset_from_directory(
        test_dir,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle
    ).repeat()  # Ensures dataset repetition every epoch

    rescale = lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl)
    train_gen = train_gen.map(rescale)
    validation_gen = validation_gen.map(rescale)
    test_gen = test_gen.map(rescale)

    # Concatenate training and validation datasets
    train_valid_gen = Dataset.concatenate(train_gen, validation_gen)
    
    return (train_valid_gen, test_gen)

def find_optimal_threshold(
    labels: np.ndarray, predictions: np.ndarray, 
    step:float=0.001, score:str='accuracy') -> float:
    '''Finds the optimal threshold value for a binary classifier.
    
    Parameters:
    -----------
    labels : array-like
        The true labels of the data.

    predictions : array-like
        The predicted probabilities of the data.

    step : float, default=0.001
        The step size for the threshold values.

    score : str, default='accuracy'
        The scoring function to use for optimization.
        Options: 'accuracy', 'precision', 'recall', 'f1'.

    Returns:
    --------
    float: The optimal threshold value.
    '''

    score_functions = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

    scorefunc = score_functions.get(score)

    if scorefunc is None:
        raise ValueError('Invalid score function!')

    thresholds = np.arange(0.0, 1.0, step)
    accuracies = [
        scorefunc(labels, (predictions > t).astype(int))
        for t in thresholds
    ]

    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_score = accuracies[best_idx]

    return (best_threshold, best_score)

def binarize_predictions(y, thresh) -> np.ndarray:
    '''Binarizes the predictions of a classifier
    using a specified threshold value.

    Parameters:
    -----------
    y : array-like
        The predicted probabilities of the data.

    thresh : float
        The threshold value for binarization.

    Returns:
    --------
    array-like: The binarized predictions.
    '''
    binarized = y > thresh
    return binarized.astype('uint8')

def augment_training_data(datagen: Dataset) -> Dataset:
    '''Augments the training data using image augmentation layers.

    Parameters:
    -----------
    datagen : tensorflow.data.Dataset
        The training dataset.

    Returns:
    --------
    tensorflow.data.Dataset: The augmented training dataset.
    '''

    # Define image augmentation layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),        
        layers.RandomRotation(0.4),            
        layers.RandomZoom(0.2),              
        layers.RandomTranslation(0.2, 0.2, interpolation='nearest'),    
        layers.RandomContrast(0.2)            
    ])
    
    # Apply the data augmentation to the training dataset
    # before fitting the model for faster training
    augmented_gen = datagen.map(
        lambda x, y: (data_augmentation(x, training=True), y),  
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return augmented_gen

def get_data_labels(datagen: Dataset, steps: int) -> np.ndarray:
    '''Extracts the labels from a dataset.

    Parameters:
    -----------
    datagen : tensorflow.data.Dataset
        The dataset containing the labels.

    steps : int
        The number of steps to extract the labels.

    Returns:
    --------
    np.ndarray: The extracted labels.
    '''

    sliced = itertools.islice(datagen, steps)
    labels = np.concatenate([y for x, y in sliced], axis=0)

    return labels.astype('uint8')

def plot_prediction(model, generator, n_images):
    """
    Test the model on random predictions
    Args:
    generator: a generator instance
    n_images : number of images to plot

    """
    
    class_names = ['Cat', 'Dog']
    
    i = 1
    # Get the images and the labels from the generator
    images, labels = generator.next()
    # Gets the model predictions
    preds = model.predict(images)
    predictions = np.argmax(preds, axis=1)
    labels = labels.astype('int32')
    plt.figure(figsize=(14, 15))
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        if predictions[i] == labels[i]:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='g') 
            plt.axis('off')
        else:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='r') 
            plt.axis('off')
        i += 1
        if i == n_images:
            break
    
    plt.show()


def calculate_percent_dogs(true_labels, predicted_labels):
    bw = np.bitwise_and(true_labels.astype('int64'), predicted_labels.astype('int64'))
    pct =  bw.sum() / len(bw) *100
    print(f'dogs: {pct:.2f}%')

def calculate_percent_cats(true_labels, predicted_labels):
    bw = np.bitwise_or(true_labels.astype('int64'), predicted_labels.astype('int64'))
    pct = (bw == 0).sum() / len(bw) * 100
    print(f'cats: {pct:.2f}%')

def save_model(model, model_dir: str, model_name: str) -> str:
    '''Saves a trained model to disk.

    Parameters:
    -----------
    model : object
        The trained model object.

    model_dir : str
        The directory to save the model.

    model_name : str
        The name of the model file.

    Returns:
    --------
    str: The path to the saved model.
    '''

    model_path = os.path.join('models', model_dir, model_name)
    model.save(model_path)
    
    return model_path

def print_performance_metrics(metrics: dict, n_decimals:int=2) -> None:
    '''Prints the performance metrics for a model.

    Parameters:
    -----------
    metrics : dict
        A dictionary containing the test metrics.
    '''

    print('------------------------------------')

    for metric, value in metrics.items():
        print(f'{metric}: {value:.{n_decimals}f}')

    print('------------------------------------')

