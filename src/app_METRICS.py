import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

'''
#######################################################################
#   PERFORMANCE METRICS FUNCTIONS
#######################################################################
'''

def PA(pred, label):    
    """
    Calculates the Pixel Accuracy (PA) between the predicted and true labels.

    Args:
        pred (numpy.ndarray): Predicted labels.
        label (numpy.ndarray): True labels.

    Returns:
        float: Pixel accuracy.
    """
    dnum = pred.shape[0] * pred.shape[1]
    num = dnum - pred.sum() - label.sum() + 2 * (pred * label).sum()
    return num / dnum

def DSC(pred, label):
    """
    Calculates the Dice Similarity Coefficient (DSC) between the predicted and true labels.

    Args:
        pred (numpy.ndarray): Predicted labels.
        label (numpy.ndarray): True labels.

    Returns:
        float: Dice similarity coefficient.
    """
    SMOOTH = 1e-8
    intersection = (pred * label).sum()
    return (2 * intersection + SMOOTH) / (pred.sum() + label.sum() + SMOOTH)

def PC(pred, label):
    """
    Calculates the Precision (PC) between the predicted and true labels.

    Args:
        pred (numpy.ndarray): Predicted labels.
        label (numpy.ndarray): True labels.

    Returns:
        float: Precision.
    """
    return (pred * label).sum() / pred.sum()

'''
#######################################################################
#   TRAINING METRICS FUNCTIONS
#######################################################################
'''

def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function.

    Args:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch < 50:
        return lr * 0.954992
    elif epoch >= 50 and epoch < 400:
        return lr
    else:
        return lr

def get_lr_metric(optimizer):
    """
    Returns a learning rate metric for a given optimizer.

    Args:
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.

    Returns:
        function: Learning rate metric function.
    """
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

def dice_coef(y_true, y_pred, smooth=1.):
    """
    Calculates the Dice coefficient between the true and predicted labels.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    yT = K.flatten(y_true)
    yP = K.flatten(y_pred)
    intersection = K.sum(yT * yP)
    DICE = (2. * intersection + smooth) / (K.sum(yT) + K.sum(yP) + smooth)
    return DICE

def weighted_dice_coef(weights):
    """
    Returns a function that calculates the weighted Dice coefficient for multi-class labels.

    Args:
        weights (list): Weights for each class.

    Returns:
        function: Weighted Dice coefficient function.
    """
    def DICE(y_true, y_pred, smooth=1.):
        weighted_DICE = 0
        for index in range(y_pred.shape[-1]):
            yT = K.flatten(y_true[:, :, :, index])
            yP = K.flatten(y_pred[:, :, :, index])
            intersection = K.sum(yT * yP)
            weighted_DICE += (2. * intersection + smooth) * weights[index] / (K.sum(yT) + K.sum(yP) + smooth)     
        return weighted_DICE / np.sum(weights)
    return DICE

def IoU(y_true, y_pred):
    """
    Calculates the Intersection over Union (IoU) between the true and predicted labels.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        float: Intersection over Union.
    """
    yT = K.flatten(y_true)
    yP = K.flatten(y_pred)
    intersection = K.sum(yT * yP)
    union = K.sum(yT) + K.sum(yP) - intersection
    IOU = intersection / union
    return IOU

def weighted_IoU(weights):
    """
    Returns a function that calculates the weighted IoU for multi-class labels.

    Args:
        weights (list): Weights for each class.

    Returns:
        function: Weighted IoU function.
    """
    def IOU(y_true, y_pred):
        weighted_IOU = 0
        for index in range(y_pred.shape[-1]):
            yT = K.flatten(y_true[:, :, :, index])
            yP = K.flatten(y_pred[:, :, :, index])
            intersection = K.sum(yT * yP)
            union = K.sum(yT) + K.sum(yP) - intersection
            weighted_IOU += intersection * weights[index] / union
        return weighted_IOU / np.sum(weights)
    return IOU

'''
#######################################################################
#   LOSS FUNCTIONS
#######################################################################
'''

def weighted_binary_crossentropy(weight):
    """
    Returns a function that calculates the weighted binary cross-entropy loss.

    Args:
        weight (float): Weight for positive class.

    Returns:
        function: Weighted binary cross-entropy loss function.
    """
    weight = tf.constant(weight)
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weight)
    return loss

def weighted_categorical_crossentropy(weights):
    """
    Returns a function that calculates the weighted categorical cross-entropy loss.

    Args:
        weights (list): Weights for each class.

    Returns:
        function: Weighted categorical cross-entropy loss function.
    """
    weights = tf.constant(weights)
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weights)
    return loss 

def weighted_categorical_crossentropy(weights):
    """
    Returns a function that calculates the weighted categorical cross-entropy loss.

    Args:
        weights (list): Weights for each class.

    Returns:
        function: Weighted categorical cross-entropy loss function.
    """
    weights = K.variable(weights) 
    def loss(y_true, y_pred):
        # scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculate loss
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss
