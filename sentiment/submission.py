#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter, defaultdict
from util import *


############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {"pretty":2, "good":0, "bad":-1, "plot":-1, "not":-1, "scenery":1}
    
    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    wordreturn=collections.defaultdict(int)
    for word in x.split():
        wordreturn[word]+=1
    return wordreturn
    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    dif={}
    for x,y in trainExamples:
        for temp in featureExtractor(x):
            weights[temp]=0
            dif[temp]=0
    for i in range(numIters):
        for x,y in trainExamples:
            if y==1:
                phi=featureExtractor(x)
                p=sigmoid(dotProduct(weights, phi))
                for k in phi:
                    dif[k]=phi[k]*(p-1)
                for z in featureExtractor(x):
                    weights[z]-=eta*dif[z]
            if y==-1:
                phi=featureExtractor(x)
                p=1-sigmoid(dotProduct(weights, phi))
                for k in phi:
                    dif[k]=phi[k]*(1-p)
                for z in featureExtractor(x):
                    weights[z]-=eta*dif[z]
    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    wordreturn=collections.defaultdict(int)
    spl=x.split()
    length=len(spl)
    spl.append('</s>')
    a='<s>'
    wordreturn[(a, spl[0])]+=1
    for i in range(length):
        wordreturn[spl[i]]+=1
        wordreturn[(spl[i], spl[i+1])]+=1
    return wordreturn
    # END_YOUR_ANSWER
    return phi
