# -*- coding: utf-8 -*-
"""

@author: Aditya
"""
import re, glob, math
from collections import defaultdict

def tokenize(message):
    message = message.lower()
    words = re.findall("[a-z0-9']+", message)
    words = set(words)
    return words

def countWords(train):
    counts = defaultdict(lambda: [0,0])
    for message, is_spam in train:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def probabWords(word_counts, total_spam, total_ham, k = 0.5):
    l = []    
    for w, (spam, ham) in word_counts.items():
        spam_p = (spam + k)/(2*k + total_spam)
        ham_p = (ham + k)/(2*k + total_ham)
        l.append((w, spam_p, ham_p))
    return l

def spam_probability(word_probs, message):
    words = tokenize(message)
    log_prob_spam = 0.
    log_prob_ham = 0.    
    for word, prob_if_spam, prob_if_ham in word_probs:
        if word in words:
            log_prob_spam += math.log(prob_if_spam)
            log_prob_ham += math.log(prob_if_ham)
        else:
            log_prob_spam += math.log(1. - prob_if_spam)
            log_prob_ham += math.log(1. - prob_if_ham)
    prob_spam = math.exp(log_prob_spam)
    prob_ham = math.exp(log_prob_ham)
    return prob_spam / (prob_spam + prob_ham)

class NaiveBayesClassifier:
    def __init__(self, k = 0.5):
        self.k = k
        self.word_probs = []
    
    def train(self, training_data):
        num_spam = len([is_spam for message, is_spam in training_data if is_spam])
        num_ham = len(training_data) - num_spam
        counts = countWords(training_data)
        self.word_probs = probabWords(counts, num_spam, num_ham, self.k)
    
    def classify(self, message):
        return spam_probability(self.word_probs, message)
