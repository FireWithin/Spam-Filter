# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:52:12 2017

@author: Aditya
"""

import glob, re,os
import spamfilter

path = r"E:\ML\spamfilterdata\*\*\*"
data = []

for fn in glob.glob(path):
    is_spam = "ham" not in fn
    with open(fn, 'r', encoding='latin-1') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = re.sub(r"^Subject: ", "", line).strip()
                data.append((subject, is_spam))

train_data = data

classifier = spamfilter.NaiveBayesClassifier()

classifier.train(data)
