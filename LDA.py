import numpy as np
from random import *
from collections import Counter

rawDocs = ["eat turkey on turkey day holiday",
    "i like to eat cake on holiday",
    "turkey trot race on thanksgiving holiday",
    "snail race the turtle",
    "time travel space race",
    "movie on thanksgiving",
    "movie at air and space museum is cool movie",
    "aspiring movie star"]

# docs is a list of lists of words
docs = [d.split(" ") for d in rawDocs]
numDocs = len(docs)

# Unique words (index is the wordID)
vocab = list(set([word for doc in docs for word in doc]))
numWords = len(vocab)

# Replace words in docs with the wordIDs
for i in range(numDocs):
    docs[i] = [vocab.index(word) for word in docs[i]]

# Number of clusters (topics)
K = 2

# Create WORD-TOPIC matrix (num topics x num words)
# (topic number -> word ID -> count of the word assigned to the topic)
wt = np.zeros((K, numWords)))

# Create TOPIC ASSIGNMENT list (num docs x num words in doc)
# (doc number -> word index in doc -> topic number)
ta = np.array([[0] * len(doc) for doc in docs])

# Create DOCUMENT-TOPIC matrix (num docs x num topics)
# (doc number -> topic number -> count of words assigned to the topic)
dt = np.zeros((numDocs, K))

# Go through each doc and randomly assign each word in the doc to one of the
# topics.
seed(77)
for (docIndex, doc) in enumerate(docs):
    # Randomly assign topic to word w.
    for (wordIndex, wordID) in enumerate(doc):
        topic = randint(K)
        ta[docIndex][wordIndex] = topic

        wt[topic][wordID] += 1

    # Populate document-topic matrix by counting words in each doc assigned
    # to each topic
    for docIndex in range(numDocs):
        for topic in range(K):
            dt[docIndex][topic] = Counter(ta[docIndex])[topic]


# GIBBS SAMPLING

# Higher alpha means each doc is likely to contain a mixture of more
# topics instead of a single topic
alpha = 1
# Higher beta means each topic is likely to contain a mixture of more
# words and not any word specifically
beta = 1

nIter = 1000

print(vocab)
print("topic assignment")
print(ta)
print("word topic:")
print(wt)
print("document topic")
print(dt)

for it in range(nIter):
    # Go through each document and reassign a new topic for each word in the doc
    for (docIndex, doc) in enumerate(docs):
        for (wordIndex, wordID) in enumerate(doc):
            curTopic = ta[docIndex][wordIndex]

            # Do not include this current word when sampling
            dt[docIndex][curTopic] -= 1
            wt[curTopic][wordID] -= 1

            # Populate the probability that this word would have topic t
            topic_probs = [0] * K

            for topic in range(K):
                # (1) Calculate probability of word w given topic t
                #   (count of word w assigned to topic t) + beta
                # = ------------------------------------------------------------------
                #   (total number of words assigned topic t) + (total num words * beta)  
                prob_word = ((wt[topic][wordID] + beta)/
                    (np.sum(wt[topic]) + numWords * beta))

                # (2) Calculate probability of topic t given doc d
                #   (count of words in doc d assigned to topic t) + alpha
                # = -----------------------------------------------------
                #   (total number of words in doc d) + (number of topics * alpha)
                prob_topic = ((dt[docIndex][topic] + alpha)/
                    (np.sum(dt[docIndex]) + K * alpha))

                topic_probs[topic] = prob_word * prob_topic

            # Choose topic t with the probability topic_probs
            new_topic = np.random.choice(K, 1, topic_probs)[0]
            #print("new topic for word " + vocab[wordID] + " is " + str(new_topic) +
            #    " and was " + str(curTopic) + " before")

            # Update matrices
            dt[docIndex][new_topic] += 1
            wt[new_topic][wordID] += 1
            ta[docIndex][wordIndex] = new_topic



print("done")
print("topic assignment")
print(ta)
print("word topic:")
print(wt)
print("document topic")
print(dt)







