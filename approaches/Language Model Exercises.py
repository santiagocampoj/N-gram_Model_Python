
#!/usr/bin/env python
# coding: utf-8

# **Exercise 1: Train a simple trigram language model**
# ---
# 
# -----
# 
# In the first exercise, we´ll save the counts directly in a dictionary
#  which defaults to the smoothing factor (_note that this is not true smoothing
#  as it does not account for the denominator and therefore does not create a
#  true probability distribution, but it is enough to get started_)

# In[1]:


import nltk
from collections import defaultdict
import numpy as np
import nltk.corpus
from nltk.corpus import brown

import numpy as np

# choose a small smoothing factor
smoothing_factor = 0.001
counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))


# We'll also define two helper functions, one to get the log probability of
# a single trigram and the second to get the log probability of a full sentence

# In[2]:


def logP(u, v, w):
    """
    Compute the log probability of a specific trigram
    """
    return np.log2(counts[(u, v)][w]) - np.log2(sum(counts[(u, v)]. values()))


def sentence_logP(S):
    """
    Adds the special tokens to the beginning and end.
    Then calculates the sum of log probabilities of
    all trigrams in the sentence.
    """
    tokens = ['*', '*'] + S + ['STOP']
    return sum([logP(u, v, w) for u, v, w in nltk.ngrams(tokens, 3)])


# We then choose the corpus. We'll use the preprocessed Brown corpus (nltk.corpus.brown), which contains many domains.
# To see the domains, you can run brown.categories(). We also split this into train, dev, and test sets, which we will use throughout.

# In[3]:

sentences = brown.sents(categories='news')
dev_idx = int(len(sentences) * .7)
test_idx = int(len(sentences) * .8)
train = sentences[:dev_idx]
dev = sentences[dev_idx:test_idx]
test = sentences[test_idx:]


# Finally, we'll collect the counts in the dictionary we set up before.

# In[4]:

for sentence in train:
    # add the special tokens to the sentences
    tokens = ['*', '*'] + sentence + ['STOP ']
    for u, v, w in nltk.ngrams(tokens, 3):
        # update the counts
        counts[(u, v)][w] += 1


# In[5]:


# Now that we have the model we can use it
sentence = "My name is Santi"
print('We are gettin the log Probability for this sentece:', sentence)

sentence_split = sentence.split()
print(sentence_logP(sentence_split))

# **Exercise 2: (3-5 minutes) **
# 
# **Try and find the sentence (len > 10 tokens) with the highest probability**
# 1. What is the sentence with the highest probability you could find?
# 2. What is it's log probability?

max_prob = float('-inf')
max_sentence = None

for sentence in test:
    if len(sentence) > 10:
        log_prob = sentence_logP(sentence)

        if log_prob > max_prob:
            max_prob = log_prob
            max_sentence = sentence

print("\n\nSentence with highest probability:", max_sentence, "\nWith log probability:", max_prob)

# **Exercise 3: Function for trigram model, define perplexity, find the best train domain (15-20 minutes) **
# ---
# 
# -----

# First, you'll need to define a function to train the trigram models. It should return the same kind of counts dictionary as in Exercise 1.

# In[6]:



def estimate_lm(corpus, smoothing_factor=0.001):
    """This function takes a corpus and returns a trigram model (counts) trained on the corpus """
    
    # Finish the code here
     # Finish the code here
    counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))
    for sentence in corpus:
        tokens = ['*', '*'] + sentence + ['STOP ']
        for u, v, w in nltk.ngrams(tokens, 3):
            counts[(u, v)][w] += 1
    return counts


# Now, you'll need to define a function to measure perplexity, which is defined as the exp2(total negative log likelihood / total_number_of_tokens). See https://web.stanford.edu/~jurafsky/slp3/3.pdf for more info.
# 
# Luckily, we already have a function to get the log likelihood of a sentence (sentence_logP). So we can iterate over the sentences in a corpus, summing the log probability of each sentence, and keeping track of the total number of tokens. Finally, you can get the NEGATIVE log likelihood and average this, finally using np.exp2 to exponentiate the previous result.

# In[7]:


def perplexity(corpus):
    """
    Perplexity is defined as the exponentiated (np.exp2) average negative log-likelihood of a sequence. 
    """
    total_log_likelihood = 0
    total_token_count = 0
    
    # Finish the code here
    for sentence in corpus:
        total_log_likelihood += sentence_logP(sentence)
        total_token_count += len(sentence)
   
    return np.exp2(-total_log_likelihood / total_token_count)


# In[8]:


test_data = [["I'm", 'not', 'giving', 'you', 'a', 'chance', ',', 'Bill', ',', 'but', 'availing', 'myself', 'of', 'your', 'generous', 'offer', 'of', 'assistance', '.'], ['Good', 'luck', 'to', 'you', "''", '.'], ['``', 'All', 'the', 'in-laws', 'have', 'got', 'to', 'have', 'their', 'day', "''", ',', 'Adam', 'said', ',', 'and', 'glared', 'at', 'William', 'and', 'Freddy', 'in', 'turn', '.'], ['Sweat', 'started', 'out', 'on', "William's", 'forehead', ',', 'whether', 'from', 'relief', 'or', 'disquietude', 'he', 'could', 'not', 'tell', '.'], ['Across', 'the', 'table', ',', 'Hamrick', 'saluted', 'him', 'jubilantly', 'with', 'an', 'encircled', 'thumb', 'and', 'forefinger', '.'], ['Nobody', 'else', 'showed', 'pleasure', '.'], ['Spike-haired', ',', 'burly', ',', 'red-faced', ',', 'decked', 'with', 'horn-rimmed', 'glasses', 'and', 'an', 'Ivy', 'League', 'suit', ',', 'Jack', 'Hamrick', 'awaited', 'William', 'at', 'the', "officers'", 'club', '.'], ['``', 'Hello', ',', 'boss', "''", ',', 'he', 'said', ',', 'and', 'grinned', '.'], ['``', 'I', 'suppose', 'I', 'can', 'never', 'expect', 'to', 'call', 'you', "'", 'General', "'", 'after', 'that', 'Washington', 'episode', "''", '.'], ['``', "I'm", 'afraid', 'not', "''", '.']]


# Finally, use *estimate_lm()* to train LMs on each domain in brown.categories() and 
# find which gives the lowest perplexity on test_data. 
# 
# 1. Which domain gives the best perplexity?
# 2. Can you think of a way to use language models to predict domain?

# In[10]:


print('\n')
for domain in brown.categories():
    train = brown.sents(categories=domain)
    
    # Finish the code here
    counts = estimate_lm(train)
    print(f"Domain '{domain}'  \tPerplexity: {perplexity(test_data)}")


# **Exercise 4: Generation **
# ---
# 
# -----

# For the next exercise, you will need to generate 10 sentences for each domain in the Brown corpus. The first thing we need is code to be able to sample the next word in a trigram. We'll do this by creating a probability distribution over the values in our trigram counts. Remember that each key in the dictionary is a tuple (u, v) and that the values is another dictionary with the count of the continuation w: count. Therefore, we can create a numpy array with the continuation values and divide by the sum of values to get a distribution. Finally, we can use np.random.multinomial to sample from this distribution.

# In[12]:


def sample_next_word(u, v):
    keys, values = zip(* counts[(u, v)]. items())
    # convert values to np.array
    values = np.array(values)
    # divide by sum to create prob. distribution
    values /= values.sum()  
    # return the key (continuation token) for the sample with the highest probability
    return keys[np.argmax(np.random.multinomial(1, values))]  


# Now we can create a function that will generate text using our trigram model. You will need to start out with the two special tokens we used to train the model, and continue adding to this output, sampling the next word at each timestep. If the word sampled is the end token ('STOP'), then stop the generation and return the sequence as a string.

# In[13]:


def generate():
    """
    Sequentially generates text using sample_next_word().
    When the token generated is 'STOP', it returns the generated tokens as a string,
    removing the start and end special tokens.
    """
    result = ['*', '*']
    
    # Finish the code here
    while result[-1] != 'STOP':
        next_word = sample_next_word(result[-2], result[-1])
        result.append(next_word)
    return ' '.join(result[1:-1])

# Finally, use the code above to generate 10 sentences per domain in the Brown corpus.

for domain in brown.categories():
    print(f"\n\Domain: {domain}\n")
    sentences = brown.sents(categories=domain)
    for i, sentence in enumerate(sentences):
        if i == 10:
            break
        print(" ".join(sentence))


# 1. Do you see any correlation between perplexity scores and generated text?

# **Exercise 5: Smoothing **
# ---
# 
# -----

# So far, we have been using a kind of stupid smoothing technique, giving up entirely on computing an actual probability distribution. For this section, let's implement a correct version of Laplace smoothing. You'll need to keep track of the vocabulary as well, and don't forget to add the special tokens.

# In[15]:


def estimate_lm_smoothed(corpus, alpha=1):
    counts = defaultdict(lambda: defaultdict(lambda: alpha))
    vocab = set()
    
    # Finish the code here
    for sentence in corpus:
        tokens = ['*', '*'] + sentence + ['STOP ']
        for u, v, w in nltk.ngrams(tokens, 3):
            counts[(u, v)][w] += 1
            vocab.add(u)
            vocab.add(v)
            vocab.add(w)

    # for context in counts:
    #     total_count = sum(counts[context].values())
    #     for word in counts[context]:
    #         counts[context][word] /= total_count

    return counts, vocab


# The main change is not in how we estimate the counts, but in how we calculate log probability for each trigram.
# Specifically, we need to add the size_of_the_vocabulary * alpha to the denominator.

# In[16]:


def logP_smoothed(u, v, w, V, alpha=1):
    # Finish the code here
    numerator = counts[(u, v)][w] + alpha
    denominator = sum(counts[(u, v)].values()) + alpha * len(V)
    return np.log2(numerator / denominator)

def sentence_logP_smoothed(S, V, alpha=1):
    """
    Adds the special tokens to the beginning and end.
    Then calculates the sum of log probabilities of
    all trigrams in the sentence using logP_smoothed.
    """
    # Finish the code here
    S = ['*', '*'] + S + ['STOP'] 
    logP = 0
    for i in range(2, len(S)):
        u, v, w = S[i-2:i+1]
        logP += logP_smoothed(u, v, w, V, alpha)
    return logP

def perplexity_smoothed(corpus, V, alpha=1):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    # Finish the code here
    logP = 0
    for sentence in corpus:
        logP += sentence_logP_smoothed(sentence, V, alpha)
    return np.exp2(-logP / len(corpus))


# Now train s_counts and vocab and compare perplexity with the original version on the heldout test set.
# vocab = set()
print("\n")
s_counts, s_vocab = estimate_lm_smoothed(train)
perplexity = perplexity_smoothed(test, s_vocab)
print(f"Perplexity on test: {perplexity}")

# **Exercise 6: Interpolation**
# ---
# 
# -----

# To be able to interpolate unigram, bigram, and trigram models, we first need to train them. So here you need to make a function that takes 1) a corpus and 2) an n-gram (1,2,3) and 3) a smoothing factor and returns the counts and vocabulary. Notice that for the unigram model, you will have to set up the dictionary in a different way than we have done until now.

# In[17]:


def estimate_ngram(corpus, N=3, smoothing_factor=1):
    vocab = set(['*', 'STOP'])
    # corpus_flat = [word for sentence in corpus for word in sentence]
    if N > 1:
        # set up the counts like before
        counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))
        for sentence in corpus:
            tokens = ['*'] * (N-1) + sentence + ['STOP']
            for ngram in nltk.ngrams(tokens, N):
                prefix = ngram[:-1]
                word = ngram[-1]
                
                counts[prefix][word] += 1
                vocab.add(word)
               
                if len(prefix) == 1:
                    vocab.add(prefix[0])
                elif len(prefix) == 2:
                    vocab.add(prefix)
                else:
                    pass

    else:
        # set them up as necessary for the unigram model
        counts = defaultdict(lambda: smoothing_factor)
        for sentence in corpus:
            tokens = sentence + ['STOP']
            
            for word in tokens:
                counts[word] += 1
                vocab.add(word)
        
    # Finish the code here
    return counts, vocab


# You will also need separate functions to get the log probability for each ngram.

# In[18]:


def logP_trigram(counts, u, v, w, vocab, alpha=1):
    # Finish the code here
    numerator = counts[(u, v)][w] + alpha
    denominator = sum(counts[(u, v)].values()) + alpha * len(vocab)
    return np.log2(numerator / denominator)


def logP_bigram(counts, u, v, vocab, alpha=1):
    # Finish the code here
    numerator = counts[u][v] + alpha
    denominator = sum(counts[u].values()) + alpha * len(vocab)
    return np.log2(numerator / denominator)

def logP_unigram(counts, u, vocab, alpha=1):
    # Finish the code here
    numerator = counts[u] + alpha
    denominator = sum(counts.values()) + alpha * len(vocab)
    return np.log2(numerator / denominator)


# In this case, the main change is in calculating the log probability of the sentence. 

# In[21]:


def sentence_interpolated_logP(S, vocab, uni_counts, bi_counts, tri_counts, lambdas=[0.5, 0.3, 0.2]):
    tokens = ['*', '*'] + S + ['STOP']
    prob = 0
    for u, v, w in nltk.ngrams(tokens, 3):
        # Calculate the log probabilities for each ngram and then multiply them by the lambdas and sum them.
        if (u, v) in tri_counts:
            tri_prob = logP_trigram(tri_counts, u, v, w, vocab)
        else:
            tri_prob = 0
        if u in bi_counts:
            bi_prob = logP_bigram(bi_counts, u, v, vocab)
        else:
            bi_prob = 0
        
        uni_prob = logP_unigram(uni_counts, w, vocab)
        
        prob += np.log2(lambdas[0] * 2**tri_prob + lambdas[1] * 2**bi_prob + lambdas[2] * 2**uni_prob)
               
    return prob

def interpolated_perplexity(corpus, vocab, uni_counts, bi_counts, tri_counts, smoothing_factor=1, lambdas=[0.5, 0.3, 0.2]):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. 
    In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    p = 0
    # Finish the code here
    for sentence in corpus:
        p += sentence_interpolated_logP(sentence, vocab, uni_counts, bi_counts, tri_counts, lambdas)
    return np.exp2(-p / len(corpus))

# Finally, train unigram, bigram, and trigram models and computer the perplexity of the interpolated model on the test set.
print("\n")
from collections import Counter

train_tokens = [token for sentence in train for token in sentence]
vocab = set(train_tokens)

uni_counts = Counter(train_tokens)
bi_counts = Counter(nltk.bigrams(train_tokens, pad_left=True, pad_right=True))
tri_counts = Counter(nltk.ngrams(train_tokens, 3, pad_left=True, pad_right=True))

test_perplexity = interpolated_perplexity(test, vocab, uni_counts, bi_counts, tri_counts)
print(f"Interpolated model perplexity on test set: {test_perplexity:}")


# train_tokens = [token for sentence in train for token in sentence]

# vocab = set(train_tokens)

# uni_counts = estimate_ngram(train, 1)
# bi_counts = estimate_ngram(train, 2)
# tri_counts = estimate_ngram(train, 3)

# test_perplexity = interpolated_perplexity(test, vocab, uni_counts, bi_counts, tri_counts)
# print(f"Interpolated model perplexity on test set: {test_perplexity:}")

# s_counts, s_vocab = estimate_lm_smoothed(train)
# perplexity = perplexity_smoothed(test, s_vocab)
# print(f"Perplexity on test: {perplexity}")
