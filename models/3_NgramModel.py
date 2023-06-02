import numpy as np
import nltk
from collections import defaultdict
from datasets import load_dataset

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts_dict = {}
        self.vocab = None
    
    def train(self, corpus):
        self.vocab = set(['*', 'STOP'])
        
        for N in range(1, self.n + 1):
            if N > 1:
                counts = defaultdict(lambda: defaultdict(lambda: self.smoothing_factor))
                for sentence in corpus:
                    tokens = ['*'] * (N-1) + sentence + ['STOP']
                    for ngram in nltk.ngrams(tokens, N):
                        prefix = ngram[:-1]
                        word = ngram[-1]

                        counts[prefix][word] += 1
                        self.vocab.add(word)
            else:
                counts = defaultdict(lambda: self.smoothing_factor)
                for sentence in corpus:
                    tokens = sentence + ['STOP']

                    for word in tokens:
                        counts[word] += 1
                        self.vocab.add(word)

            self.counts_dict[N] = counts

        return self.counts_dict, self.vocab

    
    # GENERATION
    # GENERATE based on the highest n-gram model
    def sample_next_word(self, *prev_words): # unpack and pass them
        n = len(prev_words) + 1
        prev_words = prev_words[-(n - 1):]
        keys, values = zip(*self.counts_dict[n][tuple(prev_words)].items()) # unpack and pass them

        values = np.array(values, dtype=np.float64)  #convert values to float64 otherwise it crashes
        values /= values.sum()

        return keys[np.argmax(np.random.multinomial(1, values))]


    def generate(self):
        result = ['*'] * (self.n - 1)

        while result[-1] != 'STOP':
            next_word = self.sample_next_word(*result[-(self.n - 1):]) # unpack and pass them
            result.append(next_word)

        return ' '.join(result[self.n - 1:-1])

    # PROBABILITIES
    def log_probability(self, *ngram):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
        numerator = self.counts_dict[n][tuple(prefix)][word] + self.smoothing_factor
        denominator = sum(self.counts_dict[n][tuple(prefix)].values()) + self.smoothing_factor * len(self.vocab)

        return np.log2(numerator / denominator)


   
    def sentence_log_probability(self, sentence):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0
        
        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.log_probability(*ngram)
        
        return logP

    # PERPLEXITY
    def perplexity(self, corpus):
    logP = 0
    token_count = 0

    for sentence in corpus:
        logP += self.sentence_log_probability(sentence)
        token_count += len(sentence)

    return np.exp2(-logP / token_count)

###############################################################################
###############################################################################
###############################################################################

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
    
    def logP_4gram(counts, u, v, w, x, vocab, alpha=1):
        numerator = counts[(u, v, w)][x] + alpha
        denominator = sum(counts[(u, v, w)].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)

    def logP_5gram(counts, u, v, w, x, y, vocab, alpha=1):
        numerator = counts[(u, v, w, x)][y] + alpha
        denominator = sum(counts[(u, v, w, x)].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)
###############################################################################
    def logP_unigram(self, u, alpha=1):
        counts = self.counts_dict[1]
        vocab = self.vocab
        numerator = counts[u] + alpha
        denominator = sum(counts.values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)

    def logP_bigram(self, u, v, alpha=1):
        counts = self.counts_dict[2]
        vocab = self.vocab
        numerator = counts[u][v] + alpha
        denominator = sum(counts[u].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)

    def logP_trigram(self, u, v, w, alpha=1):
        counts = self.counts_dict[3]
        vocab = self.vocab
        numerator = counts[(u, v)][w] + alpha
        denominator = sum(counts[(u, v)].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)

    def logP_4gram(self, u, v, w, x, alpha=1):
        counts = self.counts_dict[4]
        vocab = self.vocab
        numerator = counts[(u, v, w)][x] + alpha
        denominator = sum(counts[(u, v, w)].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)

    def logP_5gram(self, u, v, w, x, y, alpha=1):
        counts = self.counts_dict[5]
        vocab = self.vocab
        numerator = counts[(u, v, w, x)][y] + alpha
        denominator = sum(counts[(u, v, w, x)].values()) + alpha * len(vocab)
        return np.log2(numerator / denominator)
    
    
###############################################################################
    
    def logP_ngram(n, counts_dict, vocab, *ngram, alpha=1):
    if n == 1:
        return logP_unigram(counts_dict[1], ngram[0], vocab, alpha)
    elif n == 2:
        return logP_bigram(counts_dict[2], ngram[0], ngram[1], vocab, alpha)
    
    elif n == 3:
        return logP_trigram(counts_dict[3], ngram[0], ngram[1], ngram[2], vocab, alpha)
    elif n == 4:
        return logP_4gram(counts_dict[4], ngram[0], ngram[1], ngram[2], ngram[3], vocab, alpha)
    elif n == 5:
        return logP_5gram(counts_dict[5], ngram[0], ngram[1], ngram[2], ngram[3], ngram[4], vocab, alpha)
    else:
        raise ValueError("Invalid n-gram order")
###############################################################################
    def logP_ngram_auto(counts_dict, vocab, *ngram, alpha=1):
    n = len(ngram)
    return logP_ngram(n, counts_dict, vocab, *ngram, alpha=alpha)
###############################################################################
    def calculate_all_log_probabilities(corpus, model):
        all_log_probabilities = {}

        for n in range(1, 6):
            log_probabilities = {}

            for sentence in corpus:
                tokens = ['*'] * (n - 1) + sentence + ['STOP']
                ngrams = list(nltk.ngrams(tokens, n))

                for ngram in ngrams:
                    if ngram not in log_probabilities:
                        log_probabilities[ngram] = logP_ngram_auto(model.counts_dict, model.vocab, *ngram)

            all_log_probabilities[n] = log_probabilities

        return all_log_probabilities
###############################################################################
    def ngram_probability(self, *ngram):
        n = len(ngram)
        log_prob = self.log_probability(n, *ngram)
        return np.exp2(log_prob)



###############################################################################
###############################################################################
###############################################################################


####################################
####################################

# SMOOTH PROBABILITIES
    def log_probability_smoothed(self, *ngram, alpha=1):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
        numerator = self.counts_dict[n][tuple(prefix)][word] + alpha
        denominator = sum(self.counts_dict[n][tuple(prefix)].values()) + alpha * len(self.vocab)
        return np.log2(numerator / denominator)
    
    def sentence_log_probability_smoothed(self, sentence, alpha=1):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0
        
        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.log_probability_smoothed(*ngram, alpha=alpha)
        
        return logP
    
    # PERPLEXITY
    def perplexity_smoothed(self, corpus, alpha=1):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability_smoothed(sentence, alpha=alpha)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)


####################################
####################################


    # INTERPOLATION
    def interpolated_perplexity(self, corpus, lambdas=[0.5, 0.3, 0.2], smoothing_factor=1):
        """
        Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. 
        In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
        """
        p = 0
        for sentence in corpus:
            p += self.sentence_interpolated_logP(sentence, lambdas, smoothing_factor)
        return np.exp2(-p / len(corpus))
    
    def sentence_interpolated_logP(self, sentence, lambdas, smoothing_factor):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0

        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.interpolated_log_probability(ngram, lambdas, smoothing_factor)

        return logP
    
    def interpolated_log_probability(self, ngram, lambdas, smoothing_factor):
        interpolated_logP = 0
        for i, lambda_val in enumerate(lambdas):
            prefix, word = ngram[:i+1], ngram[i+1]
            numerator = self.counts[prefix][word] + smoothing_factor
            denominator = sum(self.counts[prefix].values()) + smoothing_factor * len(self.vocab)
            interpolated_logP += lambda_val * np.log2(numerator / denominator)

        return interpolated_logP 
    

###########
###########
###########

###
# here we just check the use of the different n-grams we just implemented
corpus = [["hello", "world"], ["world", "peace"]]
model = NgramModel()
counts_dict, vocab = model.train(corpus)

ngram_order = 3
ngram = ("*", "hello", "world")
log_probability = logP_ngram(ngram_order, counts_dict, vocab, *ngram)
print(log_probability)
###
# test the train function
corpus = [["hello", "world"], ["world", "peace"]]
model = NgramModel(corpus)
ngram = ("*", "hello", "world")
log_probability = logP_ngram_auto(model.counts_dict, model.vocab, *ngram)
print(log_probability)
##
corpus = [["hello", "world"], ["world", "peace"]]
model = NgramModel(corpus)
all_log_probabilities = calculate_all_log_probabilities(corpus, model)
##
model = NgramModel(n=3)
model.train(corpus)

unigram_prob = model.ngram_probability('the')
bigram_prob = model.ngram_probability('*', 'the')
trigram_prob = model.ngram_probability('*', '*', 'the')
##
model = NgramModel(n=5)
model.train(corpus)

logP_4gram = model.logP_4gram('*', '*', 'the', 'quick')

logP_5gram = model.logP_5gram('*', '*', 'the', 'quick', 'brown')




    
# only the first 1000 articles
wikipedia_text = []
for i in range(1000):
    article = wikipedia_dataset['train'][i]
    text = article['text']
    wikipedia_text.append(text)

from nltk.tokenize import sent_tokenize

# into sentences
from nltk.tokenize import sent_tokenize
sentences = []
for text in wikipedia_text:
    text_sentences = sent_tokenize(text)
    sentences.extend(text_sentences)

#into words
from nltk.tokenize import word_tokenize
tokenized_sentences = []
for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    tokenized_sentences.append(tokenized_sentence)

sentences = tokenized_sentences

# Split data
dev_idx = int(len(sentences) * .7)
test_idx = int(len(sentences) * .8)
train = sentences[:dev_idx]
dev = sentences[dev_idx:test_idx]
test = sentences[test_idx:]
