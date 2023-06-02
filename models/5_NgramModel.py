# requeriments 
# !pip install datasets apache_beam mwparserfromhell

import random
import numpy as np
import nltk
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from datasets import load_dataset

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts_dict = {}
        self.vocab = None
        self.log_probabilities = defaultdict(lambda: {})
    
    ##################################################
    # TRAIN
    ##################################################
    
    def train(self, corpus):
        self.vocab = set(['*', 'STOP'])
        
        for n in range(1, self.n + 1):
            if n > 1:
                counts = defaultdict(lambda: defaultdict(lambda: self.smoothing_factor))
                for sentence in corpus:
                    tokens = ['*'] * (n-1) + sentence + ['STOP']
                    
                    for ngram in nltk.ngrams(tokens, n):
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

            self.counts_dict[n] = counts

        return self.counts_dict, self.vocab
    
    # COUNTING N-GRAMS
    def ngram_count(self):
        print("\n")
        for n in range(1, min(6, self.n + 1)):
            ngram_count = 0
            
            if n == 1:
                ngram_count = len(self.counts_dict[n])
            
            else:
                for prefix in self.counts_dict[n]:
                    ngram_count += len(self.counts_dict[n][prefix])
            
            print(f"Number of {n}-grams: {ngram_count}")
    
    # Top N-Grams
    def top_ngrams(self, K=5):
        for n in range(1, min(6, self.n + 1)):
            ngram_freq = {}
            
            if n == 1:
                ngram_freq = self.counts_dict[n]
            
            else:
                for prefix in self.counts_dict[n]:
                    for word in self.counts_dict[n][prefix]:
                        ngram = prefix + (word,)
                        ngram_freq[ngram] = self.counts_dict[n][prefix][word]

            sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)
            top_k_ngrams = sorted_ngrams[:K]

            print(f"Top {K} {n}-grams with probabilities:")
            
            for ngram, count in top_k_ngrams:
                if n > 1:
                    logP = self.log_probability(*ngram)
                else:
                    logP = self.log_probability(ngram[0])
                
                probability = np.exp2(logP)
                
                print(f"{ngram}: {count}, probability: {probability}")
            
            print("\n")

    ##################################################
    # GENERATE SENTENCE
    ##################################################
    
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

    ##################################################
    # REGULAR PERPLEXITY AND PROBABILITIES
    ##################################################

    # PROBABILITIES
    def log_probability(self, *ngram):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
        if n == 1:
            numerator = self.counts_dict[n][tuple(prefix)][word]  # no smoothing_factor
            denominator = sum(self.counts_dict[n][tuple(prefix)].values())
        else:
            numerator = self.counts_dict[n][tuple(prefix)][word]  # no smoothing_factor
            denominator = sum(self.counts_dict[n][tuple(prefix)].values())  
            
        # x = np.log2(numerator / denominator)
        # print(x)

        # logP = np.log2(numerator / denominator)
        # self.log_probabilities[n][ngram] = logP
        return np.exp2(numerator / denominator)

    
    def top_log_probabilities(self, K=5):
        for n in range(1, min(6, self.n + 1)):
            sorted_log_probabilities = sorted(self.log_probabilities[n].items(), key=lambda x: x[1], reverse=True)
            top_k_log_probabilities = sorted_log_probabilities[:K]

            print(f"Top {K} {n}-grams with log probabilities:")
            for ngram, logP in top_k_log_probabilities:
                print(f"{ngram}: log probability: {logP}")
            print("\n")
    
    def all_ngrams_log_probabilities(self):
    ngrams_log_probs = []

    for n in range(1, self.n + 1):
        if n == 1:
            prefix = ()
            words_counts = self.counts_dict[n]
            for word, count in words_counts.items():
                ngram = (word,)
                try:
                    log_prob = self.log_probability(*ngram)
                except ZeroDivisionError:
                    continue
                ngrams_log_probs.append((ngram, log_prob))
        else:
            for prefix, words_counts in list(self.counts_dict[n].items()):
                for word, count in words_counts.items():
                    ngram = prefix + (word,)
                    try:
                        log_prob = self.log_probability(*ngram)
                    except ZeroDivisionError:
                        continue
                    ngrams_log_probs.append((ngram, log_prob))

    return ngrams_log_probs
    
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
            sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
            for i in range(self.n - 1, len(sentence)):
                ngram = tuple(sentence[i - self.n + 1:i + 1])
                logP += self.log_probability(*ngram)
                token_count += len(sentence)

        return np.exp2(-logP / token_count)

    ##################################################
    # SMOOTHED PERPLEXITY AND PROBABILITIES
    ##################################################

    # PROBABILITIES SMOOTHED
    def log_probability_smoothed(self, *ngram, alpha=1):
      n = len(ngram)
      prefix, word = ngram[:-1], ngram[-1]

      if n == 1:
          numerator = self.counts_dict[n][word] + alpha
          denominator = sum(self.counts_dict[n].values()) + alpha * len(self.vocab)
      
      else:
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
    
    # PERPLEXITY SMOOTHED
    def perplexity_smoothed(self, corpus, alpha=1):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability_smoothed(sentence, alpha=alpha)
            token_count += len(sentence)

        return np.exp2(-logP / token_count) 
    
    ##################################################
    # SMOOTHED ABSOLUTE DISCOUNTIGN PERPLEXITY AND PROBABILITIES
    ##################################################
    
    def log_probability_smoothed_absolute_discounting(self, *ngram, alpha=1, discount=0.75):
      n = len(ngram)
      prefix, word = ngram[:-1], ngram[-1]

      if n == 1:
          numerator = max(self.counts_dict[n][word] - discount, 0) + alpha
          denominator = sum(self.counts_dict[n].values()) - discount * len(self.vocab) + alpha * len(self.vocab)
      
      else:
          numerator = max(self.counts_dict[n][tuple(prefix)][word] - discount, 0) + alpha
          denominator = sum(self.counts_dict[n][tuple(prefix)].values()) - discount * len(self.vocab) + alpha * len(self.vocab)

      return numerator / denominator

    def sentence_log_probability_smoothed_absolute_discounting(self, sentence, alpha=1, discount=0.75):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0

        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.log_probability_smoothed_absolute_discounting(*ngram, alpha=alpha, discount=discount)

        return logP
    
    def perplexity_smoothed_absolute_discounting(self, corpus, alpha=1, discount=0.75):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability_smoothed_absolute_discounting(sentence, alpha=alpha, discount=discount)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)

    ######################################################
    # INTERPOLATION
    ######################################################

    # INTERPOLATED PROBABILITY
    def interpolated_log_probability(self, ngram, lambdas):
      interpolated_logP = 0
      max_n = len(lambdas) + 1

      for i in range(1, min(len(ngram) + 1, max_n)):
          prefix, word = ngram[:-i], ngram[-i]
          logP = self.log_probability_smoothed(*prefix, word, alpha=self.smoothing_factor)
          interpolated_logP += lambdas[i - 1] * logP

      return interpolated_logP

    # INTERPOLATE SENTECE PROBABILITY
    def sentence_interpolated_logP(self, sentence, lambdas):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0

        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.interpolated_log_probability(ngram, lambdas)

        return logP

    # INTERPOLATED PERPLEXITY
    def interpolated_perplexity(self, corpus, lambdas=[0.1, 0.2, 0.2, 0.25, 0.25]):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_interpolated_logP(sentence, lambdas)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)
    
#############################################
#############################################
#############################################

############################
# WIKIPEDIA DATASET
############################

# load dataset
wikipedia_dataset = load_dataset("wikipedia", "20220301.en")

nltk.download('punkt')

# only the first 1000 articles
wiki_text = []
for i in range(1000):
    article = wikipedia_dataset['train'][i]
    text = article['text']
    wiki_text.append(text)

# into sentences
wiki_sentences = []
for text in wiki_text:
    text_sentences = sent_tokenize(text)
    wiki_sentences.extend(text_sentences)

# into words
wiki_tokenized_sentences = []
for sentence in wiki_sentences:
    tokenized_sentence = word_tokenize(sentence)
    wiki_tokenized_sentences.append(tokenized_sentence)

wiki_sentences = wiki_tokenized_sentences

# train, dev, and test 
wiki_dev_idx = int(len(wiki_sentences) * .7)
wiki_test_idx = int(len(wiki_sentences) * .8)
wiki_train = wiki_sentences[:wiki_dev_idx]
wiki_dev = wiki_sentences[wiki_dev_idx:wiki_test_idx]
wiki_test = wiki_sentences[wiki_test_idx:]

# print and check
print(f"\n100 first characters of wiki_train:\n {wiki_train[:100]}")
print(f"\n100 first characters of wiki_train:\n {wiki_dev[:100]}")
print(f"\n100 first characters of wiki_train:\n {wiki_test[:100]}")

############################
# TWEET_EVAL DATASET
############################

# load dataset
tweet_eval_dataset = load_dataset("tweet_eval", 'emotion')

# only the first 1000 tweets
tweet_text = []
for i in range(1000):
    tweet = tweet_eval_dataset['train'][i]
    text = tweet['text']
    tweet_text.append(text)

# into sentences
tweet_sentences = []
for text in tweet_text:
    text_sentences = sent_tokenize(text)
    tweet_sentences.extend(text_sentences)

#into words
tweet_tokenized_sentences = []
for sentence in tweet_sentences:
    tokenized_sentence = word_tokenize(sentence)
    tweet_tokenized_sentences.append(tokenized_sentence)

tweet_sentences = tweet_tokenized_sentences

# train, dev, and test 
tweet_dev_idx = int(len(tweet_sentences) * 0.7)
tweet_test_idx = int(len(tweet_sentences) * 0.8)
tweet_train = tweet_sentences[:tweet_dev_idx]
tweet_dev = tweet_sentences[tweet_dev_idx:tweet_test_idx]
tweet_test = tweet_sentences[tweet_test_idx:]

# print and check
print(f"\n100 first characters of tweet_train:\n {tweet_train[:100]}")
print(f"\n100 first characters of tweet_train:\n {tweet_dev[:100]}")
print(f"\n100 first characters of tweet_train:\n {tweet_test[:100]}")

############################
# IMDB DATASET
############################

# load dataset
imdb_dataset = load_dataset("imdb")

# only the first 1000 
imdb_text = []
for i in range(1000):
    review = imdb_dataset['train'][i]
    text = review['text']
    imdb_text.append(text)

# into sentences
imdb_sentences = []
for text in imdb_text:
    text_sentences = sent_tokenize(text)
    imdb_sentences.extend(text_sentences)

#into words
imdb_tokenized_sentences = []
for sentence in imdb_sentences:
    tokenized_sentence = word_tokenize(sentence)
    imdb_tokenized_sentences.append(tokenized_sentence)

imdb_sentences = imdb_tokenized_sentences

# train, dev, and test
imdb_dev_idx = int(len(imdb_sentences) * 0.7)
imdb_test_idx = int(len(imdb_sentences) * 0.8)
imdb_train = imdb_sentences[:imdb_dev_idx]
imdb_dev = imdb_sentences[imdb_dev_idx:imdb_test_idx]
imdb_test = imdb_sentences[imdb_test_idx:]

# print and check
print(f"\n100 first characters of imdb_train:\n {imdb_train[:100]}")
print(f"\n100 first characters of imdb_train:\n {imdb_dev[:100]}")
print(f"\n100 first characters of imdb_train:\n {imdb_test[:100]}")

#############################################
#############################################

print("########## SENTENCES GENERATION AND PROBABILITY ##########")

# train N-gram model
ngram_model = NgramModel(n=3, smoothing_factor=1)
ngram_model.train(wiki_train)

# generate sentences | calculate their probabilities
num_sentences = 1000
min_sentence_length = 10

sentences_log_probs = []
sentences_log_probs_sm = []
sentences_log_probs_interp = []
sentences_log_probs_ad = []

lambdas = (0.3, 0.3, 0.4) # test different lambdas

for _ in range(num_sentences):
    generated_sentence = ngram_model.generate().split()
    
    if len(generated_sentence) > min_sentence_length:
        log_prob = ngram_model.sentence_log_probability(generated_sentence)
        sentences_log_probs.append((generated_sentence, log_prob))

        log_prob_smoothed = ngram_model.sentence_log_probability_smoothed(generated_sentence)
        sentences_log_probs_sm.append((generated_sentence, log_prob_smoothed))
        
        log_prob_ad = ngram_model.sentence_log_probability_smoothed_absolute_discounting(generated_sentence)
        sentences_log_probs_ad.append((generated_sentence, log_prob_ad))

        log_prob_interp = ngram_model.sentence_interpolated_logP(generated_sentence, lambdas=lambdas)
        sentences_log_probs_interp.append((generated_sentence, log_prob_interp))


# highest probability
highest_log_prob_sentence, highest_log_prob = max(sentences_log_probs, key=lambda x: x[1])
highest_log_prob_sentence_sm, highest_log_prob_sm = max(sentences_log_probs_sm, key=lambda x: x[1])
highest_log_prob_sentence_ad, highest_log_prob_ad = max(sentences_log_probs_ad, key=lambda x: x[1])
highest_log_prob_sentence_interp, highest_log_prob_interp = max(sentences_log_probs_interp, key=lambda x: x[1])

print("\nSentence with the highest probability (non-smoothed):", ' '.join(highest_log_prob_sentence))
print("Log probability (non-smoothed):", highest_log_prob)

print("\nSentence with the highest probability (smoothed):", ' '.join(highest_log_prob_sentence_sm))
print("Log probability (smoothed):", highest_log_prob_sm)

print("\nSentence with the highest probability (Absolute Discounting):", ' '.join(highest_log_prob_sentence_ad))
print("Log probability (Absolute Discounting):", highest_log_prob_ad)

print("\nSentence with the highest probability (interpolated):", ' '.join(highest_log_prob_sentence_interp))
print("Log probability (interpolated):", highest_log_prob_interp)

#############################################

# train N-gram model
ngram_model = NgramModel(n=3, smoothing_factor=1)
ngram_model.train(wiki_train)

ngrams_log_probs = ngram_model.all_ngrams_log_probabilities()
ngrams_log_probs_sorted = sorted(ngrams_log_probs, key=lambda x: x[1], reverse=True)

top_ngrams = ngrams_log_probs_sorted[:3]
for ngram, log_prob in top_ngrams:
    print(f"{' '.join(ngram)}: {log_prob}")

print("\n")
top_ngrams = ngrams_log_probs_sorted[15_900:15_905]
for ngram, log_prob in top_ngrams:
    print(f"{' '.join(ngram)}: {log_prob}")

print("\n")
random_ngrams = random.sample(ngrams_log_probs, 5)
for ngram, log_prob in random_ngrams:
    print(f"{' '.join(ngram)}: {log_prob}")

#############################################

print("########## Calculate any probability of any sentence ##########")

input_sentence = "This is an example sentence to calculate its probability."
tokenized_input_sentence = word_tokenize(input_sentence)

log_prob_input_sentence = ngram_model.sentence_log_probability(tokenized_input_sentence)
log_prob_input_sentence_smoothed = ngram_model.sentence_log_probability_smoothed(tokenized_input_sentence)
log_prob_input_sentence_ad = ngram_model.sentence_log_probability_smoothed_absolute_discounting(tokenized_input_sentence)
log_prob_input_sentence_interp = ngram_model.sentence_interpolated_logP(tokenized_input_sentence, lambdas=lambdas)

print("\nInput sentence:", input_sentence)

print("\nLog probability (non-smoothed):", log_prob_input_sentence)
print("Log probability (smoothed):", log_prob_input_sentence_smoothed)
print("Log probability (Absolute Discounting):", log_prob_input_sentence_ad)
print("Log probability (interpolated):", log_prob_input_sentence_interp)

#############################################

print("########## CHECK THE SANITY OF THE CODE BY HAND ##########")

ngram_model = NgramModel(n=3, smoothing_factor=0.01)
ngram_model.train(wiki_train)

sat_prob = ngram_model.log_probability("my", "elephant", "sat")
slept_prob = ngram_model.log_probability("my", "unicorn", "slept")

print("Log probability of ('my', 'elephant', 'sat'):", sat_prob)
print("Log probability of ('my', 'unicorn', 'slept'):", slept_prob)

print("\n")

sat_prob = ngram_model.probability("my", "elephant", "sat")
slept_prob = ngram_model.probability("my", "unicorn", "slept")

print("Probability of ('my', 'elephant', 'sat'):", sat_prob)
print("Probability of ('my', 'unicorn', 'slept'):", slept_prob)

#############################################

print("########## COUNTS AND TOP N-GRAMS ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(wiki_train)

ngram_model.ngram_count()

# ngram_model.top_ngrams(K=5)
ngram_model.top_ngrams()

# top 3 n-grams and probabilities
ngrams_log_probs = ngram_model.all_ngrams_log_probabilities()
ngrams_log_probs_sorted = sorted(ngrams_log_probs, key=lambda x: x[1], reverse=True)

top_3_ngrams = ngrams_log_probs_sorted[:3]

print("Top 3 n-grams with log probabilities:")
for ngram, log_prob in top_3_ngrams:
    print(f"{ngram}: {log_prob}")

# generate 
generated_sequence = ngram_model.generate()
print("\nGenerated sequence:", generated_sequence)

#############################################
#############################################
#############################################

print("""
=============================================
                TEST
=============================================

A - REGULAR PERPLEXITY AND PROBABILITIES

=============================================

1.1. Train on Wiki

        - Test on Tweet
        - Test on IMDB
        
1.2. Train on Tweet

        - Test on Wiki
        - Test on IMDB

1.3. Train on IMDB

        - Test on Wiki
        - Test on Tweet
""")

print("########## 1.1. WIKI DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(wiki_train)

# Numbers of each n-gram
ngram_model.ngram_count()

# n-gram and probabilities
ngram_model.top_ngrams()

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity(wiki_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(tweet_test)
print("\nPerplexity on TWEET test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(imdb_test)
print("\nPerplexity on IMDB test set:", test_perplexity)

print("########## 1.2. TWEET DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(tweet_train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity(tweet_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(wiki_test)
print("\nPerplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(imdb_test)
print("\nPerplexity on IMDB test set:", test_perplexity)ç

print("########## 1.3. IMDB DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(imdb_train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity(imdb_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(wiki_test)
print("\nPerplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity(tweet_test)
print("\nPerplexity on TWEET test set:", test_perplexity)

print("""
=============================================
                TEST
=============================================

B - SMOOTHED PERPLEXITY AND PROBABILITIES

=============================================

1.1. Train on Wiki

        - Test on Tweet
        - Test on IMDB
        
1.2. Train on Tweet

        - Test on Wiki
        - Test on IMDB

1.3. Train on IMDB

        - Test on Wiki
        - Test on Tweet
""")

print("########## 1.1. WIKI DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(wiki_train)

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# alpha value for smoothing
alpha = 1

# eval on the dev
dev_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_dev, alpha=alpha)
print("\nSmoothed Perplexity on dev set:", dev_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_test, alpha=alpha)
print("\nSmoothed Perplexity on TWEET test set:", test_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_test, alpha=alpha)
print("\nSmoothed Perplexity on IMDB test set:", test_perplexity_smoothed)

print("########## 1.2. TWEET DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(tweet_train)

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# alpha value for smoothing
alpha = 1

# eval on the dev
dev_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_dev, alpha=alpha)
print("\nSmoothed Perplexity on dev set:", dev_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_test, alpha=alpha)
print("\nSmoothed Perplexity on WIKI test set:", test_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_test, alpha=alpha)
print("\nSmoothed Perplexity on IMDB test set:", test_perplexity_smoothed)

print("########## 1.3. IMDB DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(imdb_train)

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# alpha value for smoothing
alpha = 1

# eval on the dev
dev_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_dev, alpha=alpha)
print("\nSmoothed Perplexity on dev set:", dev_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_test, alpha=alpha)
print("\nSmoothed Perplexity on WIKI test set:", test_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_test, alpha=alpha)
print("\nSmoothed Perplexity on TWEET test set:", test_perplexity_smoothed)

print("""
=============================================
                TEST
=============================================

C - SMOOTHED ABSOLUTE DISCOUNTIGN PERPLEXITY AND PROBABILITIES

=============================================

1.1. Train on Wiki

        - Test on Tweet
        - Test on IMDB
        
1.2. Train on Tweet

        - Test on Wiki
        - Test on IMDB

1.3. Train on IMDB

        - Test on Wiki
        - Test on Tweet
""")

print("########## 1.1. WIKI DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(wiki_train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(wiki_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(tweet_test)
print("\nPerplexity on TWEET test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(imdb_test)
print("\nPerplexity on IMDB test set:", test_perplexity)

print("########## 1.2. TWEET DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(tweet_train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(tweet_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(wiki_test)
print("\nPerplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(imdb_test)
print("\nPerplexity on IMDB test set:", test_perplexity)

print("########## 1.3. IMDB DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(imdb_train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(imdb_dev)
print("\nPerplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(wiki_test)
print("\nPerplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.perplexity_smoothed_absolute_discounting(tweet_test)
print("\nPerplexity on TWEET test set:", test_perplexity)

print("""
=============================================
                TEST
=============================================

D - INTERPOLATED PERPLEXITY AND PROBABILITIES

=============================================

1.1. Train on Wiki

        - Test on Tweet
        - Test on IMDB
        
1.2. Train on Tweet

        - Test on Wiki
        - Test on IMDB

1.3. Train on IMDB

        - Test on Wiki
        - Test on Tweet
""")

print("########## 1.1. WIKI DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)  # max n-gram value to 6 | to use 1-grams to 5-grams | look up later (maybe 5)
ngram_model.train(wiki_train)
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.interpolated_perplexity(wiki_dev, lambdas)
print("\nInterpolated Perplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(tweet_test, lambdas)
print("\nInterpolated Perplexity on TWEET test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(imdb_test, lambdas)
print("\nInterpolated Perplexity on IMDB test set:", test_perplexity)

print("########## 1.2. TWEET DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)  # max n-gram value to 6 | to use 1-grams to 5-grams | look up later (maybe 5)
ngram_model.train(tweet_train)
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.interpolated_perplexity(tweet_dev, lambdas)
print("\nInterpolated Perplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(wiki_test, lambdas)
print("\nInterpolated Perplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(imdb_test, lambdas)
print("\nInterpolated Perplexity on IMDB test set:", test_perplexity)

print("########## 1.3. IMDB DATA ##########")

# train n-gram model
ngram_model = NgramModel(n=3)  # max n-gram value to 6 | to use 1-grams to 5-grams | look up later (maybe 5)
ngram_model.train(imdb_train)
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.interpolated_perplexity(imdb_dev, lambdas)
print("\nInterpolated Perplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(wiki_test, lambdas)
print("\nInterpolated Perplexity on WIKI test set:", test_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(tweet_test, lambdas)
print("\nInterpolated Perplexity on TWEET test set:", test_perplexity)
