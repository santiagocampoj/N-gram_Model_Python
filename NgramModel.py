import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from collections import defaultdict
import random
from datasets import load_dataset
from itertools import product
import matplotlib.pyplot as plt

nltk.download('punkt')

def read_imdb_data(path):
    data = []
    for label in ['pos', 'neg']:
        for file in os.listdir(os.path.join(path, label)):
            with open(os.path.join(path, label, file), 'r', encoding='utf-8') as f:
                data.append((f.read(), label))
    return data

train_data = read_imdb_data('aclImdb/train')
test_data = read_imdb_data('aclImdb/test')

print(train_data[:1])

def read_tweet_eval_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            data.append(stripped_line)
    return data

tweet_eval_train = read_tweet_eval_data('tweeteval/datasets/emotion/train_text.txt')
tweet_eval_test = read_tweet_eval_data('tweeteval/datasets/emotion/test_text.txt')

print(tweet_eval_train[:1])

def read_wikitext_data(base_path, sub_path):
    path = os.path.join(base_path, sub_path)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            data.append(stripped_line)
    return data

wikitext_train = read_wikitext_data(path, 'wikitext-103/wiki.train.tokens')
wikitext_test = read_wikitext_data(path, 'wikitext-103/wiki.test.tokens')

#############################################
# PREPROCESS DATA
#############################################

# WIKI
# only the first 1000 articles
wiki_text = wikitext_train[:1000]

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

# print(len(wiki_train))

#############################################

# TWEET
# Tokenize tweet_eval_train
tweet_eval_train_tokenized = []
for text in tweet_eval_train:
    tokenized_text = word_tokenize(text)
    tweet_eval_train_tokenized.append(tokenized_text)

# Tokenize tweet_eval_test
tweet_eval_test_tokenized = []
for text in tweet_eval_test:
    tokenized_text = word_tokenize(text)
    tweet_eval_test_tokenized.append(tokenized_text)

# Split train data into train and dev
tweet_eval_dev_idx = int(len(tweet_eval_train_tokenized) * .7)
tweet_train = tweet_eval_train_tokenized[:tweet_eval_dev_idx]
tweet_dev = tweet_eval_train_tokenized[tweet_eval_dev_idx:]

# Use the whole test data
tweet_test = tweet_eval_test_tokenized

# print(len(tweet_train))

#############################################

# IMDB
imdb_data = train_data[:1000] + test_data[:1000]

# Tokenize sentences
imdb_tokenized_sentences = []
for text, label in imdb_data:
    tokenized_text = word_tokenize(text)
    imdb_tokenized_sentences.append(tokenized_text)

# Split into train, dev, and test
imdb_dev_idx = int(len(imdb_tokenized_sentences) * .7)
imdb_test_idx = int(len(imdb_tokenized_sentences) * .8)
imdb_train = imdb_tokenized_sentences[:imdb_dev_idx]
imdb_dev = imdb_tokenized_sentences[imdb_dev_idx:imdb_test_idx]
imdb_test = imdb_tokenized_sentences[imdb_test_idx:]

# print(imdb_train[:1])
# print(len(imdb_train))

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts_dict = {}
        self.vocab = None
    
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
        ngram_counts = []
        for i in range(1, self.n + 1):
            ngram_count = 0
            for ngram_dict in self.counts_dict[i].values():
                if isinstance(ngram_dict, dict):
                    ngram_count += len(ngram_dict)
                else:
                    ngram_count += 1
            ngram_counts.append(ngram_count)
        return ngram_counts

    # TOP N-GRAMS
    def top_ngrams(self, K=5):
        top_ngrams_results = {}
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

            top_ngrams_with_log_probs = []
            for ngram, _ in top_k_ngrams:
                logP = self.log_probability(*ngram) if n > 1 else self.log_probability(ngram[0])
                top_ngrams_with_log_probs.append((ngram, logP))
            
            top_ngrams_results[n] = top_ngrams_with_log_probs
        return top_ngrams_results

    ##################################################
    # GENERATE SENTENCE
    ##################################################
    
    def sample_next_word(self, *prev_words):  # unpack and pass them
        n = len(prev_words) + 1
        prev_words = prev_words[-(n - 1):]

        if n == 1:
            keys, values = zip(*self.counts_dict[n].items())
        else:
            keys, values = zip(*self.counts_dict[n][tuple(prev_words)].items())  # unpack and pass them

        values = np.array(values, dtype=np.float64)  # convert values to float64 otherwise it crashes
        values /= values.sum()

        return keys[np.argmax(np.random.multinomial(1, values))]

    def generate(self):
        if self.n == 1:
            result = []
        else:
            result = ['*'] * (self.n - 1)

        while True:
            if self.n == 1:
                prev_words = tuple()
            else:
                prev_words = tuple(result[-(self.n - 1):])
            next_word = self.sample_next_word(*prev_words) 

            if next_word == 'STOP':
                break

            result.append(next_word)

        if self.n == 1:
            return ' '.join(result)
        else:
            return ' '.join(result[self.n - 1:])

    ##################################################
    # REGULAR PERPLEXITY AND PROBABILITIES
    ##################################################

    def probability(self, *ngram):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
        if n == 1:
            numerator = self.counts_dict[n][word] + self.smoothing_factor
            denominator = sum(self.counts_dict[n].values()) + self.smoothing_factor * len(self.vocab)
        else:
            numerator = self.counts_dict[n][tuple(prefix)].get(word, 0) + self.smoothing_factor
            denominator = sum(self.counts_dict[n][tuple(prefix)].values()) + self.smoothing_factor * len(self.vocab)

        return numerator / denominator

    def log_probability(self, *ngram):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
        if n == 1:
            numerator = self.counts_dict[n][word]  # no smoothing_factor
            denominator = sum(self.counts_dict[n].values())
        else:
            numerator = self.counts_dict[n][tuple(prefix)][word]  # no smoothing_factor
            denominator = sum(self.counts_dict[n][tuple(prefix)].values())
        
        return np.log2(numerator / denominator)

    def sentence_log_probability(self, sentence):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0

        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.log_probability(*ngram)

        return logP
    
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

    # PERPLEXITY
    def perplexity(self, corpus):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability(sentence)
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
    
    def log_probability_smoothed_absolute_discounting(self, *ngram, alpha=0.01, discount=0.75):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]

        if n == 1:
            numerator = max(self.counts_dict[n][word] - discount, 0) + alpha
            denominator = sum(self.counts_dict[n].values()) - discount * len(self.vocab) + alpha * len(self.vocab)
        
        else:
            numerator = max(self.counts_dict[n][tuple(prefix)][word] - discount, 0) + alpha
            denominator = sum(self.counts_dict[n][tuple(prefix)].values()) - discount * len(self.vocab) + alpha * len(self.vocab)

        # print(f"numerator: {numerator}, denominator: {denominator}") 

        return np.log2(numerator / denominator)  # return the logarithm of probability


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

######################################################
######################################################
######################################################

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

# ######################################################
# ######################################################
# ######################################################

print("########## N-GRAMS AND THEIR LOG_PROBABILITIES ##########")

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
    
    
# ######################################################
# ######################################################
# ######################################################

print("########## N-GRAMS AND THEIR LOG_PROBABILITIES ##########")

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

# ######################################################
# ######################################################
# ######################################################

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
    
# ######################################################
# ######################################################
# ######################################################

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
    
# ######################################################
# ######################################################
# ######################################################

print("########## COUNTS AND TOP N-GRAMS ##########")

# train n-gram model
ngram_model = NgramModel(n=1)
ngram_model.train(wiki_train)

ngram_model.ngram_count()
ngram_model.top_ngrams()

# generate 
generated_sequence = ngram_model.generate()
print("\nGenerated sequence:", generated_sequence)
    
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

ngram_sizes = range(1, 6)  # from 1-gram to 5-gram
results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n, smoothing_factor=0.01)
    ngram_model.train(wiki_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # on the wiki_dev dataset
    dev_perplexity = ngram_model.perplexity(wiki_dev)
    print("\nPerplexity on dev set:", dev_perplexity)

    # on the tweet_test dataset
    tweet_test_perplexity = ngram_model.perplexity(tweet_test)
    print("Perplexity on TWEET test set:", tweet_test_perplexity)

    # on the imdb_test dataset
    imdb_test_perplexity = ngram_model.perplexity(imdb_test)
    print("Perplexity on IMDB test set:", imdb_test_perplexity)

    # top 5 n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = []
    for i in range(1, n + 1):
        ngram_count = 0
        for ngram_dict in ngram_model.counts_dict[i].values():
            if isinstance(ngram_dict, dict):
                ngram_count += len(ngram_dict)
            else:
                ngram_count += 1
        ngram_counts.append(ngram_count)

    results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs[n],
        'dev': dev_perplexity,
        'tweet_test': tweet_test_perplexity,
        'imdb_test': imdb_test_perplexity,
        'ngram_counts': ngram_counts,  # key 
    }
    
# Plot
def plot_ngram_counts(results):
    ngram_sizes = list(results.keys())
    max_n = max(ngram_sizes)

    colors = plt.cm.viridis(np.linspace(0, 1, max_n))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))

    for n in ngram_sizes:
        ngram_counts = results[n]['ngram_counts']

        for i, count in enumerate(ngram_counts):
            if n == ngram_sizes[0]:
                label = f"{i + 1}-grams"
            else:
                label = None
            bar_position = n - 1 + i * bar_width
            bar = ax.bar(bar_position, count, color=colors[i], width=bar_width, label=label)
            ax.text(bar_position, count + 0.01, str(count), ha='center', va='bottom')

    ax.set_ylabel('Number of N-grams')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Number of N-grams for each N-gram model')
    
    x_tick_positions = []
    for x in range(len(ngram_sizes)):
        x_tick_position = x + bar_width * (max_n - 1) / 2
        x_tick_positions.append(x_tick_position)
    ax.set_xticks(x_tick_positions)
    
    xticklabels = []
    for n in ngram_sizes:
        label = f"{n}-gram"
        xticklabels.append(label)

    ax.set_xticklabels(xticklabels)

    legend_labels = []
    for i in range(max_n):
        legend_label = f"{i + 1}-grams"
        legend_labels.append(legend_label)

    legend_colors = []
    for i in range(max_n):
        legend_color = colors[i]
        legend_colors.append(legend_color)

    legend_handles = []
    for legend_color in legend_colors:
        legend_handle = plt.Rectangle((0, 0), 1, 1, color=legend_color)
        legend_handles.append(legend_handle)

    ax.legend(handles=legend_handles, labels=legend_labels)
    
    plt.tight_layout()
    plt.show()

plot_ngram_counts(results)   

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'tweet_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, tweet_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(results)

##########################################

print("########## 1.2. TWEET DATA ##########")

ngram_sizes = range(1, 6)  # from 1-gram to 5-gram
tweet_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n, smoothing_factor=0.01)
    ngram_model.train(tweet_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # on the tweet_dev dataset
    dev_perplexity = ngram_model.perplexity(tweet_dev)
    print("\nPerplexity on dev set:", dev_perplexity)

    # on the wiki_test dataset
    wiki_test_perplexity = ngram_model.perplexity(wiki_test)
    print("Perplexity on WIKI test set:", wiki_test_perplexity)

    # on the imdb_test dataset
    imdb_test_perplexity = ngram_model.perplexity(imdb_test)
    print("Perplexity on IMDB test set:", imdb_test_perplexity)

    # top 5 n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = []
    for i in range(1, n + 1):
        ngram_count = 0
        for ngram_dict in ngram_model.counts_dict[i].values():
            if isinstance(ngram_dict, dict):
                ngram_count += len(ngram_dict)
            else:
                ngram_count += 1
        ngram_counts.append(ngram_count)

    tweet_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs[n],
        'dev': dev_perplexity,
        'wiki_test': wiki_test_perplexity,
        'imdb_test': imdb_test_perplexity,
        'ngram_counts': ngram_counts,
    }

# plot
plot_ngram_counts(tweet_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(tweet_results)


##########################################

print("########## 1.3. IMDB DATA ##########")

ngram_sizes = range(1, 6)  # from 1-gram to 5-gram
imdb_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n, smoothing_factor=0.01)
    ngram_model.train(imdb_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # on the imdb_dev dataset
    dev_perplexity = ngram_model.perplexity(imdb_dev)
    print("\nPerplexity on dev set:", dev_perplexity)

    # on the wiki_test dataset
    wiki_test_perplexity = ngram_model.perplexity(wiki_test)
    print("Perplexity on WIKI test set:", wiki_test_perplexity)

    # on the tweet_test dataset
    tweet_test_perplexity = ngram_model.perplexity(tweet_test)
    print("Perplexity on TWEET test set:", tweet_test_perplexity)

    # top 5 n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = []
    for i in range(1, n + 1):
        ngram_count = 0
        for ngram_dict in ngram_model.counts_dict[i].values():
            if isinstance(ngram_dict, dict):
                ngram_count += len(ngram_dict)
            else:
                ngram_count += 1
        ngram_counts.append(ngram_count)

    imdb_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs[n],
        'dev': dev_perplexity,
        'wiki_test': wiki_test_perplexity,
        'tweet_test': tweet_test_perplexity,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(imdb_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'tweet_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and tweet_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(imdb_results)

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

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
smooth_wiki_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(wiki_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the wiki_dev dataset
    dev_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_dev, alpha=alpha)
    print("\nSmoothed Perplexity on dev set:", dev_perplexity_smoothed)

    # eval on the tweet_test dataset
    tweet_test_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_test, alpha=alpha)
    print("Smoothed Perplexity on TWEET test set:", tweet_test_perplexity_smoothed)

    # eval on the imdb_test dataset
    imdb_test_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_test, alpha=alpha)
    print("Smoothed Perplexity on IMDB test set:", imdb_test_perplexity_smoothed)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    smooth_wiki_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity_smoothed,
        'tweet_test': tweet_test_perplexity_smoothed,
        'imdb_test': imdb_test_perplexity_smoothed,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(smooth_wiki_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'tweet_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, tweet_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(smooth_wiki_results)

########################################################

print("########## 1.2. TWEET DATA ##########")

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
smooth_tweet_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(tweet_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the tweet_dev dataset
    dev_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_dev, alpha=alpha)
    print("\nSmoothed Perplexity on TWEET dev set:", dev_perplexity_smoothed)

    # eval on the wiki_test dataset
    wiki_test_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_test, alpha=alpha)
    print("Smoothed Perplexity on WIKI test set:", wiki_test_perplexity_smoothed)

    # eval on the imdb_test dataset
    imdb_test_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_test, alpha=alpha)
    print("Smoothed Perplexity on IMDB test set:", imdb_test_perplexity_smoothed)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    smooth_tweet_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity_smoothed,
        'wiki_test': wiki_test_perplexity_smoothed,
        'imdb_test': imdb_test_perplexity_smoothed,
        'ngram_counts': ngram_counts,
    }

# PLOT
plot_ngram_counts(smooth_tweet_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(smooth_tweet_results)

########################################################

print("########## 1.3. IMDB DATA ##########")

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
smooth_imdb_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(imdb_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the imdb_dev dataset
    dev_perplexity_smoothed = ngram_model.perplexity_smoothed(imdb_dev, alpha=alpha)
    print("\nSmoothed Perplexity on IMDB dev set:", dev_perplexity_smoothed)

    # eval on the wiki_test dataset
    wiki_test_perplexity_smoothed = ngram_model.perplexity_smoothed(wiki_test, alpha=alpha)
    print("Smoothed Perplexity on WIKI test set:", wiki_test_perplexity_smoothed)

    # eval on the tweet_test dataset
    tweet_test_perplexity_smoothed = ngram_model.perplexity_smoothed(tweet_test, alpha=alpha)
    print("Smoothed Perplexity on TWEET test set:", tweet_test_perplexity_smoothed)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    smooth_imdb_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity_smoothed,
        'wiki_test': wiki_test_perplexity_smoothed,
        'tweet_test': tweet_test_perplexity_smoothed,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(smooth_imdb_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'tweet_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and tweet_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(smooth_imdb_results)

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

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
disc_wiki_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(wiki_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the dev
    dev_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(wiki_dev)
    print("\nAbsolute Discounting Perplexity on dev set", dev_perplexity_disc)

    # eval on the test
    tweet_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(tweet_test)
    print("\nAbsolute Discounting Perplexity on test set:", tweet_test_perplexity_disc)

    # eval on the test
    imdb_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(imdb_test)
    print("\nAbsolute Discounting Perplexity on test set:", imdb_test_perplexity_disc)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    disc_wiki_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity_disc,
        'tweet_test': tweet_test_perplexity_disc,
        'imdb_test': imdb_test_perplexity_disc,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(disc_wiki_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'tweet_test': 'green', 'imdb_test': 'red'}  # 'dev' color added
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, tweet_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(disc_wiki_results)

###############################################

print("########## 1.2. TWEET DATA ##########")

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
disc_tweet_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(wiki_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the dev
    dev_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(tweet_dev)
    print("\nAbsolute Discounting Perplexity on dev set", dev_perplexity_disc)

    # eval on the test
    wiki_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(wiki_test)
    print("\nAbsolute Discounting Perplexity on test set:", tweet_test_perplexity_disc)

    # eval on the test
    imdb_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(imdb_test)
    print("\nAbsolute Discounting Perplexity on test set:", imdb_test_perplexity_disc)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    disc_tweet_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity_disc,
        'wiki_test': wiki_test_perplexity_disc,
        'imdb_test': imdb_test_perplexity_disc,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(disc_tweet_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'imdb_test': 'red'} 
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(disc_tweet_results)

###########################################################

print("########## 1.3. IMDB DATA ##########")

ngram_sizes = range(1, 6)  # from 1-gram to 5-gram
disc_imdb_results = {}

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(imdb_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # alpha value for smoothing
    alpha = 0.01

    # eval on the dev
    dev_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(imdb_dev)
    print("\nAbsolute Discounting Perplexity on dev set", dev_perplexity_disc)

    # eval on the test
    wiki_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(wiki_test)
    print("\nAbsolute Discounting Perplexity on test set:", wiki_test_perplexity_disc)

    # eval on the test
    tweet_test_perplexity_disc = ngram_model.perplexity_smoothed_absolute_discounting(tweet_test)
    print("\nAbsolute Discounting Perplexity on test set:", tweet_test_perplexity_disc)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    disc_imdb_results[n] = {
      'top_ngrams_and_probs': top_ngrams_and_probs,
      'dev': dev_perplexity_disc,
      'wiki_test': wiki_test_perplexity_disc,
      'tweet_test': tweet_test_perplexity_disc,  
      'ngram_counts': ngram_counts,
}

    plot_ngram_counts(disc_imdb_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'tweet_test': 'red'}  
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and tweet_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(disc_imdb_results)

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

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
interpolated_wiki_results = {}
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(wiki_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # eval on the wiki_dev dataset
    dev_perplexity = ngram_model.interpolated_perplexity(wiki_dev, lambdas)
    print("\nInterpolated Perplexity on dev set:", dev_perplexity)

    # eval on the tweet_test dataset
    tweet_test_perplexity = ngram_model.interpolated_perplexity(tweet_test, lambdas)
    print("Interpolated Perplexity on TWEET test set:", tweet_test_perplexity)

    # eval on the imdb_test dataset
    imdb_test_perplexity = ngram_model.interpolated_perplexity(imdb_test, lambdas)
    print("Interpolated Perplexity on IMDB test set:", imdb_test_perplexity)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    interpolated_wiki_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity,
        'tweet_test': tweet_test_perplexity,
        'imdb_test': imdb_test_perplexity,
        'ngram_counts': ngram_counts,
    }

    plot_ngram_counts(interpolated_wiki_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'tweet_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, tweet_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(interpolated_wiki_results)

print("########## 1.2. TWEET DATA ##########")

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
interpolated_tweet_results = {}
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(tweet_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # eval on the tweet_dev dataset
    dev_perplexity = ngram_model.interpolated_perplexity(tweet_dev, lambdas)
    print("\nInterpolated Perplexity on dev set:", dev_perplexity)

    # eval on the wiki_test dataset
    wiki_test_perplexity = ngram_model.interpolated_perplexity(wiki_test, lambdas)
    print("Interpolated Perplexity on WIKI test set:", wiki_test_perplexity)

    # eval on the imdb_test dataset
    imdb_test_perplexity = ngram_model.interpolated_perplexity(imdb_test, lambdas)
    print("Interpolated Perplexity on IMDB test set:", imdb_test_perplexity)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    interpolated_tweet_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity,
        'wiki_test': wiki_test_perplexity,
        'imdb_test': imdb_test_perplexity,
        'ngram_counts': ngram_counts,
    }

    plot_ngram_counts(interpolated_tweet_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'imdb_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and imdb_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(interpolated_tweet_results)

print("########## 1.3. IMDB DATA ##########")

ngram_sizes = range(1, 6)  # from 2-gram to 5-gram
interpolated_imdb_results = {}
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

for n in ngram_sizes:
    print(f"\n\n========== Evaluating {n}-gram model ==========")

    # train 
    ngram_model = NgramModel(n=n)
    ngram_model.train(imdb_train)

    # generate
    generated_sequence = ngram_model.generate()
    print("\nGenerated sequence:", generated_sequence)

    # eval on the imdb_dev dataset
    dev_perplexity = ngram_model.interpolated_perplexity(imdb_dev, lambdas)
    print("\nInterpolated Perplexity on dev set:", dev_perplexity)

    # eval on the wiki_test dataset
    wiki_test_perplexity = ngram_model.interpolated_perplexity(wiki_test, lambdas)
    print("Interpolated Perplexity on WIKI test set:", wiki_test_perplexity)

    # eval on the tweet_test dataset
    tweet_test_perplexity = ngram_model.interpolated_perplexity(tweet_test, lambdas)
    print("Interpolated Perplexity on TWEET test set:", tweet_test_perplexity)

    # top n-grams with log probabilities
    top_ngrams_and_probs = ngram_model.top_ngrams(K=5)

    # n-gram count
    ngram_counts = ngram_model.ngram_count()

    interpolated_imdb_results[n] = {
        'top_ngrams_and_probs': top_ngrams_and_probs,
        'dev': dev_perplexity,
        'wiki_test': wiki_test_perplexity,
        'tweet_test': tweet_test_perplexity,
        'ngram_counts': ngram_counts,
    }

plot_ngram_counts(interpolated_imdb_results)

def plot_results(results):
    ngram_sizes = list(results.keys())

    # colors for each dataset
    colors = {'dev': 'blue', 'wiki_test': 'green', 'tweet_test': 'red'}
    datasets = list(colors.keys())

    # width of a bar 
    bar_width = 0.25

    # position of bars on x-axis
    index = np.arange(len(ngram_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        perplexities = []
        for n in ngram_sizes:

            perplexity = results[n][dataset]
            perplexities.append(perplexity)

        bars = ax.bar(index + i * bar_width, perplexities, color=colors[dataset], width=bar_width, label=dataset)

        # perplexity values above each barplot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xlabel('N-gram Model')
    ax.set_title('Perplexity on dev, wiki_test, and tweet_test sets for different n-gram models')
    ax.set_xticks(index + bar_width)

    xticklabels = []
    for n in ngram_sizes:
        xticklabels.append(f"{n}-gram")
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # y-axis to log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

plot_results(interpolated_imdb_results)
