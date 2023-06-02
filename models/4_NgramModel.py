import numpy as np
import nltk
from collections import defaultdict

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts_dict = {}
        self.vocab = None
    
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

    
    ##################
    ##################
    ##################
    # I'm not sure about it

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

    ##################
    ##################
    ##################
    

    # PROBABILITIES    
    def log_probability(self, *ngram):
        n = len(ngram)
        prefix, word = ngram[:-1], ngram[-1]
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

    ##################
    # PROBABILITIES SMOOTHED
    ##################

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
    
    ##################
    ##################

    # PERPLEXITY
    def perplexity(self, corpus):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability(sentence)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)

    # PERPLEXITY SMOOTHED
    def perplexity_smoothed(self, corpus, alpha=1):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_log_probability_smoothed(sentence, alpha=alpha)
            token_count += len(sentence)

        return np.exp2(-logP / token_count) 
        
    

    ######################################################
    # now this is the interpolation assignment part:
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

    #INTERPOLATED PERPLEXITY
    def interpolated_perplexity(self, corpus, lambdas=[0.1, 0.2, 0.2, 0.25, 0.25]):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.sentence_interpolated_logP(sentence, lambdas)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)

    ######################################################
    # NEW INTERPOLATED PROBABILITY
    def new_interpolated_log_probability(self, ngram, lambdas):
        interpolated_logP = 0

        max_n = self.n

        for i in range(1, min(len(ngram) + 1, max_n)):
            prefix, word = ngram[:-i], ngram[-i]
            padded_prefix = ('*',) * (i - 1 - len(prefix)) + prefix
            if i == 1:
                logP = self.logP_unigram(word)
            elif i == 2:
                logP = self.logP_bigram(padded_prefix[0], word)
            elif i == 3:
                logP = self.logP_trigram(padded_prefix[0], padded_prefix[1], word)
            elif i == 4:
                logP = self.logP_4gram(padded_prefix[0], padded_prefix[1], padded_prefix[2], word)
            elif i == 5:
                logP = self.logP_5gram(padded_prefix[0], padded_prefix[1], padded_prefix[2], padded_prefix[3], word)
            else:
                raise ValueError("n-gram too long for the implemented n-grams")
            
            interpolated_logP += lambdas[i - 1] * logP

        return interpolated_logP
    
    # NEW INTERPOLATED SENTENCE PROBABILITY
    def new_sentence_interpolated_logP(self, sentence, lambdas):
    sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
    logP = 0

    for i in range(self.n - 1, len(sentence)):
        ngram = tuple(sentence[i - self.n + 1:i + 1])

        interpolated_probability = 0
        
        for j in range(len(lambdas)):
            log_probability = self.new_interpolated_log_probability(ngram, lambdas)
            probability = 2 ** log_probability
            
            weighted_probability = lambdas[j] * probability
            interpolated_probability += weighted_probability

        log_interpolated_probability = np.log2(interpolated_probability)
        
        logP += log_interpolated_probability

    return logP

    
    # NEW INTERPOLATED PERPLEXITY
    def new_interpolated_perplexity(self, corpus, lambdas=[0.1, 0.2, 0.2, 0.25, 0.25]):
        logP = 0
        token_count = 0

        for sentence in corpus:
            logP += self.new_sentence_interpolated_logP(sentence, lambdas)
            token_count += len(sentence)

        return np.exp2(-logP / token_count)


    
###############
###############
###############

# corpus
nltk.download('punkt')

# only the first 1000 articles
wikipedia_text = []
for i in range(1000):
    article = wikipedia_dataset['train'][i]
    text = article['text']
    wikipedia_text.append(text)

# into sentences
sentences = []
for text in wikipedia_text:
    text_sentences = sent_tokenize(text)
    sentences.extend(text_sentences)

# into words
tokenized_sentences = []
for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    tokenized_sentences.append(tokenized_sentence)

sentences = tokenized_sentences

# train, dev, and test sets
dev_idx = int(len(sentences) * .7)
test_idx = int(len(sentences) * .8)
train = sentences[:dev_idx]
dev = sentences[dev_idx:test_idx]
test = sentences[test_idx:]


####
# possible solution for the different corpora

# only the next 1000 articles after the first 1000 and so on
wikipedia_text = []
for i in range(1000, 2000): 
    article = wikipedia_dataset['train'][i]
    text = article['text']
    wikipedia_text.append(text)
####


#######
print("########## SENTENCES GENERATION AND PROBABILITY ##########")

# train N-gram model
ngram_model = NgramModel(n=3, smoothing_factor=1)
ngram_model.train(train)

# generate sentences | calculate their probabilities
num_sentences = 1000
min_sentence_length = 10

sentences_log_probs = []
sentences_log_probs_sm = []
sentences_log_probs_interp = []
sentences_log_probs_new_interp = []

# lambdas = (0.3, 0.3, 0.4) # test different lambdas

for _ in range(num_sentences):
    generated_sentence = ngram_model.generate().split()
    
    if len(generated_sentence) > min_sentence_length:
        log_prob = ngram_model.sentence_log_probability(generated_sentence)
        sentences_log_probs.append((generated_sentence, log_prob))

        log_prob_smoothed = ngram_model.sentence_log_probability_smoothed(generated_sentence)
        sentences_log_probs_sm.append((generated_sentence, log_prob_smoothed))

        log_prob_interp = ngram_model.sentence_interpolated_logP(generated_sentence, lambdas=lambdas)
        sentences_log_probs_interp.append((generated_sentence, log_prob_interp))

        log_prob_new_interp = ngram_model.new_sentence_interpolated_logP(generated_sentence, lambdas=lambdas)
        sentences_log_probs_new_interp.append((generated_sentence, log_prob_new_interp))



# highest probability
highest_log_prob_sentence, highest_log_prob = max(sentences_log_probs, key=lambda x: x[1])
highest_log_prob_sentence_sm, highest_log_prob_sm = max(sentences_log_probs_sm, key=lambda x: x[1])
highest_log_prob_sentence_interp, highest_log_prob_interp = max(sentences_log_probs_interp, key=lambda x: x[1])
highest_log_prob_sentence_new_interp, highest_log_prob_new_interp = max(sentences_log_probs_new_interp, key=lambda x: x[1])


print("\nSentence with the highest probability (non-smoothed):", ' '.join(highest_log_prob_sentence))
print("Log probability (non-smoothed):", highest_log_prob)

print("\nSentence with the highest probability (smoothed):", ' '.join(highest_log_prob_sentence_sm))
print("Log probability (smoothed):", highest_log_prob_sm)

print("\nSentence with the highest probability (interpolated):", ' '.join(highest_log_prob_sentence_interp))
print("Log probability (interpolated):", highest_log_prob_interp)

print("\nSentence with the highest probability (new interpolated):", ' '.join(highest_log_prob_sentence_new_interp))
print("Log probability (new interpolated):", highest_log_prob_new_interp)


print("\n\n\n########## Calculate any probability of any sentence ##########")

input_sentence = "This is an example sentence to calculate its probability."
tokenized_input_sentence = word_tokenize(input_sentence)

log_prob_input_sentence = ngram_model.sentence_log_probability(tokenized_input_sentence)
log_prob_input_sentence_smoothed = ngram_model.sentence_log_probability_smoothed(tokenized_input_sentence)
log_prob_input_sentence_interp = ngram_model.sentence_interpolated_logP(tokenized_input_sentence, lambdas=lambdas)
log_prob_input_sentence_new_interp = ngram_model.new_sentence_interpolated_logP(tokenized_input_sentence, lambdas=lambdas)

print("\nInput sentence:", input_sentence)
print("\nLog probability (non-smoothed):", log_prob_input_sentence)
print("Log probability (smoothed):", log_prob_input_sentence_smoothed)
print("Log probability (interpolated):", log_prob_input_sentence_interp)
print("Log probability (new interpolated):", log_prob_input_sentence_new_interp)


#######


##################
# testing
##################

print("########## REGULAR PERPLEXITY AND PROBABILITIES ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(train)

# generate 
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.perplexity(dev)
print("\nPerplexity on dev set:", dev_perplexity)

# evalon the test
test_perplexity = ngram_model.perplexity(test)
print("\nPerplexity on test set:", test_perplexity)

print("########## SMOOTHED PERPLEXITY AND PROBABILITIES ##########")

# train n-gram model
ngram_model = NgramModel(n=3)
ngram_model.train(train)

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# alpha value for smoothing
alpha = 1

# eval on the dev
dev_perplexity_smoothed = ngram_model.perplexity_smoothed(dev, alpha=alpha)
print("\nSmoothed Perplexity on dev set:", dev_perplexity_smoothed)

# eval on the test
test_perplexity_smoothed = ngram_model.perplexity_smoothed(test, alpha=alpha)
print("\nSmoothed Perplexity on test set:", test_perplexity_smoothed)

print("########## INTERPOLATED PERPLEXITY AND PROBABILITIES ##########")

# train n-gram model
ngram_model = NgramModel(n=6)  # max n-gram value to 6 | to use 1-grams to 5-grams
ngram_model.train(train)
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev
dev_perplexity = ngram_model.interpolated_perplexity(dev, lambdas)
print("\nInterpolated Perplexity on dev set:", dev_perplexity)

# eval on the test
test_perplexity = ngram_model.interpolated_perplexity(test, lambdas)
print("\nInterpolated Perplexity on test set:", test_perplexity)

print("########## NEW INTERPOLATED PERPLEXITY AND PROBABILITIES ##########")

# train n-gram model
ngram_model = NgramModel(n=6)  # max n-gram value to 6 | to use 1-grams to 5-grams
ngram_model.train(train)
lambdas = [0.1, 0.2, 0.2, 0.25, 0.25]  # lambda values for interpolation

# generate
generated_sequence = ngram_model.generate()
print("\n\nGenerated sequence:", generated_sequence)

# eval on the dev 
new_dev_perplexity = ngram_model.new_interpolated_perplexity(dev, lambdas)
print("\nNew Interpolated Perplexity on dev set:", new_dev_perplexity)

# eval on the test
new_test_perplexity = ngram_model.new_interpolated_perplexity(test, lambdas)
print("\nNew Interpolated Perplexity on test set:", new_test_perplexity)
