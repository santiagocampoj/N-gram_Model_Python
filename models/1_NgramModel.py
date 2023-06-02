import nltk
from collections import Counter, defaultdict
import random
from datasets import load_dataset

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts_dict = {}
        self.counts = None
        self.vocab = None
    
    def train(self, corpus):
        self.vocab = set(['*', 'STOP'])
        self.counts = defaultdict(lambda: defaultdict(lambda: self.smoothing_factor))
        
        for sentence in corpus:
            tokens = ['*'] * (self.n-1) + sentence + ['STOP']
            for ngram in nltk.ngrams(tokens, self.n):
                prefix = ngram[:-1]
                word = ngram[-1]
                self.counts[prefix][word] += 1
                self.vocab.add(word)
                if len(prefix) == 1:
                    self.vocab.add(prefix[0])
                elif len(prefix) == 2:
                    self.vocab.add(prefix)
                else:
                    pass
     
    def sample_next_word(self, *prev_words):
        prev_words = prev_words[-(self.n - 1):]
        keys, values = zip(*self.counts[prev_words].items())
        values = np.array(values)
        values /= values.sum()  
        
        return keys[np.argmax(np.random.multinomial(1, values))]
    
    def sample_next_word(self, *prev_words):
        prev_words = prev_words[-(self.n - 1):]
        keys, values = zip(*self.counts[prev_words].items())
        values = np.array(values, dtype=np.float64)  #convert values to float64 otherwise it crashes
        values /= values.sum()
        
        return keys[np.argmax(np.random.multinomial(1, values))]

    
    def generate(self):
        result = ['*'] * (self.n - 1)
        while result[-1] != 'STOP':
            next_word = self.sample_next_word(*result[-(self.n - 1):])
            result.append(next_word)
        
        return ' '.join(result[self.n - 1:-1])
    
    def log_probability(self, *ngram):
        prefix, word = ngram[:-1], ngram[-1]
        numerator = self.counts[prefix][word] + self.smoothing_factor
        denominator = sum(self.counts[prefix].values()) + self.smoothing_factor * len(self.vocab)
        return np.log2(numerator / denominator)
    
    def sentence_log_probability(self, sentence):
        sentence = ['*'] * (self.n - 1) + sentence + ['STOP']
        logP = 0
        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1:i + 1])
            logP += self.log_probability(*ngram)
        
        return logP
    
     def perplexity(self, corpus):
        logP = 0
        for sentence in corpus:
            logP += self.sentence_log_probability(sentence)
       
    return np.exp2(-logP / len(corpus))

    
####
# we check the use of this
####

corpus = [
    "This is a test sentence",
    "Another test sentence is here",
    "The code works fine",
    "Let's see how the model performs",
    "Python programming is fun"
]

tokenized_corpus = []
for sentence in corpus:
    tokenized_sentence = nltk.word_tokenize(sentence)
    tokenized_corpus.append(tokenized_sentence)

# train the model
ngram_model = NgramModel(n=3)
ngram_model.train(tokenized_corpus)

# generation
generated_sequence = ngram_model.generate()
print("Generated sequence:", generated_sequence)

# evaluation corpus
held_out_corpus = [
    "This is another sentence",
    "Testing the model now"
]

# tokenize evaluation corpus
tokenized_held_out_corpus = []
for sentence in held_out_corpus:
    tokenized_sentence = nltk.word_tokenize(sentence)
    tokenized_held_out_corpus.append(tokenized_sentence)

# perplexity
perplexity = ngram_model.perplexity(tokenized_held_out_corpus)
print("Perplexity on held-out data:", perplexity)


##########
# checking with nltk brown corpus
##########

import nltk
from nltk.corpus import brown

nltk.download('brown')


sentences = brown.sents(categories='news')
dev_idx = int(len(sentences) * .7)
test_idx = int(len(sentences) * .8)
train = sentences[:dev_idx]
dev = sentences[dev_idx:test_idx]
test = sentences[test_idx:]

# train
ngram_model = NgramModel(n=3)
ngram_model.train(train)

# generation
generated_sequence = ngram_model.generate()
print("Generated sequence:", generated_sequence)

# dev time
dev_perplexity = ngram_model.perplexity(dev)
print("Perplexity on dev set:", dev_perplexity)

# test time
test_perplexity = ngram_model.perplexity(test)
print("Perplexity on test set:", test_perplexity)


# other way to do it
# train 
ngram_models = NgramModels(n=3)
ngram_models.train(train)

# generate 10 sentences for each n-gram model
generated_sentences = ngram_models.generate_sentences(num_sentences=10)
for n, sentences in generated_sentences.items():
    print(f"Generated sentences for {n}-gram model:")
    for sentence in sentences:
        print(sentence)
    print()

    # sentence
sample_ngram = ('the', 'code')
sample_sentence = ['This', 'is', 'a', 'test', 'sentence']

log_probs = ngram_models.log_probabilities(sample_ngram)
sentence_log_probs = ngram_models.sentence_log_probabilities(sample_sentence)

print("Log probabilities for n-gram:", log_probs)
print("Sentence log probabilities:", sentence_log_probs)

perplexities = ngram_models.perplexities(test)
print("Perplexities on test set:", perplexities)
