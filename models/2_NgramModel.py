import numpy as np
import nltk
from collections import defaultdict
from datasets import load_dataset

class NgramModel:
    def __init__(self, n=3, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.counts = None
        self.vocab = None
    
    # TRAIN
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
    # GENERATION
    def sample_next_word(self, *prev_words):
        prev_words = prev_words[-(self.n - 1):]
        keys, values = zip(*self.counts[prev_words].items())
        values = np.array(values, dtype=np.float64)  #convert values to float64 otherwise it crashes
        values /= values.sum()
        
        return keys[np.argmax(np.random.multinomial(1, values))]


    def generate(self):
        result = ['*'] * (self.n - 1)
        
        while result[-1] != 'STOP':
            #next_word = self.sample_next_word(*result[-(self.n - 1):])
            context = result[-(self.n - 1):]
            next_word = self.sample_next_word(*context)
            
            result.append(next_word)
        
        return ' '.join(result[self.n - 1:-1])

    # PROBABILITIES
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

    # PERPLEXITY
    def perplexity(self, corpus):
        logP = 0
       
        for sentence in corpus:
            logP += self.sentence_log_probability(sentence)
        
        return np.exp2(-logP / len(corpus))
    
    
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

# # train
# ngram_models = NgramModels(max_n=3)
# ngram_models.train(train)

# # generate
# generated_sentences = ngram_models.generate_sentences(num_sentences=10)
# for n, sentences in generated_sentences.items():
#     print(f"Generated sentences for {n}-gram model:")
#     for sentence in sentences:
#         print(sentence)
#     print()

# # prob
# sample_ngram = ('the', 'code')
# sample_sentence = ['This', 'is', 'a', 'test', 'sentence']

# log_probs = ngram_models.log_probabilities(sample_ngram)
# sentence_log_probs = ngram_models.sentence_log_probabilities(sample_sentence)

# print("Log probabilities for n-gram:", log_probs)
# print("Sentence log probabilities:", sentence_log_probs)

# # perplexity
# perplexities = ngram_models.perplexities(test)
# print("Perplexities on test set:", perplexities)


ngram_model = NgramModel(n=3)  
ngram_model.train(train)

# generate a sequence
generated_sequence = ngram_model.generate()
print("Generated sequence:", generated_sequence)

# evaluate the model on the dev set
# test the lambdas
#custom_lambdas = [0.05, 0.15, 0.2, 0.25, 0.35]
#dev_perplexity = ngram_model.interpolated_perplexity(dev, lambdas=custom_lambdas)

dev_perplexity = ngram_model.interpolated_perplexity(dev)
print("Interpolated Perplexity on dev set:", dev_perplexity)

# evaluate the model on the test set
test_perplexity = ngram_model.interpolated_perplexity(test)
print("Interpolated Perplexity on test set:", test_perplexity)

