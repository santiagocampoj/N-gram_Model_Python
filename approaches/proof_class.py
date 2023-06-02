# We create the class to get the ngrams from 1 to 5. We just take the previous code and convert it to a class called ngram_model. This class 
# is expecting a sentence and any n-gran function the sentence parsed.

class ngram_model:
    def __init__(self, sentence):
        self.sentence = sentence
        self.words = sentence.split()
    
    def unigrams(self, words):
        return self.words
    
    def bigrams(self, words):
        bigrams = []
        for i in range(len(self.words) - 1):
            bigram = (self.words[i], self.words[i+1])
            bigrams.append(bigram)
        return bigrams
    
    def trigrams(self, words):
        trigrams = []
        for i in range(len(self.words) - 2):
            trigram = (self.words[i], self.words[i+1], self.words[i+2])
            trigrams.append(trigram)
        return trigrams
    
    def fourgrams(self, words):
        fourgrams = []
        for i in range(len(self.words) - 3):
            fourgram = (self.words[i], self.words[i+1], self.words[i+2], self.words[i+3])
            fourgrams.append(fourgram)
        return fourgrams
    
    def fivegrams(self, words):
        fivegrams = []
        for i in range(len(self.words) - 4):
            fivegram = (self.words[i], self.words[i+1], self.words[i+2], self.words[i+3], self.words[i+4])
            fivegrams.append(fivegram)
        return fivegrams

sentence = "I am gonna code a unigram, bigram, trigram, fourgram and fivegram and I am not sure about how to write four and five."
model = ngram_model(sentence)


print("\nUnigram",model.unigrams(model.words))
print("\nBigram",model.bigrams(model.words))
print("\ntrigram",model.trigrams(model.words))
print("\nFourigram",model.fourgrams(model.words))
print("\nFivegram",model.fivegrams(model.words))

# we get this output

# Unigram ['I', 'am', 'gonna', 'code', 'a', 'unigram,', 'bigram,', 'trigram,', 'fourgram', 'and', 'fivegram', 'and', 'I', 'am', 'not', 'sure', 'about', 'how', 'to', 'write', 'four', 'and', 'five.']

# Bigram [('I', 'am'), ('am', 'gonna'), ('gonna', 'code'), ('code', 'a'), ('a', 'unigram,'), ('unigram,', 'bigram,'), ('bigram,', 'trigram,'), ('trigram,', 'fourgram'), ('fourgram', 'and'), ('and', 'fivegram'), ('fivegram', 'and'), ('and', 'I'), ('I', 'am'), ('am', 'not'), ('not', 'sure'), ('sure', 'about'), ('about', 'how'), ('how', 'to'), ('to', 'write'), ('write', 'four'), ('four', 'and'), ('and', 'five.')]

# trigram [('I', 'am', 'gonna'), ('am', 'gonna', 'code'), ('gonna', 'code', 'a'), ('code', 'a', 'unigram,'), ('a', 'unigram,', 'bigram,'), ('unigram,', 'bigram,', 'trigram,'), ('bigram,', 'trigram,', 'fourgram'), ('trigram,', 'fourgram', 'and'), ('fourgram', 'and', 'fivegram'), ('and', 'fivegram', 'and'), ('fivegram', 'and', 'I'), ('and', 'I', 'am'), ('I', 'am', 'not'), ('am', 'not', 'sure'), ('not', 'sure', 'about'), ('sure', 'about', 'how'), ('about', 'how', 'to'), ('how', 'to', 'write'), ('to', 'write', 'four'), ('write', 'four', 'and'), ('four', 'and', 'five.')]

# Fourigram [('I', 'am', 'gonna', 'code'), ('am', 'gonna', 'code', 'a'), ('gonna', 'code', 'a', 'unigram,'), ('code', 'a', 'unigram,', 'bigram,'), ('a', 'unigram,', 'bigram,', 'trigram,'), ('unigram,', 'bigram,', 'trigram,', 'fourgram'), ('bigram,', 'trigram,', 'fourgram', 'and'), ('trigram,', 'fourgram', 'and', 'fivegram'), ('fourgram', 'and', 'fivegram', 'and'), ('and', 'fivegram', 'and', 'I'), ('fivegram', 'and', 'I', 'am'), ('and', 'I', 'am', 'not'), ('I', 'am', 'not', 'sure'), ('am', 'not', 'sure', 'about'), ('not', 'sure', 'about', 'how'), ('sure', 'about', 'how', 'to'), ('about', 'how', 'to', 'write'), ('how', 'to', 'write', 'four'), ('to', 'write', 'four', 'and'), ('write', 'four', 'and', 'five.')]

# Fivegram [('I', 'am', 'gonna', 'code', 'a'), ('am', 'gonna', 'code', 'a', 'unigram,'), ('gonna', 'code', 'a', 'unigram,', 'bigram,'), ('code', 'a', 'unigram,', 'bigram,', 'trigram,'), ('a', 'unigram,', 'bigram,', 'trigram,', 'fourgram'), ('unigram,', 'bigram,', 'trigram,', 'fourgram', 'and'), ('bigram,', 'trigram,', 'fourgram', 'and', 'fivegram'), ('trigram,', 'fourgram', 'and', 'fivegram', 'and'), ('fourgram', 'and', 'fivegram', 'and', 'I'), ('and', 'fivegram', 'and', 'I', 'am'), ('fivegram', 'and', 'I', 'am', 'not'), ('and', 'I', 'am', 'not', 'sure'), ('I', 'am', 'not', 'sure', 'about'), ('am', 'not', 'sure', 'about', 'how'), ('not', 'sure', 'about', 'how', 'to'), ('sure', 'about', 'how', 'to', 'write'), ('about', 'how', 'to', 'write', 'four'), ('how', 'to', 'write', 'four', 'and'), ('to', 'write', 'four', 'and', 'five.')]
