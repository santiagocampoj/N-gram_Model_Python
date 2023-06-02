# In this code I am trying to figure out what is the best way to code the n-gram model. To do so, I will be coding step by step 
# what I consider the correct path to follow.

# First to warm up, I easily coded different n-grams from 1 to 5. What it means is get the words and the next one to the right, 
# as many times to the right as neeed (obviously from 1[just the word itself] up to 5).

sentence = "I am gonna code a unigram, bigram, trigram, fourgram and fivegram and I am not sure about how to write four and five."

words = sentence.split()

unigrams = []
for word in words:
    unigrams.append(word)

bigrams = []
for i in range(len(words) -1):
    bigram = (words[i], words[i+1])
    bigrams.append(bigram)

trigrams = []
for i in range(len(words) -2):
    trigram = (words[i], words[i+1], words[i+2])
    trigrams.append(trigram)

fourgrams = []
for i in range(len(words) - 3):
    fourgram = (words[i], words[i+1], words[i+2], words[i+3])
    fourgrams.append(fourgram)

fivegrams = []
for i in range(len(words) - 4):
    fivegram = (words[i], words[i+1], words[i+2], words[i+3], words[i+4])
    fivegrams.append(fivegram)

    
print("\nUnigrams:", unigrams)
print("\nBigrams:", bigrams)
print("\nTrigrams:", trigrams)
print("\nFourgrams:", fourgrams)
print("\nFivegrams:", fivegrams)

# That's what the program outputs:

# Unigrams: ['I', 'am', 'gonna', 'code', 'a', 'unigram,', 'bigram,', 'trigram,', 'fourgram', 'and', 'fivegram', 'and', 'I', 'am', 'not', 'sure', 'about', 'how', 'to', 'write', 'four', 'and', 'five.']

# Bigrams: [('I', 'am'), ('am', 'gonna'), ('gonna', 'code'), ('code', 'a'), ('a', 'unigram,'), ('unigram,', 'bigram,'), ('bigram,', 'trigram,'), ('trigram,', 'fourgram'), ('fourgram', 'and'), ('and', 'fivegram'), ('fivegram', 'and'), ('and', 'I'), ('I', 'am'), ('am', 'not'), ('not', 'sure'), ('sure', 'about'), ('about', 'how'), ('how', 'to'), ('to', 'write'), ('write', 'four'), ('four', 'and'), ('and', 'five.')]

# Trigrams: [('I', 'am', 'gonna'), ('am', 'gonna', 'code'), ('gonna', 'code', 'a'), ('code', 'a', 'unigram,'), ('a', 'unigram,', 'bigram,'), ('unigram,', 'bigram,', 'trigram,'), ('bigram,', 'trigram,', 'fourgram'), ('trigram,', 'fourgram', 'and'), ('fourgram', 'and', 'fivegram'), ('and', 'fivegram', 'and'), ('fivegram', 'and', 'I'), ('and', 'I', 'am'), ('I', 'am', 'not'), ('am', 'not', 'sure'), ('not', 'sure', 'about'), ('sure', 'about', 'how'), ('about', 'how', 'to'), ('how', 'to', 'write'), ('to', 'write', 'four'), ('write', 'four', 'and'), ('four', 'and', 'five.')]

# Fourgrams: [('I', 'am', 'gonna', 'code'), ('am', 'gonna', 'code', 'a'), ('gonna', 'code', 'a', 'unigram,'), ('code', 'a', 'unigram,', 'bigram,'), ('a', 'unigram,', 'bigram,', 'trigram,'), ('unigram,', 'bigram,', 'trigram,', 'fourgram'), ('bigram,', 'trigram,', 'fourgram', 'and'), ('trigram,', 'fourgram', 'and', 'fivegram'), ('fourgram', 'and', 'fivegram', 'and'), ('and', 'fivegram', 'and', 'I'), ('fivegram', 'and', 'I', 'am'), ('and', 'I', 'am', 'not'), ('I', 'am', 'not', 'sure'), ('am', 'not', 'sure', 'about'), ('not', 'sure', 'about', 'how'), ('sure', 'about', 'how', 'to'), ('about', 'how', 'to', 'write'), ('how', 'to', 'write', 'four'), ('to', 'write', 'four', 'and'), ('write', 'four', 'and', 'five.')]

# Fivegrams: [('I', 'am', 'gonna', 'code', 'a'), ('am', 'gonna', 'code', 'a', 'unigram,'), ('gonna', 'code', 'a', 'unigram,', 'bigram,'), ('code', 'a', 'unigram,', 'bigram,', 'trigram,'), ('a', 'unigram,', 'bigram,', 'trigram,', 'fourgram'), ('unigram,', 'bigram,', 'trigram,', 'fourgram', 'and'), ('bigram,', 'trigram,', 'fourgram', 'and', 'fivegram'), ('trigram,', 'fourgram', 'and', 'fivegram', 'and'), ('fourgram', 'and', 'fivegram', 'and', 'I'), ('and', 'fivegram', 'and', 'I', 'am'), ('fivegram', 'and', 'I', 'am', 'not'), ('and', 'I', 'am', 'not', 'sure'), ('I', 'am', 'not', 'sure', 'about'), ('am', 'not', 'sure', 'about', 'how'), ('not', 'sure', 'about', 'how', 'to'), ('sure', 'about', 'how', 'to', 'write'), ('about', 'how', 'to', 'write', 'four'), ('how', 'to', 'write', 'four', 'and'), ('to', 'write', 'four', 'and', 'five.')]
