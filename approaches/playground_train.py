##### Trainig #####

corpus = ["I like to eat pizza", "Pizza is my favorite food", "I eat pizza every day", "I also enjoy eating burgers"]
sentence = []

for i in range(len(corpus)):
    for token in corpus[i].split():
        sentence.append(token)
        
#print(sentence)

bigrams = []
bigrams_count = {}
unigrams_count = {}

for i in range(len(sentence) -1):
    if i < len(sentence) - 1: # stop before 
        bigrams.append((sentence[i], sentence[i+1]))
        
        if (sentence[i], sentence[i+1]) in bigrams_count:
            bigrams_count[(sentence[i], sentence[i+1])] += 1
        else:
            bigrams_count[(sentence[i], sentence[i+1])] = 1
            
    if sentence[i] in unigrams_count:
        unigrams_count[sentence[i]] += 1
    else:
        unigrams_count[sentence[i]] = 1

print("\n All unigrams: ", "\n", (sentence))
print("\n All bigrams:", "\n", (bigrams))

print("\n Bigrams frequency:", "\n", (bigrams_count))
print("\n Unigrams frequency:", "\n", (unigrams_count))
