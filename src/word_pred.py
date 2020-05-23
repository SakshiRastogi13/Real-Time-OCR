# -*- coding: utf-8 -*-
import nltk
import re
from collections import OrderedDict
from collections import Counter
import heapq
# Open file and copy content of file


def pred_W(s,f):
    training_data = f.read()

# Cleaning

    cleaned_training_data = re.sub('[^A-Za-z]+', ' ', training_data)
    cleaned_training_data = cleaned_training_data.lower()

# Tokenize training data
    tokens = nltk.word_tokenize(cleaned_training_data)


    words = OrderedDict()
    words= [tokens[i] for i in range(len(tokens)-1)]
    words = tuple(words)

# Making Bigrams
    word_pairs = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    l= len(word_pairs)
    bigrams = set(word_pairs)

# Store probability of second word that come after first word in a corpus 
# of all bigrams

    bigrams_to_cnt_mapping = Counter([(x, y) for x,y in bigrams])
    bigrams_to_cnt_mapping = bigrams_to_cnt_mapping.most_common(l) # it gives bigrams to count dict else if you remove most_common() it will give only bigram
    bigrams_to_cnt_mapping = tuple(bigrams_to_cnt_mapping)
    bigrams_list = [bigram[0] for bigram in bigrams_to_cnt_mapping]
    bigrams_tuple = tuple(bigrams_list)
    bigrams_cnt   =  [bigram[1] for bigram in bigrams_to_cnt_mapping]

    bigrams_first =  [bigram[0] for bigram in bigrams_tuple]
    bigrams_second = [bigram[1] for bigram in bigrams_tuple]

    conditionalprob = OrderedDict()

    for i in range(len(bigrams_to_cnt_mapping)):
        bigram_cnt = bigrams_cnt[i]
        first_word  = bigrams_first[i]
        second_word = bigrams_second[i]
        word_first_cnt = words.count(first_word)
        cond_prob = bigram_cnt / int(word_first_cnt)
        conditionalprob[first_word + " " + second_word] = cond_prob


    user_input =s

# Store bigrams whose first word match with user input  
    matched = []
    for i in range(len(bigrams_to_cnt_mapping)):
        if user_input == bigrams_first[i]:
            matched.append(bigrams_first[i] + " " + bigrams_second[i])
    
# Return top 5 bigrams second word whose conditional prob are maximum

    top_next_words = {}
    for singleBigram in matched:
        top_next_words[singleBigram] = conditionalprob[singleBigram]
    
    topBigrams = heapq.nlargest(5, top_next_words, key = top_next_words.get)
    #print("Value of B is:- ")
    #print(topBigrams)
    if(len(topBigrams)>0):
        print("Did you mean?")
        for b in topBigrams:
            print(b)
     
