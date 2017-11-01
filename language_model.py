#Entered query is split and words within the sentence undergo permutations to from various sentences.
#pwords2->bigram model(probability of a two words occuring together)
#cPword->calculates conditional probability.
#load_counts->dictionary of all the bigrams of the sentence.
#product->gives the probability of the entire sentence by multiplying the conditional probabilities of the bigram obtained from load_counts.
#pwords2 uses cPword and the product function to calculate the probability of two word occuring together given that the previous word has occured.

#Bag of words concept
import re
from itertools import permutations as perm
import string
from collections import Counter
import text_segment as ts
import extract_books as s2
def pdist(counter):
    "A list of the probability of each word in the doc"
    N=sum(counter.values())
    return lambda x: counter[x]*1.0/N*1.0

def pword(w):
    return p(w)

def pwords_author(words):
    "Probability of words, assuming each word is independent of others."
    return product(p1w_author(w) for w in words)
def pwords_content(words):
    return product(p1w_content(w) for w in words)


def splits(words,start=0,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]


def pwords2_author(words, prev='<S>'):
    "The probability of a sequence of words, using bigram data, given prev word."
    return product(cPword_author(w, (prev if (i == 0) else words[i-1]) )
                   for (i, w) in enumerate(words))

def pwords2_content(words, prev='<S>'):
    "The probability of a sequence of words, using bigram data, given prev word."
    return product(cPword_content(w, (prev if (i == 0) else words[i-1]) )
                   for (i, w) in enumerate(words))


#conditional probability
def cPword_author(word, prev):
    "Conditional probability of word, given previous word."
    bigram = prev + ' ' + word
    if p2w_author(bigram) > 0 and p1w_author(prev) > 0:
        return p2w_author(bigram) / p1w_author(prev)
    else: # Average the back-off value and zero.
        return p1w_author(word) / 2

#conditional probability
def cPword_content(word, prev):
    "Conditional probability of word, given previous word."
    bigram = prev + ' ' + word
    if p2w_content(bigram) > 0 and p1w_content(prev) > 0:
        return p2w_content(bigram) / p1w_content(prev)
    else: # Average the back-off value and zero.
        return p1w_content(word) / 2


def load_counts(filename):
    bigramw=[]
    trigramw=[]
    keyb={}
    c=Counter()
    for line in filename.split('\n'):
        word=ts.tokens(line)
        bigramw=zip(word,word[1:])
        bigramlist.extend(bigramw)
        trigramw=zip(word,word[1:],word[2:])
        trigramlist.extend(trigramw)
        for (a,b) in zip(word,word[1:]):
            c[a+' '+b]=c[a+' '+b]+1
    return c

def product(p):
    result=1
    for x in p:
        result*=x
    return result
#for gving result of bigram model
def display_lang_author(sentence):
    #sent='this is good'
    temp=list(perm(sentence.split()))
    print temp
    temp_prob=0.0
    temp_sent=''
    sent=[]
    for word in temp:
        s=' '.join(word)
        sent.append(s)
    print sent
    for snt in sent:
        if temp_prob < pwords2_author(snt.split(' ')):
            temp_prob=pwords2_author(snt.split(' '))
            temp_sent=snt
    return temp_sent

def display_lang_content(sentence):
    #sent='this is good'
    temp=list(perm(sentence.split()))
    print temp
    temp_prob=0.0
    temp_sent=''
    sent=[]
    for word in temp:
        s=' '.join(word)
        sent.append(s)
    print sent
    for snt in sent:
        if temp_prob < pwords2_content(snt.split(' ')):
            temp_prob=pwords2_content(snt.split(' '))
            temp_sent=snt
    return temp_sent


bigramlist=[]
trigramlist=[]

author=s2.author
author_count=ts.count_author
author_dict=ts.author_dict
counts_1=load_counts(author)
p2w_author=ts.pdist(counts_1)
p1w_author=ts.pdist(author_count)

content=s2.content
content_count=ts.count_content
content_dict=ts.content_dict
counts_2=load_counts(content)
p2w_content=ts.pdist(counts_2)
p1w_content=ts.pdist(content_count)


#print pwords2('henry james'.split())
#print display_lang_author('robert of letters')
#print display_lang_content('victoria queen have')
