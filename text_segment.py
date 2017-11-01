#pwords->returns the product of the probabilities of the words in the sentence.
#Entered sentence is split and checked for a valid sentence using pwords
#if the query is not a valid sentence it is joined and sent to segment function
#segment function->will segment the query into most proable sequence of words.
#segment function uses pwords to calculate the proabilities.

import re
import extract_books as s2
from collections import Counter


def tokens(text):
    "Returns the words contained in the sentence"
    return re.findall('[a-z]+',text.lower())
def pdist(counter):
    "A list of the probability of each word in the doc"
    N=sum(counter.values())
    return lambda x:counter[x]*1.0/N*1.0

def pword_author(w):
    return author_dict(w)


def pword_content(w):
    return content_dict(w)


def pwords_author(words):
    return product([pword_author(w) for w in words])

def pwords_content(words):
    return product([pword_content(w) for w in words])


def splits(words,start,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]


def segment_author(word):
    "Returns the most probable segment"
    if not word:
        return []
    else:
        candidate=([first]+ segment_author(rest)
                    for (first,rest) in splits(word,1))
        return max(candidate,key=pwords_author)

def segment_content(word):
    "Returns the most probable segment"
    if not word:
        return []
    else:
        candidate=([first]+ segment_content(rest)
                    for (first,rest) in splits(word,1))
        return max(candidate,key=pwords_content)



def product(p):
    result=1
    for x in p:
        result*=x
    return result
def display_segment_author(sentence):
    text=''
    if pwords_author(sentence.split(' '))>0:
        return sentence
    else:
        sentence=''.join(sentence.split(' '))
        for word in sentence.split(' '):
                txt=' '.join(segment_author(word))
                text=text+txt+' '

        return text

def display_segment_content(sentence):
    text=''
    if pwords_content(sentence.split(' '))>0:
        return sentence
    else:
        sentence=''.join(sentence.split(' '))
        for word in sentence.split(' '):
                txt=' '.join(segment_content(word))
                text=text+txt+' '

        return text


author=s2.author
content=s2.content
count_author=Counter(re.findall('[a-z]+',author.lower()))
count_content=Counter(re.findall('[a-z]+',content.lower()))

author_dict=pdist(count_author)
content_dict=pdist(count_content)
#print display_segment_author('henryjames')
#print display_segment_content('spanishamerican')
