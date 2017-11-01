# extract names of books and corresponding authors.
#author variable holds all the information regarding book titles and author
import re
import string
import os
import codecs
import json

def extract_book_content(files):
    string_content=''

    for filename in files:
        with codecs.open(root[:]+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
                string_content +='\n'+file_name.read()
    return string_content


def extract_book_author(files):
    info=[]
    books=[]
    string=''
    mapper={}
    solr_dict={}
    for filename in files:
        with codecs.open(root[:]+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:

            text=file_name.readline()
            text1=file_name.read()
            solr_dict[filename]=text1

            info.append(text.lower())
            book_info=''

            non_useful=['the','project',"gutenberg's",'ebooks','etexts','etext','ebook','gutenberg','this','presented','file','s']
            result=[word  for word in re.findall('[a-z0-9]+',text.lower()) if word not in non_useful]

            book_info=' '.join(result)
            book_info=re.sub("of","",book_info,count=1).strip()
            books.append(book_info)
            mapper[book_info]="http://127.0.0.1:5000/js/Dataset-1/"+filename
    #print books
    #print book_info
    string='\n'.join(books)
    #print(string)

    return string,mapper,solr_dict
def extract_book_path(search_query):
    path=[]
    keys=[]
    for key in mapper.keys():
        if search_query in key:
            path.append(mapper[key])
            keys.append(key)
    return path,keys




raw_path=['C:\Users\Aishwarya Sadasivan\Dataset-1']
for path in raw_path:
    for root,dir,files in os.walk(path):
        print ("Files in: " + root[:])
author,mapper,solr_dict= extract_book_author(files)
content=extract_book_content(files)
#print(type(solr_dict))
file_path,file_key=extract_book_path('history')
#print (file_path)
