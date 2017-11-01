#front-end server side script to help bring together the various modules
#there are 2 web pages home.html and query.html
#home.html->this page displays a search box and takes in the search query.
#query.html->this shall take the query and pass it to text segment, spell check as well as language modelling
import spell_check as sc
import spell_check_content as sc_content
import text_segment as ts
import language_model as lm
from flask import Flask
from flask import request
from flask import render_template,send_from_directory
import string
import re
import extract_books as s2

app = Flask(__name__,static_folder='js',template_folder='template')
#@app.route('/<string:page_name>/')
#def render_static(page_name):
#    return render_template('%s.html' % page_name)
@app.route('/js/<path:path>/')
def static_page(page_name):
    return send_from_directory('js' ,path)

@app.route('/',methods=["GET","POST"])
def process_author():

    answer1=' '.join(i for i in sc.display_author(request.form['query_author'].lower()))
    print answer1
    #answer1=ts.display(request.form['query'])
    #for w in ts.display(request.form['query']):
        #answer1=answer1+w+" "

    answer2=ts.display_segment_author(answer1)
    print answer2
    #answer2=' '.join(i for i in sc.display(answer1))
    final=lm.display_lang_author(answer2)
    dloads,keys=s2.extract_book_path(final)
    return 'Showing Results for:'+final+render_template('doc.html',dloads=dloads,keys=keys)
    #return  '<br>'.join(s2.extract_book_path(final))

@app.route('/query.html',methods=["GET","POST"])
def process_content():

    answer_content1=' '.join(i for i in sc.display_content(request.form['query_content'].lower()))
    print answer_content1
    #answer1=ts.display(request.form['query'])
    #for w in ts.display(request.form['query']):
        #answer1=answer1+w+" "
    answer_content2=ts.display_segment_content(answer_content1)
    print answer_content2
    #answer2=' '.join(i for i in sc.display(answer1))
    final_content=lm.display_lang_content(answer_content2)
    return 'Did You Mean:'+final_content


if __name__ == '__main__':
    app.run()
