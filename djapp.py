#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2025  Yogesh Rajashekharaiah
# All Rights Reserved

""" Document search single page Django app """

import os
import time
from datetime import datetime
from collections import OrderedDict
import markdown
from humanize import precisedelta

from django.conf import settings
from django.urls import path
from django.shortcuts import render

from coreutils import getlgr, RequestsOps, LMOps, VectorEmbeddings


def fmt_ftresults(results: str, fltr_inp: str) -> (list[tuple], int):
    """ Formats BM25 search results for HTML display
    results: full text index(Solr) results: json string
    fltr_inp: user search term filtered with stop and question words
    Returns: List[Tuple]: Tuple contains the filepath, sample search terms highlighted text
    """
    doclst = []
    doccntr = None
    for doccntr, item in enumerate(results["response"]["docs"], 1):
        ## extract words around the search terms, ignore ?,.
        # e.g. quality. or quality, or quality?
        txtlst = item["doctext"].replace('?','').replace(',','').replace('.','').split()

        # NOT, AND, OR has special meaning in Solr search, remove them for the filtered text
        inpl = fltr_inp.replace("AND", '').replace("OR", '').replace("NOT", '')

        # Remove " from the input query string e.g. searching for "all good work" -> next to each other
        # remove anything after ~, e.g. "apache lucene"~5 means within 5 words of each other
        inpl = inpl.replace('?','').replace(',','').replace('.','').replace('"','').lower().split('~')[0].split()

        # Get the input search terms, indexes the terms in the received fulltext in sorted order
        indxs = []
        # Left strip +- from the input query string + -  has special meaning to must include/exclude the term
        # L/R strip (), grouping clauses
        for term in inpl:
            indxs +=  [i for i, x in enumerate(txtlst) if x.lower()==term.lstrip('(').lstrip('+').lstrip('-').rstrip(')')]
        indxs = list(OrderedDict.fromkeys(indxs))
        maxind = len(txtlst) -1
        #texts = []
        txt = ''
        # get 5 words before and after the search terms
        # show ~100 words for each document
        for cntr, ind in enumerate(indxs):
            if (lft:=ind -5) <= 0:
                lft = 0
            if (rgt:=ind +5) > maxind:
                rgt = maxind
            try:
                # Add emsp after each sentence chunk for visual separation
                txt = f"{txt} {' '.join(txtlst[lft:rgt])} &emsp;"
            except IndexError:
                pass
            if cntr > 9:
                txt = f"{txt} ..."
                break
        # If we are not able to get texts around the search terms, show few chars from the document
        txt = txt or item["doctext"][:500]
        txtlst = txt.split()
        # Iterate the texts for all search terms, convert the terms to <span> for highlighting
        for term in inpl:
            for cntr, each in enumerate(txtlst):
                if each.lower() == term.lstrip('+').lstrip('-'):
                    txtlst[cntr] = f'<span class="srchterm">{each}</span>'
        # document ts is stored as int. Convert to readable format
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item["docts"]))
        doclst.append((item["docpath"], mtime, ' '.join(txtlst)))
    return doclst, doccntr

def req_docs(inp: str) -> (list[tuple], str, str, str):
    """ Accepts user input, query Solr/Opensearch for fulltext/semantic search texts 
    Returns: List of documents: based on query terms
             AI generated summary: if the query has questions, or asks for explaination
             desc: No of documents found based on BM25 search
             err: In case of errors, "error description" else None
    """
    # stop and question words, borrowed from NLTK
    stp_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
     'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'did', 'doing',
     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
     'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'here', 'there',  'all', 'any', 'both', 'each', 'few', 'more', 'most',  'by', 'for', 'once',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'below', 'to',
     'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
     "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    q_words = {'what', 'which', 'who', 'whom', 'when', 'where', 'whose', 'why', 'how'}
    addl_q_words = {'explain', 'describe', 'elaborate', 'summarize', 'examine', 'evaluate',
                    'analyze', 'clarify', 'diagnose', 'assess' }

    err = None
    descr = None
    ft_result = None
    # If there is a question, enable AI summary
    ai_summary = bool(inp.endswith('?'))

    linp = ""
    for cntr, item in enumerate(inp.split()):
        lwitem = item.lower()
        # NOT, AND, OR has special meaning in Solr search, below if order is important
        # if the first word is "question like" then create AI summary, .e.g. describe, explain
        if cntr==0 and (lwitem in q_words or lwitem in addl_q_words):
            ai_summary = True
        elif item in ("AND", "OR", "NOT"):
            linp = f"{linp} {item}"
        # ignore stop and question words
        elif lwitem not in q_words and lwitem not in stp_words:
            linp = f"{linp} {lwitem}"

    if linp:
        params = {'q':f"{linp}", "fl":"doctext+docpath+docts"}
        try:
            resp = requestsession.requests_get("fulltext", params=params)
        except Exception as exc:
            _lgrdj.error("Unable to search documents")
            _lgrdj.error(exc)
            err = "Unable to search documents. Check the logs files."
        else:
            if resp.ok:
                results = resp.json()
                if results['response']['numFound'] > 0:
                    ft_result, doccnt = fmt_ftresults(results, linp)
                    descr = f"Documents found:{doccnt}"
                else:
                    descr = "No document found"
            else:
                _lgrdj.error("Error searching documents")
                _lgrdj.error(resp.text)
                err = "Error searching documents. Check the logs files."
    else:
        err = "Enter good search terms"

    # check if question is asked, use AI to summarize
    # Generate summary only if the BM25 search has results
    ai_summary = ai_summary and ft_result
    if ai_summary:
        ## Get similar texts from VectorDB (OpenSearch)
        # replace " - grouped search terms
        btime = datetime.now()
        # text stored in vectorDB is lowercase text
        linp = linp.replace('"','')
        _lgrdj.info("AI summary:%s", linp)
        qrydata = emdb.get_qry_for_similar_texts(linp)
        # Get the top 10 hits(files) from the BM25 and filter vectorDB on those file chunks only
        # "post_filter": {"bool": {"should": [{"match_phrase": { "docpath": "file1"}},
                             # {"match_phrase": { "docpath": "file2"}}]}}
        should_lst = [{"match_phrase": {"docpath": item[0]}}
                      for cntr, item in enumerate(ft_result) if cntr <10]
        qrydata["post_filter"] = {"bool": {"should": should_lst}}
        try:
            resp = requestsession.requests_get("vec", data=qrydata)
            _lgrdj.info("VectorDB: %s", precisedelta(datetime.now() - btime, minimum_unit="milliseconds"))
        except Exception as exc:
            # In case of errors, ignore ai_summary
            _lgrdj.error("Unable to make a GET for embeddings")
            _lgrdj.error(exc)
            ai_summary = False
        else:
            if resp.ok and resp.json()["hits"]["total"]["value"] > 0:
                # Add the context to the input query
                query_cntxt = f"{linp} "
                query_cntxt += ' '.join((each["fields"]["docchunk"][0]
                                      for each in resp.json()["hits"]["hits"]))
                ai_summary = markdown.markdown(lmdl.mdl_response(query_cntxt))
                _lgrdj.info("%s took %s", linp, precisedelta(datetime.now() - btime))
            else:
                _lgrdj.error("Unable to get similar texts for %s: %s", linp, resp.json())
                ai_summary = False
    return ft_result, ai_summary, descr, err

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
BASE_DIR = os.path.dirname(__file__)

settings.configure(
    DEBUG=False,
    SECRET_KEY="AB123xyz$#@!",
    ALLOWED_HOSTS=ALLOWED_HOSTS,
    ROOT_URLCONF=__name__,
    MIDDLEWARE_CLASSES=(
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ),
    TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
    ],
    INSTALLED_APPS=(
    'django.contrib.staticfiles',
    'django.contrib.contenttypes',
    ),
    STATICFILES_DIRS=(
    os.path.join(BASE_DIR, 'static'),
    ),
    STATIC_URL='/static/',
)


def index(request):
    """ Main method for the search page"""
    descr = None
    result = None
    err = None
    ai_summary = None
    inp_txt = request.GET.get('inp_txt', '')
    if inp_txt:
        result, ai_summary, descr, err = req_docs(inp_txt)
    context = {"inp_txt": inp_txt, "result": result, "ai_summary": ai_summary,
               "descr": descr, "err": err}
    return render(request, 'home.html', context)

urlpatterns = (
    path('', index),
    )


if __name__ == "__main__":
    _lgrdj = getlgr("searchdocsUI")
    requestsession = RequestsOps()
    lmdl = LMOps()
    emdb = VectorEmbeddings()
    from django.core.management import execute_from_command_line
    args = ['searchdocuments', 'runserver', '0.0.0.0:8000', '--noreload', '--skip-checks', '--nothreading', '--insecure' ]
    execute_from_command_line(args)
