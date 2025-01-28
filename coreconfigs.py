#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2025  Yogesh Rajashekharaiah
# All Rights Reserved

""" All configuration items
 CHANGE: Solr, OpenSearch URL list below
 IMPORTANT: Embedding model "_CHUNKTOKENIZER"
            Once we use and store embeddings in VectorDB, it cannot be changed.
            If we change the model, we need to reingest all documents(vectors) into VectorDB
"""

# Logger file settings
_MAX_LOGFILESIZE = 5000000 #5MB files
_MAX_LOG_BACKUP_FILES = 5

# GET/POST Requests timeouts
_GET_TIMEOUT = 1.0
_POST_TIMEOUT = 5.0

# CUDA type for torch
_CUDA_ARCHTYPE = "Ada"

# Solr for BM25 search, on full text
# no / at the beginning or the end or URL parts below
# searchdocuments is the name of the collection (index)
# list all the host:port where the searchdocuments collection shard replicas are present
# e.g. ["http://<fqdn1>:8983/solr", "http://<fqdn2>:8983/solr",]
_FTBASE = ["http://192.168.56.103:8983/solr",]
_FTINDEX = "searchdocuments"
_FTPOST = "update/json/docs"
_FTGET = "select"
# verify certificates?
_FTTLSVERIFY = False


# OpenSearch for storing vectors on text chunks
# no / at the beginning or the end or URL parts below
# searchdocs is the name of the index
# list all the host:port where the searchdocs index shards are present
_VECBASE = ["http://192.168.56.103:9200",]
# indexname Should be lowercase, do not start with _ or -, no special chars
_VECINDEX = "searchdocuments"
_VECPOST = "_bulk"
_VECGET = "_search"
_VECTLSVERIFY = False
# Return top N results from the vectorDB search
_MAX_VECSEARCH_RESULTS = 50


# Text Generation/Summarization model, Using a small language model for performance
# This model has max token length 8192 , reducing the max input tokens for faster response
# https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct

# Saved models path
_MODEL_PATH = "C:\\searchdocuments\\models"

_LM_MDL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_LM_MSG_TMPLT = [{"role": "system",
                  "content": "Provide a concise, objective summary of the input text focusing on main points."
                 },
                 {"role": "user", "content": ''}
                ]
_LM_MAX_INPUT_TKNS = 2048
# Max text generation/summarization output
_LM_MAX_OUTPUT_TKNS = 512
_DO_SAMPLE = True
_TOP_K = 50
_TOP_P = 0.95
_REPETITION_PENALTY = 1.1


# DOCLING related settings, vectorDB embeddings model
# https://huggingface.co/nasa-impact/nasa-ibm-st.38m?library=sentence-transformers
_CHUNKTOKENIZER = "nasa-impact/nasa-ibm-st.38m"

# Configure the VectorDB dimension to match the model dimension
_EMBED_DIM = 576

# Used by docling document chunker
_MAXCHNKLEN = 512

# PDF options, Change the backend to any text e.g. xxx to use the default backend
# "pypdfium" backend is faster and more memory efficient than the default backend
# But, at the expense of worse quality results, especially in table structure recovery
_PDF_BACKEND = "xxxx"
_DO_OCR = False
_OCR_LANG = ["en", ]
_PAGE_IMAGES = False
_PICTURE_IMAGES = False
_TABLE_STRUCTURE = True
_CELL_MATCHING = True
_NUM_THREADS = 4

"""
https://ds4sd.github.io/docling/faq/ question on offline
pre-download the tokenizer/embeddings model to this location
see below on how to pre-download

Default HF model folder is <home>/.cache/huggingface/hub

#To download the docling model into local directory run the below step
# Refer https://github.com/DS4SD/docling/blob/main/docling/pipeline/standard_pdf_pipeline.py

import os
os.environ["HF_HUB_CACHE"] = _MODEL_PATH
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
download_path = snapshot_download(
    repo_id="ds4sd/docling-models",
    force_download=True,
    local_dir=f"{_MODEL_PATH}",
    revision="v2.1.0",
)

Once we use the Embedding model "_CHUNKTOKENIZER" and store embeddings in VectorDB
It cannot be changed. If we change the model then we need to reingest chunks,vectors into VectorDB
e.g.
_CHUNKTOKENIZER = "hkunlp/instructor-large"
_CHUNKTOKENIZER = "BAAI/bge-base-en-v1.5"
_CHUNKTOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


#To download the _CHUNKTOKENIZER or _LM_MDL models into local directory run the below step
You can verify _EMBED_DIM, _MAXCHNKLEN

from sentence_transformers import SentenceTransformer
mdl = SentenceTransformer(_CHUNKTOKENIZER)  #OR _LLM_MDL
print(mdl.tokenizer.model_max_length) #_MAXCHNKLEN / _LLM_MAXTKN_LEN

txtenc = mdl.encode("hello world")
print(txtenc.size) #_EMBED_DIM
"""
