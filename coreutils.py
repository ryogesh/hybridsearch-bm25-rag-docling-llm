#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2025  Yogesh Rajashekharaiah
# All Rights Reserved

""" coreutils module: Provides common utilities """

import os
import sys
import random
import json
import typing
import logging
# Just importing logging can raise error on RotatingFileHandler
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path
from urllib.parse import urlencode as urlparse_encode
import urllib3
import requests
from ftfy import fix_text
import torch

from coreconfigs import (_CHUNKTOKENIZER, _MODEL_PATH, _EMBED_DIM, _MAX_VECSEARCH_RESULTS, _FTBASE,
                         _FTPOST, _FTINDEX, _FTGET, _FTTLSVERIFY, _VECBASE, _CUDA_ARCHTYPE, _VECPOST,
                        _VECINDEX, _VECGET, _VECTLSVERIFY, _LM_MDL, _LM_MSG_TMPLT, _LM_MAX_INPUT_TKNS,
                        _LM_MAX_OUTPUT_TKNS, _REPETITION_PENALTY, _DO_SAMPLE, _TOP_K, _TOP_P,
                        _MAX_LOG_BACKUP_FILES, _MAX_LOGFILESIZE, _GET_TIMEOUT, _POST_TIMEOUT)

# Import SentenceTransformer, transformers after setting the HF_HUB_CACHE location
# If not HF will not use the pre-downloaded models in _MODEL_PATH location
# Instead HF will download to the user local cache
os.environ["TORCH_CUDA_ARCH_LIST"] = _CUDA_ARCHTYPE
os.environ["HF_HUB_CACHE"] = _MODEL_PATH
import transformers
from sentence_transformers import SentenceTransformer


def good_text(txt: str) -> str:
    """ Basic filters to cleanup text """
    # Replace any windows special characters, mojibake ...
    txt = fix_text(txt.replace("<missing-text>",'').replace('\n', ' '))
    txt = ' '.join(txt.split())
    return txt

def getlgr(lgrname: str, loglevel: int=logging.INFO):
    """ Function provides a logger
    Parameters
    -----------
    loglevel : int
        Set the loglevel (default: INFO)
    lgrname : str
        Set the logger name

    Returns
    -------
    loghandle : lgr object
        lgr for generating file output; file name will be the lgrname+dt.log
    """
    lgr = logging.getLogger(lgrname)
    lgr.setLevel(loglevel)
    if not lgr.handlers:
        # Create the logs folder if not present
        lgfldr = Path(Path(__name__).parent.resolve().as_posix(), "logs")
        lgfldr.mkdir(parents=False, exist_ok=True)
        logdt = time.strftime('%Y-%m-%d', time.localtime())
        flh = RotatingFileHandler(Path(lgfldr, f"{lgrname} {logdt}.log"),
                                  maxBytes=_MAX_LOGFILESIZE,
                                  backupCount=_MAX_LOG_BACKUP_FILES)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(funcName)s() %(message)s")
        flh.setFormatter(fmt)
        lgr.addHandler(flh)
        lgr.propagate = False
    return lgr

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RequestsOps():
    """ For fulltext, vectorDB index GET/POST operations """
    def __init__(self, get_timeout: float=_GET_TIMEOUT, post_timeout: float=_POST_TIMEOUT):
        """ Default post timeout is set to a higher value """
        self._req_session = requests.Session()
        self._get_timeout = get_timeout
        self._post_timeout = post_timeout

    def requests_session_close(self):
        """ Make sure to close the session """
        if self._req_session:
            self._req_session.close()

    def requests_post(self, uri_type: typing.Literal["fulltext", "vec"], data: dict|str=None,
                     headers: dict=None, params: dict=None, verify: bool=False):
        """ TLS verify is set to False, set to true for certificate verification 
        uri_type = "fulltext" -> Solr Fulltext  or "vec" -> OpenSearch VectorDB 
        data = if dictionary, will be converted as json strings
        """
        # Only json data is accepted if headers is None, else specify the header
        if data:
            if isinstance(data, dict):
                data = json.dumps(data)
            if not headers:
                headers = {"Content-type": "application/json"}
        if uri_type == "fulltext":
            posturi = f"{random.choice(_FTBASE)}/{_FTINDEX}/{_FTPOST}"
            verify = verify or _FTTLSVERIFY
        elif uri_type == "vec":
            posturi = f"{random.choice(_VECBASE)}/{_VECINDEX}/{_VECPOST}"
            verify = verify or _VECTLSVERIFY
        if params:
            posturi = f"{posturi}?{urlparse_encode(params, )}"
        req = requests.Request('POST', posturi, data=data, headers=headers)
        req_prepped = self._req_session.prepare_request(req)
        resp = self._req_session.send(req_prepped,
                                      timeout=self._post_timeout,
                                      verify=verify)
        return resp

    def requests_get(self, uri_type: typing.Literal["fulltext", "vec"], data: dict|str=None,
                     headers: dict=None, params: dict=None, verify: bool=False):
        """ uri_type = "fulltext" -> Solr Fulltext  or "vec" -> OpenSearch VectorDB """
        if data:
            if isinstance(data, dict):
                data = json.dumps(data)
            if not headers:
                headers = {"Content-type": "application/json"}
        if uri_type == "fulltext":
            geturi = f"{random.choice(_FTBASE)}/{_FTINDEX}/{_FTGET}"
            verify = verify or _FTTLSVERIFY
        elif uri_type == "vec":
            geturi = f"{random.choice(_VECBASE)}/{_VECINDEX}/{_VECGET}"
            verify = verify or _VECTLSVERIFY
        if params:
            # safe + because the field list is passed with + in Solr. e.g. field1+field2
            geturi = f"{geturi}?{urlparse_encode(params, safe='+')}"
        req = requests.Request('GET', geturi, data=data, headers=headers)
        req_prepped = self._req_session.prepare_request(req)
        resp = self._req_session.send(req_prepped,
                                      timeout=self._get_timeout,
                                      verify=verify)
        return resp


class VectorEmbeddings():
    """ Use model to generate embeddings """
    def __init__(self, mdl=None):
        if mdl:
            self.emb_mdl = mdl
        else:
            self.emb_mdl = SentenceTransformer(_CHUNKTOKENIZER)
            ## Verify embedding dimension size before processing
            embeddings = self.emb_mdl.encode("Hello World")
            if _EMBED_DIM < embeddings.size:
                print(f"VectorDB dimension={_EMBED_DIM} < Model embedding dimension={embeddings.size}")
                print("Choose a different model or change embedding dimension on DB.")
                print("Exiting...")
                sys.exit(1)
            else:
                print("Embedding model ok.")

    def get_qry_for_similar_texts(self, text: str, max_results: int=_MAX_VECSEARCH_RESULTS) -> dict:
        """
        1. Generate text embedding on the input text
        2. Compare similarity against vectorDB and get texts similar to the input text.
        """
        embeddings = self.emb_mdl.encode(text)
        embed_str = embeddings.tolist()

        qrydct = {"_source": "false",
                  "fields": ["docchunk", "docpath"],
                  "query": {"knn": {"chunkvec": {"vector": embed_str, "k": max_results }}}
                 }
        return qrydct


class LMOps():
    """For Language Model generation/summarization """
    def __init__(self):
        """ Penalize repetitions - 1.1, default temp is 6. Less randomness, stick to existing document texts """
        self.pipeline = transformers.pipeline("text-generation",
                                              model=_LM_MDL,
                                              torch_dtype=torch.bfloat16,
                                              device_map="auto",
                                             )
        self.gconfigdct = self.pipeline.model.generation_config.to_dict()
        self.gconfigdct["max_new_tokens"] = _LM_MAX_OUTPUT_TKNS
        self.gconfigdct["do_sample"] = _DO_SAMPLE
        self.gconfigdct["top_k"] = _TOP_K
        self.gconfigdct["top_p"] = _TOP_P
        self.gconfigdct["repetition_penalty"] = _REPETITION_PENALTY
        self.gconfigdct["pad_token_id"] = self.pipeline.model.config.eos_token_id
        self._msg_template = _LM_MSG_TMPLT

    def mdl_response(self, qry: str, temp: int=6) -> str:
        """ Function returns the answer from the Language Model
        Invoked with the context (RAG), qry is with context
        """
        if temp < 1 or temp > 9:
            temp = 6
        # 10 tokens less than the max tokens
        if len(qry_texts_lst := qry.split()) >= (maxlen := _LM_MAX_INPUT_TKNS-10):
            qry = ' '.join(qry_texts_lst[:maxlen])
        self._msg_template[1]['content'] = qry
        prompt = self.pipeline.tokenizer.apply_chat_template(self._msg_template, tokenize=False,
                                                             add_generation_prompt=True)
        self.gconfigdct["temperature"] = temp/10
        gconfig = transformers.GenerationConfig(**self.gconfigdct)
        outputs = self.pipeline(prompt, generation_config=gconfig)
        res = outputs[0]["generated_text"].split("<|im_start|>assistant\n")[1]
        return res
