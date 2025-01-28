#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2025  Yogesh Rajashekharaiah
# All Rights Reserved

""" Script to 
1. Iterate files under a directory, or work on a single file
2. Read text from files, save text into Solr (for BM25 search)
3. Read chunks, generate embeddings, store chunks and embeddings in Opensearch (VectorDB)
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
from humanize import precisedelta

from coreconfigs import _VECINDEX, _MODEL_PATH, _MAXCHNKLEN, _CHUNKTOKENIZER, _CUDA_ARCHTYPE
from coreconfigs import (_NUM_THREADS, _DO_OCR, _OCR_LANG, _PAGE_IMAGES, _PICTURE_IMAGES,
                        _TABLE_STRUCTURE, _CELL_MATCHING, _PDF_BACKEND)

# docling modules uses SentenceTransformer
# If HF_HUB_CACHE location is not set then HF will not use the pre-downloaded models in _MODEL_PATH location
# Instead HF will download to the user local cache
os.environ["TORCH_CUDA_ARCH_LIST"] = _CUDA_ARCHTYPE
os.environ["HF_HUB_CACHE"] = _MODEL_PATH

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                AcceleratorOptions,
                                                AcceleratorDevice,
                                                TableFormerMode)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

from coreutils import RequestsOps, getlgr, VectorEmbeddings, good_text

class DoclingOps(VectorEmbeddings):
    """ Docling operations on supported file types
    Extract text, contexts from documents
    """
    def __init__(self):
        """ Use docling to extract text, contexts from pdf document """
        super().__init__()
        self._chunker = HybridChunker(tokenizer=_CHUNKTOKENIZER, max_tokens=_MAXCHNKLEN)
        self.doc = None
        self.fulltext = None
        self._docconv = None
        self._chunk_iter = None

        self._ploptions = PdfPipelineOptions()
        self._ploptions.artifacts_path = _MODEL_PATH
        self._ploptions.do_ocr = _DO_OCR
        self._ploptions.ocr_options.lang = _OCR_LANG
        self._ploptions.generate_page_images = _PAGE_IMAGES
        self._ploptions.generate_picture_images = _PICTURE_IMAGES
        self._ploptions.do_table_structure = _TABLE_STRUCTURE
        self._ploptions.table_structure_options.do_cell_matching = _CELL_MATCHING
        self._ploptions.table_structure_options.mode = TableFormerMode.ACCURATE
        self._ploptions.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA,
                                                                 num_threads=_NUM_THREADS)
        if _PDF_BACKEND == "pypdfium":
            backend = PyPdfiumDocumentBackend
        else:
            backend = DoclingParseV2DocumentBackend
        # Using the defaults for other formats:
        # DOCX, PPTX, ASCIIDOC, MD, HTML, XML
        # Images: PNG, JPG, TIFF, BMP, GIF
        # Default pdf, image backend is DoclingParseV2DocumentBackend
        self._docconv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self._ploptions,
                backend=backend
                ),
            },
        )

    def convert_doc(self, docpath, stream=None) -> None:
        """ Accepts a Path or BytesIO object. Extracts all texts from the doc
        Docling doesn't have a feature to extract text by pages. Requires custom code.
        Iterates the document chunks (with context) and sets the chunked iterator
        """
        if stream:
            # If docpath is str, and not a Path
            try:
                fname = docpath.as_posix()
            except AttributeError:
                fname = docpath
            self.doc = self._docconv.convert(DocumentStream(name=fname, stream=stream)).document
        else:
            self.doc = self._docconv.convert(docpath).document
        self.fulltext = good_text(self.doc.export_to_text())
        self._chunk_iter = (good_text(self._chunker.serialize(chunk=chunk).replace('\n', ' '))
                                for chunk in self._chunker.chunk(self.doc)
                           )

    def get_embeddings_for_vdb(self, docpath) -> dict:
        """ Generator object to create chunktext, embedding """        
        for txtchunk in self._chunk_iter:
            # sometimes the chunker creates small chunks e.g. header text only, ignore them
            if len(txtchunk.split()) > 5:
                # Convert text chunks to lower case, generate and store the embeddings with chunks
                txt = txtchunk.lower()
                embeddings = self.emb_mdl.encode(txt)
                embed_str = embeddings.tolist()
                # Required json format for OpenSearch to index
                # {"chunkvec": <vector>, "docchunk": "<txt>", "docpath": "<fullfilepath>" }
                dct = {"chunkvec": embed_str,
                       "docchunk": txt,
                       "docpath":docpath.as_posix()
                      }
                yield dct


def index_fulltext(flname, overwrite_on_dup='n', tlsverify=False) -> None:
    """Index full text into Solr, if OpenSearch is used instead of Solr, make changes to this function
    Inputs
    --------
    flname : Path object
    overwrite_on_dup: on file duplicate (filehash match), even if the filename is different
                'n' -> do not overwrite existing index, 'y' -> overwrite existing index
    
    In case of Solr errors, application exits
    """
    # Do not overwrite existing index if the filehash matches
    #overwrite: Ignore uniqueness check on index, can speed up writes to Solr
    #_version_ : controls updates, if the document exists, the updates will be rejected
    if overwrite_on_dup == 'n':
        params = {"overwrite": "false", "_version_": -1}
    else:
        params = {"overwrite": "true"}
    fname = flname.as_posix()
    data = {"id":dlgdoc.doc.origin.binary_hash,
            "docts": flname.stat().st_mtime,
            "docpath":fname,
            "doctext":dlgdoc.fulltext}
    try:
        resp = requestsession.requests_post("fulltext", data=data, params=params, verify=tlsverify)
    except Exception as exc:
        _lgrft.error("Unable to make a POST")
        _lgrft.error(exc)
        _lgrft.error("Data:%s", data)
        print("Unable to post to full text indexing. Check the logs. Exiting...")
        sys.exit(1)
    if resp.ok:
        _lgrft.info("Indexed: %s, status: %s", fname, resp.status_code)
    else:
        # version conflict due to existing document
        if resp.status_code == 409:
            _lgrft.info("Ignore: %s, already exists - filehash: %s", fname, dlgdoc.doc.origin.binary_hash)
        else:
            _lgrft.error("Full text index failed:%s, status:%s", fname, resp.status_code)
            _lgrft.error(resp.text)
            _lgrft.error("Data:%s", data)
            print("Error during full text indexing. Check the logs. Exiting...")
            sys.exit(1)

def index_embds(flname, overwrite_on_dup='n', tlsverify=False) -> None:
    """Index document chunks with embeddings into OpenSearch
    If Solr is used instead of OpenSearch, make changes to this function
    
    Inputs
    --------
    flname : Path object
    
    In case of OpenSearch errors, application exists
    """
    if overwrite_on_dup =='n':
        indxtype = "create"
    else:
        indxtype = "index"

    data = ""
    # iterate over the embeddings and build the newline json data
    for cntr, line in enumerate(dlgdoc.get_embeddings_for_vdb(flname)):
        # Using the file hash + chunks numbering as the unique id
        indx_name = {f"{indxtype}": {"_index":_VECINDEX, "_id":f"{dlgdoc.doc.origin.binary_hash}{cntr}"}}
        data = f"{data}{json.dumps(indx_name)}\n{json.dumps(line)}\n"
    try:
        resp = requestsession.requests_post("vec", data=data, verify=tlsverify)
    except Exception as exc:
        _lgremb.error("Unable to make a POST for embeddings")
        _lgremb.error(exc)
        _lgremb.error("vectorDB Data:%s", data)
        print("Unable to post to vectorDB. Check the logs. Exiting...")
        sys.exit(1)
    if (resp.ok and resp.json()["errors"]) or not resp.ok:
        # If indxtype = "create", and there is a duplicate will result in version conflict code
        if resp.json()["items"][0]["create"]["error"]["type"] == "version_conflict_engine_exception":
            _lgremb.info("Ignore embeddings for: %s, already exists - filehash: %s", flname, dlgdoc.doc.origin.binary_hash)
        else:
            _lgremb.error("Chunks index failed:%s", flname)
            _lgremb.error(resp.json())
            _lgremb.error("vectorDB Data:%s", data)
            print("Error during saving to vectorDB. Check the logs. Exiting...")
            sys.exit(1)
    else:
        _lgremb.info("Embeddings: %s, status: %s", flname, resp.status_code)

def process_files(argsdct):
    """ Accepts a file or a folder and then extracts text. Ignores symlinks, will iterate a folder.
    After the document full text is extracted: Will index full text for BM25 search +
                                               index text chunks and Embeddings for Semantic search
    Required: fl_or_fldr - file or a folder name,  string or Path object                                         
    Defaults:
            prevrun_dt=None: process all files, ignore file mtime check
                Expected format: '%Y-%m-%d %H:%M:%S' e.g. '2025-01-15 21:21:08'
            fulltext=True:  Index document full text
            embeddings=True: Index document text chunks with embeddings
            overwrite_on_dup=True: To avoid duplicates in index, if a file has been indexed before
                              do not reindex, even if the file name is different
            tlsverify=False: Ignore TLS certificate validation
       
    """
    # If the function arg is dictionary then extract values,
    # If str or Path: create the argument dictionary
    try:
        fl_or_fldr = argsdct["fl_or_fldr"]
    except TypeError:
        # Text or Path object, hence TypeError
        fl_or_fldr = argsdct
        argsdct = {}
        argsdct["fl_or_fldr"] = fl_or_fldr
        ## All other values will be undefined, hence the .get below

    prevrun_dt = argsdct.get("prevrun_dt", None)
    overwrite_on_dup = argsdct.get("overwrite_on_dup", 'n')
    fulltext = argsdct.get("fulltext", 'y')
    embeddings = argsdct.get("embeddings", 'y')
    tlsverify = argsdct.get("tlsverify", 'n')
    if isinstance(tlsverify, str):
        tlsverify = bool(tlsverify == 'y')

    if fulltext == 'n' and embeddings == 'n':
        _lgrm.info("Both fulltext and embeddings indexing are set to False.")
        _lgrm.info("Nothing to do. Exiting...")
        print("Not indexing fulltext or into vectorDB. Exiting...")
        sys.exit(0)
    if isinstance(fl_or_fldr, str):
        fl_or_fldr = Path(fl_or_fldr)
    # Ignore symlinks
    if fl_or_fldr.is_symlink():
        _lgrm.info("Ignore symlink: %s", fl_or_fldr)
        return
    # Check for None or 0 for the file mtime restriction
    if prevrun_dt:
        try:
            prevrun_dt = int(time.mktime(time.strptime(prevrun_dt, '%Y-%m-%d %H:%M:%S')))
        except (TypeError, ValueError):
            _lgrm.error("Time format is incorrect: %s", prevrun_dt)
            _lgrm.error('Expected time format:"%Y-%m-%d %H:%M:%S" e.g. "2025-01-15 21:21:08". Exiting...')
            return
    else:
        prevrun_dt = 0

    def processfile(fname) -> None:
        """Invokes Docling, gets the fulltexts and text chunks 
           Depending on flags: indexes fulltext and/or text chunks with embeddings into vectorDB
        """
        flname = fname.as_posix()
        btime = datetime.now()
        try:
            dlgdoc.convert_doc(fname)
        except Exception as exc:
            _lgrm.error(exc)
        else:
            _lgrm.info("%s took %s", flname, precisedelta(datetime.now() - btime))
            # Index embeddings and fulltext documents async
            if fulltext == 'y':
                # Use batch indexing, not one at a time
                index_fulltext(fname, overwrite_on_dup, tlsverify)
            if embeddings == 'y':
                index_embds(fname, overwrite_on_dup, tlsverify)

    fl_cntr = 0
    print(f"Start:{(flname := fl_or_fldr.as_posix())} at {time.strftime('%x %X')}")
    print("...")
    _lgrm.info("Start: %s at %s", flname, time.strftime("%x %X"))
    if fl_or_fldr.is_dir():
        for fname in fl_or_fldr.iterdir():
            if fname.is_dir():
                argsdct["fl_or_fldr"] = fname
                process_files(argsdct)
            elif fname.is_file() and not fname.is_symlink():
                # Process files if file modified time > the last ingestion time
                if (fts := int(fname.stat().st_mtime)) > prevrun_dt:
                    fl_cntr += 1
                    processfile(fname)
                else:
                    _lgrm.info("Ignore:%s, file timestamp:%s <= %s", fname.as_posix(), fts, prevrun_dt)
    elif fl_or_fldr.is_file():
        # No mtime check when a single file is processed from the command line
        fl_cntr = 1
        processfile(fl_or_fldr)
    _lgrm.info("End:%s at %s, Total files:%s", flname, time.strftime("%x %X"), fl_cntr)
    print(f"End:{flname} at {time.strftime('%x %X')}, Total files:{fl_cntr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Index document fulltext and chunks with embeddings into Solr/Opensearch.",
                                     epilog='''e.g. To index a file with all defaults run: python ./store_docs.py "/tmp/test1.pdf" ,
                                     for all files under a folder including all subfolders run: python ./store_docs.py "/folder1/alldocsfolder/"
                                     ''')
    parser.add_argument('fl_or_fldr',
                        help='Processes a file for all files under the folder(and subfolders). Required:True')
    parser.add_argument('-r', '--prevrun_dt', default=None,
                        help='Process files after timestamp, e.g."2025-01-15 21:21:08". If None, all files are processed. i.e. File mtime is not checked. Default: None')
    parser.add_argument('-o', '--overwrite_on_dup', default='n', choices={'y', 'n'},
                        help="If a file has been indexed before do not reindex, even if the file name is different. Default: n")
    parser.add_argument('-t', '--fulltext', default='y', choices={'y', 'n'},
                        help="Index document fulltext. Default: y")
    parser.add_argument('-e', '--embeddings', default='y', choices={'y', 'n'},
                        help="Index document chunks and embeddings into VectorDB. Default: y")
    parser.add_argument('-s', '--tlsverify', default='n', choices={'y', 'n'},
                        help="Verify TLS certificate. Default: n")
    args = parser.parse_args()
    _lgrm = getlgr("storedocs")
    _lgrft = getlgr("fulltext")
    _lgremb = getlgr("embds")
    dlgdoc = DoclingOps()
    requestsession = RequestsOps()
    process_files(vars(args))

    # FT indexing is done without commit for performance.  Final commit before exiting
    try:
        # expungeDeletes is less expensive than optimize, use after proper testing
        response = requestsession.requests_post("fulltext", params={"commit": "true", "expungeDeletes": "true"})
    except Exception as err:
        _lgrm.error("Unable to make a commit on fulltext index")
        _lgrm.error(err)
        print("Unable to commit full text index. Check the logs. Exiting...")
    if not response.ok:
        _lgrm.error("Full text index commit failed for folder")
        _lgrm.error("Response error:%s", response.status_code)
        print("Full text index commit error. Check the logs. Exiting...")
    requestsession.requests_session_close()
