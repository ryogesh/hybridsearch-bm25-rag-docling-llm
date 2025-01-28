# Hybrid Search with BM25 and RAG with vector database



## Overview

Tool to interact with language models for document search and summarization, enabled by BM25 and Retrieval-Augmented Generation(RAG). Context embeddings are stored and retrieved from a vector database (OpenSearch). Term search (BM25) uses Solr.



## Features
- Extracts texts from files. Supported file formats : pdf, docx, xlsx, pptx, html, md, asciidoc, images: jpg, png, tiff, bmp, gif
- Indexes full texts in Solr for BM25 search
- Creates context-enriched text chunks, generates embeddings using custom model. Saves the text chunks and embeddings in vectorDB, OpenSearch
- UI for search. Generates AI summary.


## Installation
### Prerequisites

- [Python](https://www.python.org/downloads/) 3.10 or greater
- check requirements.txt for required python libraries
- requires nvidia-gpu

### Solr and OpenSearch

-  [Refer](https://medium.com/@yogi_r/hybrid-search-with-bm25-and-vector-database-with-rag-e0fe189f93aa) for OpenSearch, Solr installation details and document search UI examples


### Download models to local directory
By default HuggingFace models are downloaded to "<user_home>"/.cache/huggingface/hub folder. In order to download to a common folder. Follow the below steps
Get the _MODEL_PATH location, launch a python session and 

    ```
	# Download Docling models
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
	
	# Download Language models, _CHUNKTOKENIZER, _LM_MDL
	from sentence_transformers import SentenceTransformer
	mdl = SentenceTransformer(_LM_MDL)
	print(mdl.tokenizer.model_max_length)
	
    ```
	

## Application

- storedocs.py: Script to read files from a folder(and all subfolders), saves full text to Solr, generate embeddings on context-enriched text chunks and saves to OpenSearch
- djapp.py: Single page Django application, UI, for document search and summarization

### MUST DO
- coreconfigs.py: Application configurations. Review and edit this "important" file.


## Getting Started

### Application config and run
- Download the repo
- Perform the installation and configuration steps (see above)
- Edit coreconfigs.py to update the Solr, OpenSearch API endpoints. Optionally choose different language models.
- run storedocs.py to generate texts and save index on Solr and embeddings on OpenSearch. See the available options when ingesting documents.

    ```
	python ./storedocs.py -h
	usage: storedocs.py [-h] [-r PREVRUN_DT] [-o {y,n}] [-t {y,n}] [-e {y,n}] [-s {y,n}] fl_or_fldr

	Index document fulltext and chunks with embeddings into Solr/Opensearch.

	positional arguments:
	  fl_or_fldr            Processes a file for all files under the folder(and subfolders). Required:True

	options:
	  -h, --help            show this help message and exit
	  -r PREVRUN_DT, --prevrun_dt PREVRUN_DT
							Process files after timestamp, e.g."2025-01-15 21:21:08". If None, all files are processed.
							i.e. File mtime is not checked. Default: None
	  -o {y,n}, --overwrite_on_dup {y,n}
							If a file has been indexed before do not reindex, even if the file name is different. Default:
							n
	  -t {y,n}, --fulltext {y,n}
							Index document fulltext. Default: y
	  -e {y,n}, --embeddings {y,n}
							Index document chunks and embeddings into VectorDB. Default: y
	  -s {y,n}, --tlsverify {y,n}
							Verify TLS certificate. Default: n

	e.g. To index a file with all defaults run: python ./store_docs.py "/tmp/test1.pdf" , for all files under a folder
	including all subfolders run: python ./store_docs.py "/folder1/alldocsfolder/"

    ```

