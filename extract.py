# Read PDF 
import PyPDF2 
import os 
from tqdm import tqdm 
import argparse
from keyword_extraction import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def pdf_to_terms(model, pdf_filename, args):
    terms = []
    with open(pdf_filename, 'rb') as pdfFileObj:
        # creating a pdf reader object 
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
        # printing number of pages in pdf file 
        for page_num in range (pdfReader.numPages):
            pageObj = pdfReader.getPage(page_num) 
            try:
                extractedText = pageObj.extractText()
                terms += model.extract_keyword(extractedText, top_n = args.top_n, diversity = args.diversity, n_gram_range = (args.min_ngram, args.max_ngram))
            except:
                pass
        return set(terms)



if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Hyperparameters of the Algorithm')
    parser.add_argument('--path', type=str,
                        help='Folder\'s path where the papers are stored')
    parser.add_argument('--outp_filename', type=str,
                        help='CSV filename where the results produced by the algorithm will be stored')
    
    parser.add_argument('--top_n', type=int, default = 5,
                        help='Number of candidate terms of each pdf')
    parser.add_argument('--diversity', type=float, default = 0.1,
                        help='Low diversity, means that the terms will be similar, otherwise will be returned diverse keyphrases')
    parser.add_argument('--min_ngram', type=int, default = 1, 
                        help='Tuple with the length of sortest and longer sequense of phrases that it could be returned. Default = (1, 3) means that the algorithm will return phrases with one to three tokens.')

    parser.add_argument('--max_ngram', type=int, default = 3, 
                        help='Tuple with the length of sortest and longer sequense of phrases that it could be returned. Default = (1, 3) means that the algorithm will return phrases with one to three tokens.')

    args = parser.parse_args() # parse arguments 

    if args.min_ngram >args.max_ngram:
        raise Exception (f"Must provide arguments: min_ngram <= max_ngram, get: {args.min_ngram} and {args.max_ngram}")
    
    model = "distilbert-base-nli-mean-tokens"
    key_ext_model = keyword_extractor(model)

    ext_terms = set()
    for file in tqdm(os.listdir(args.path)):
        if ".pdf" in file:
            path = os.path.join(args.path, file)
            cand_terms = pdf_to_terms(key_ext_model, path, args)
            ext_terms.union(cand_terms)

    # save terms to a csv file
    np.savetxt(args.outp_filename, (list(cand_terms)), delimiter=',', fmt='%s')