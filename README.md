# KGExtractor 
**KGExtractor** is an AI system that tries to extract relevant keywords from a corpus of scientific papers. KGExtractor was proposed and used in the development of the ontology of adolescents' digital leisure, presented in the article entitled: **Introducing an ontology of adolescents' digital leisure**.

**KGExtractor** gets as an input the path of the folder where the pdfs are stored. Then it extracts the text from each pdf and uses a text encoding model to find the similarity between the doc of the paper and each candidate term of the text. 
The extracted terms are stored in a .csv file (a path for this filename is also required). 

The systems architecture is shown in the following Figure.
![alt text](https://github.com/geofila/KGExtraction/blob/main/System%20Architecture.png)

##Extract algorithm
The following command will read each one of the texts on the folder *"pdf_corpus"* and will store the extracted terms on the file *"extraxted_terms.csv."*
```
python extract.py  --path "pdf_corpus" --outp_filename "extraxted_terms.csv"
```

This algorithm has additional arguments in order to be able to adjust its results to our task. 
For example, the following command will extract  20 keyphrases from each pdf, which will be between 2-4 tokens (cannot return a single keyword). Diversity defines how diverse the result will be (for more: ```python extract.py --help```)

```
python extract.py  --path "pdf_corpus" --outp_filename "extraxted_terms.csv" --min_ngram 2 --max_ngram 4 --top_n 20 --diversity 0.5
```


##Requirements
- sentence-transformers
- pyPDF2