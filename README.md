# Catamaran
This is the repo of "CATAMARAN: A Cross-lingual Long Text Abstractive Summarization Dataset"

Catamaran
|_  CATAMARAN_Samples.json  # A CATAMARAN sample that contains several text-summary pairs. If you want to get access of the whole dataset, please contact Littlered_Lynn@outlook.com

    Crawler.py  # The crawler script of New York Times Chinese
    
    DataFilter.py # The script to perform cleaning and filtering of the raw data obtained by crawler
    
    Analyzer.py # Analyze the basic characteristics of our dataset
    
    Pipeline.py # The training, evaluating and generating pipeline with our dataset using mBART of Huggingface
    
    CalRouge.py # The script to calculate the rouge-1, rouge-2 and rouge-l metrics of the generated result
