# COS 470   Repository

### Description
A public repository for my programming in COS 470: Introduction to Information Retrieval.

## Table of Contents - Projects
### &nbsp;&nbsp;- Project 1
##### Description: The first project focused on a Stack Exchange subforum and included:
1) Analyzing the collection (in my case, the archived data from the game development stack exchange)
2) Implementing a Boolean Retrieval System (AND, OR)
3) Implementing a simple Inverted Index System
4) Creation of a diverse query set
5) Evaluation of both systems (Precision & nDCG at cuts 5 and 10, Avg. Retrieval Time)

In order for the code to work, the archived data must be downloaded from: https://archive.org/download/stackexchange/gamedev.stackexchange.com.7z

The contents must be unzipped in the same directory as the code.

Post.py & post_parser_record.py ARE NOT mine, nor did I have any hand in coding them. They were tools provided for this project, although similiar much simpler and rudimentary variants of the code in post_parser_record.py can be found in the IR_Project1_SourceCode_Collection-Analysis.py file, namely for handling other forms of data.

An important note: The IRSystems contains two documented code blocks for the creation of a postings file, and an indexes file, mainly because it's more efficient, but it's also a pain to upload larger file to github. Once the files are created, these code blocks can be commented out for a massively significant performance increase.

Summarization of Files:
- IR_Project1_SourceCode_Collection-Analysis.py -> Contains analysis on the Game Development Stack Exchange data.
- IR_Project1_SourceCode_IRSystems.py ->  Implementations and evaluation of boolean retrieval & simple inverted index systems over the collection.
- Post.py -> Utility code for reading post data.  (Not mine).
- post_parser_record.py -> Utility code for reading post data. (Not mine).

## Table of Contents - Assignments
### &nbsp;&nbsp;- Assignment 1
&nbsp;&nbsp;&nbsp;&nbsp; -> Question1.ipynb : 

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Basic Python **|** Debugging **|** Introduction to NLTK, pandas, matplotlib and NumPy **|** Data Handling

&nbsp;&nbsp;&nbsp;&nbsp; -> Question2.ipynb : 
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Precision & Recall **|** Exploratory Data Analysis **|** Boolean Retrieval

### &nbsp;&nbsp;- Assignment 2
&nbsp;&nbsp;&nbsp;&nbsp; -> Questions3_5.ipynb : 
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Inverted Index **|** Tokenization **|** Calculating Kendall Tau Correlation **|** Writing to File

&nbsp;&nbsp;&nbsp;&nbsp; -> indexes.tsv : 
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Saved Indexes From Inverted Index

&nbsp;&nbsp;&nbsp;&nbsp; -> qrels_assignment2 : 
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Generated qrels for queries from Questions3_5.ipynb 

&nbsp;&nbsp;&nbsp;&nbsp; -> assignment2_run.tsv : 
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ---> Generated results for queries from Questions3_5.ipynb (rank not considered, score is document count of terms)
