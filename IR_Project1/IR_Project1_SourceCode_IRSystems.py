# Ryan Reed
# COS 470
# Code for parts 2-4
# Boolean Search, Inverted Index, and Querying
########################################################################################################################
# importing
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from post_parser_record import PostParserRecord
from collections import defaultdict
from heapq import nlargest
import re
from ast import literal_eval
import csv
import json
import time
from sklearn.metrics import ndcg_score, dcg_score

# initialization of the post_reader object to read in post data
post_reader = PostParserRecord("Posts.xml")
# initializing the set of stopwords
stop_words = set(stopwords.words('english'))
########################################################################################################################
# utility function for parsing and cleaning text.
def text_processor(input_text, remove_stopwords):
    # filters out xml tags
    input_text = re.sub('<[^<]+>', "", input_text)
    input_text = re.sub(r"[(,,.;@/>¤\\#'}{’//=\"-:?*\[\]<!&$)]+\ *", " ", input_text.lower())
    list_of_words = word_tokenize(input_text)
    if remove_stopwords == True :
        list_of_words = [x for x in list_of_words if (x not in stop_words) and (x != 'p')]
    else:
        list_of_words = [x for x in list_of_words if (x != 'p')]
    return list_of_words
########################################################################################################################
# Boolean Search System

# this block of code creates a tsv file with the list of postings. for time efficiency, the search system accesses the
# tsv instead of creating a new list of postings.
# uncomment this to create the postings file

"""for question in post_reader.map_questions:
  post_contents = post_reader.map_questions[question].title + " " + post_reader.map_questions[question].body
  cleaned_post_contents = text_processor(post_contents, True)

  curr_post_id = post_reader.map_questions[question].post_id
  for word in cleaned_post_contents:
    if word not in postings_dict:
      postings_dict[word] = []
    if curr_post_id not in postings_dict[word]:
      postings_dict[word].append(curr_post_id)

for answer in post_reader.map_just_answers:
  post_contents = post_reader.map_just_answers[answer].body
  cleaned_post_contents = text_processor(post_contents, True)

  curr_post_id = post_reader.map_just_answers[answer].post_id
  for word in cleaned_post_contents:
    if word not in postings_dict:
      postings_dict[word] = []
    if curr_post_id not in postings_dict[word]:
      postings_dict[word].append(curr_post_id)

df_postings = pd.DataFrame(postings_dict.items(), columns=['Word', 'post_ids'])
df_postings.to_csv('postings.tsv', sep="\t", index = False)"""

# loading the created tsv into a dataframe
postings_df = pd.read_csv('postings.tsv', sep='\t')
postings_df['post_ids'] = postings_df['post_ids'].apply(literal_eval)

# intersecting two postings lists
def intersect(p1, p2, operation):
  result = list()
  if operation == 'AND': # if operation to perform is AND
    result = [post_id_val for post_id_val in p1 if post_id_val in p2]
  elif (operation == 'OR'): # if operation to perform is OR
        list_union = p1 + p2
        result = [post_id_val for post_id_val in list_union]
  return result

# handles queries, utilizes the intersect function above to compute the resulting indices of the boolean search.
def query_handler(query, operation):
    in_tokens = text_processor(query, True)
    results = list()
    list_query_results = list()

    for token in in_tokens:
        query_string = '(Word == "{}")'.format(token)
        list_query_results.append(postings_df.query(query_string)['post_ids'])
    results = list_query_results[0].tolist()[0]
    del list_query_results[0]
    # this loop performs the continued intersection (or union) of each token of the query, to create a single
    # appropriate query result
    while ((len(list_query_results)) != 0 and ((len(results)) != 0)):
        results = intersect(results, list_query_results.pop().tolist()[0], operation)
    return sorted(set(results))
########################################################################################################################
# Simple Inverted Index Retrieval
# function for building the inverted index, following code block is commented out.
# It is quicker to load the index from the file created rather than repeatedly create the index.
# If an index file is needed, uncomment this and it will create one for retrieval.
"""def build_index():
    index_dict = dict()
    for post in post_reader.map_questions: # code block for questions
        post_text = post_reader.map_questions[post].title + " " + post_reader.map_questions[post].body
        list_of_words = text_processor(post_text, True)
        set_of_words = set(list_of_words)
        for token in set_of_words:
            term_frequency = list_of_words.count(token)
            if token not in index_dict:
                index_dict[token] = list()
            index_dict[token].append({post_reader.map_questions[post].post_id : term_frequency})

    for post in post_reader.map_just_answers: # code block for answers
        post_text = post_reader.map_just_answers[post].body
        list_of_words = text_processor(post_text, True)
        set_of_words = set(list_of_words)
        for token in set_of_words:
            term_frequency = list_of_words.count(token)
            if token not in index_dict:
                index_dict[token] = list()
            index_dict[token].append({post_reader.map_just_answers[post].post_id : term_frequency})
    return index_dict
# writing indexes to tsv file
# indexes_dict = dict()
built_index = build_index()
json.dump(built_index, open('inverted_indexes.tsv', 'w', encoding="utf-8")) """

# loading data from the built index
inverted_index = json.load(open('inverted_indexes.tsv', 'r', encoding="utf-8"))

# inverted index searching (term-at-a-time)
def index_search(query_string):
    list_of_terms = text_processor(query_string, True)
    document_scores = dict()
    for term in list_of_terms:
        if term in inverted_index:
            for x in inverted_index[term]:
                for key,value in x.items():
                    if key not in document_scores:
                        document_scores[key] = 0
                    document_scores[key] += value

    return sorted(document_scores, key=document_scores.get, reverse=True)[:50]
########################################################################################################################
# Building Query Set

# assembling the set of queries, wherein the queries are a sample of 20 randomly selected question titles
query_set = [
    'How will I make my character jump only when it is pressed once',
    'Understanding UV coordinates',
    'Languages to Learn For Game Development',
    'How can I effectively manage a hobby game project?',
    'Why is it so expensive to develop an MMO?',    # 5

    'What are the most popular software development methodologies used by game studios?',
    'Tools for creating 2d tile based maps',
    'What are some good resources for building a voxel engine?',
    'What are some common ways to generate revenue from a free game?',
    'What is a good algorithm to detect collision between moving spheres?', # 10

    'Free ebooks about game development',
    'Pros and Cons of Various 3D Game Engines',
    'Is it reasonable to write a game engine in C?',
    'What are the challenges and benefits of writing games with a functional language?',
    'Why is there a lack of games for Linux?',  # 15

    'What are some ways to prevent or reduce cheating in online multiplayer games?',
    'Good resources for learning about game architecture?',
    'Effective marketing strategies for independent game projects',
    'Blender For Game Development, Pros And Cons',
    'What are the makings of a good Character'  # 20
]
########################################################################################################################
# Evaluation Sections Ahead

# Relevancy scores are as follows:
# Relevant -> 2
# Partially Relevant -> 1
# Not Relevant -> 0
########################################################################################################################
# Evaluation of Systems - Boolean Search System
bool_system_res = dict()
time_list = list()
for query in query_set:
    start = time.time()
    query_res = query_handler(query, 'OR')
    if len(query_res) > 10:
        bool_system_res[query] = query_res[:10]
    else:
        bool_system_res[query] = query_res
    end = time.time()
    time_list.append(end - start)

"""for x in bool_system_res:
    print(x, "->", bool_system_res[x], "\n")"""
print("Average Boolean retrieval time :", (sum(time_list) / len(time_list)))

# calculating average P@5
precision_total = (((2 + 3 + 6 + 4 + 6 + 1 + 4 + 2 + 2 + 3) / 10)/20)
print("Boolean Average P@5 : ", precision_total)

# calculating average P@10
precision_total = (((2 + 4 + 5 + 8 + 12 + 8 + 2 + 2 + 7 + 4 + 2 + 2 + 3) / 20)/20)
print("Boolean Average P@10 : ", precision_total)

# storing relevancy scores
relevancies = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 2, 0, 0, 0],
    [2, 0, 0, 0, 1, 0, 0, 0, 0, 2],
    [1, 0, 2, 1, 2, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 2, 2, 0, 2, 2, 2],
    [1, 2, 1, 2, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# computing list of nDCG values
ideal_order_relevancies = [sorted(item, reverse=True) for item in relevancies]
ndcg_list = list()

for ideal, relevance in zip(ideal_order_relevancies, relevancies):
    ndcg_list.append(ndcg_score(np.array([ideal]), np.array([relevance])))

# computing nDCG at cut 5
ndcg_list_5 = ndcg_list[:5]
print("Boolean Average nDCG @ 5 :", (sum(ndcg_list_5) / len(ndcg_list_5)))

# computing nDCG at cut 10
print("Boolean Average nDCG @ 10 :", (sum(ndcg_list) / len(ndcg_list)))

# choosing query 5 for dual assessment through a peer (Cohen's Kappa)
personal_assessment = 12/20
other_assessment = 8/20
agreement_kappa = 1 - ((1 - personal_assessment)/ (1 -other_assessment))
print("Boolean Agreement Measure :", agreement_kappa, "\n")

########################################################################################################################
# Evaluation of Systems - Simple Inverted Index
II_system_res = dict()
time_list = list()
for query in query_set:
    start = time.time()
    query_res = index_search(query)
    II_system_res[query] = query_res[:10]
    end = time.time()
    time_list.append(end - start)

print("Average Inverted Index retrieval time :", (sum(time_list) / len(time_list)))

# calculating average P@5
precision_total = ((4 + 6 + 1 + 6 + 6 + 3 + 2 + 7 + 3 + 2 + 5 + 2)/10)/20
print("Inverted Index Average P@5 : ", precision_total)

# calculating average P@10
precision_total = ((8 + 8 + 1 + 12 + 8 + 6 + 5 + 2 + 13 + 6 + 5 + 6 + 4)/20)/20
print("Inverted Index Average P@10 : ", precision_total)

# storing relevancy scores
relevancies2 = [
    [1, 1, 1, 1, 0, 2, 2, 0, 0, 0],
    [1, 1, 2, 2, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 1, 2, 1, 1, 2, 1, 0, 2],
    [2, 0, 1, 1, 2, 1, 0, 1, 0, 0],
    [2, 0, 0, 0, 1, 2, 1, 0, 0, 0],
    [0, 0, 2, 0, 0, 1, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [2, 2, 2, 0, 1, 2, 2, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 1, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# computing list of nDCG values
ideal_order_relevancies2 = [sorted(item, reverse=True) for item in relevancies2]
ndcg_list = list()

for ideal, relevance in zip(ideal_order_relevancies2, relevancies2):
    ndcg_list.append(ndcg_score(np.array([ideal]), np.array([relevance])))

# computing nDCG at cut 5
ndcg_list_5 = ndcg_list[:5]
print("Inverted Index Average nDCG @ 5 :", (sum(ndcg_list_5) / len(ndcg_list_5)))

# computing nDCG at cut 10
print("Inverted Index Average nDCG @ 10 :", (sum(ndcg_list) / len(ndcg_list)))

# choosing query 5 for dual assessment through a peer (Cohen's Kappa)
personal_assessment = 12/20
other_assessment = 9/20
agreement_kappa = 1 - ((1 - personal_assessment)/ (1 -other_assessment))
print("Inverted Index Agreement Measure :", agreement_kappa)