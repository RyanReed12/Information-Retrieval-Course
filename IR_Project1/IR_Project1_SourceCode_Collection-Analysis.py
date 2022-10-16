# Ryan Reed
# COS 470 - Project 1
# This source code file contains the analysis of the collection.
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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import itertools
import textstat

# initialization of the post_reader object to read in post data
post_reader = PostParserRecord("Posts.xml")
# initializing the set of stopwords
stop_words = set(stopwords.words('english'))

# this is a utility function
# this function processes an input string, and removes stopwords based on a boolean passing
# performing punctuation removal, conversion to lowercase, stopword removal, and tokenization
# returned is a list of the tokens from the input string after processing
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
# step 3: analyzing the collection
########################################################################################################################
# step 3.1
# this function computes a word frequency dictionary from
# the 20 most common terms in the collection.
def step3_1_common_terms(remove_stopwords):
    res_dict = dict()
    for post in post_reader.map_questions:  # loop for questions
        list_of_tokens = text_processor(post_reader.map_questions[post].title + " " + post_reader.map_questions[post].body, remove_stopwords)
        for token in list_of_tokens:
            if (token not in res_dict):
                res_dict[token] = 0
            res_dict[token] += 1

    for post in post_reader.map_just_answers:   # loop for answers
        list_of_tokens = text_processor(post_reader.map_just_answers[post].body, remove_stopwords)
        for token in list_of_tokens:
            if (token not in res_dict):
                res_dict[token] = 0
            res_dict[token] += 1
    # sort resulting dict in reverse order
    sorted_res_dict = ((dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))))
    return dict(list(sorted_res_dict.items())[:20])

common_terms_stopwords = step3_1_common_terms(False)
common_terms_no_stopwords = step3_1_common_terms(True)

# word cloud generation with stopwords
common_wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(common_terms_stopwords)
plt.imshow(common_wordcloud)
plt.title("WordCloud for Terms (Stopword Inclusive)")
plt.show()
# word cloud generation without stopwords
common_wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(common_terms_no_stopwords)
plt.imshow(common_wordcloud)
plt.title("WordCloud for Terms (Stopword Exclusive)")
plt.show()
# plot the words
plt.bar(common_terms_no_stopwords.keys(),common_terms_no_stopwords.values(), width=0.33, bottom=None)
plt.xticks(rotation = 90, fontsize=6)
plt.title("Frequency of Top-20 Words (Stopword Exclusive)")
plt.show()
########################################################################################################################
# step 3.2
# this function returns the dictionary of all
# question post tags, sorted by frequency
def get_tags():
    tag_dict = dict()
    for post in post_reader.map_questions:
        for tag in post_reader.map_questions[post].tags:
            if tag not in tag_dict:
                tag_dict[tag] = 0
            tag_dict[tag] += 1
    sorted_tag_dict = ((dict(sorted(tag_dict.items(), key=lambda item: item[1], reverse=True))))
    return sorted_tag_dict


# top-10 common tags
sorted_tags = dict(list(get_tags().items()))
print(dict(list(sorted_tags.items())[:10]))

# distribution visualization for top-20 tags + other tags
tag_dict_20 = dict(list(sorted_tags.items())[:20])
slice_sorted_tags = dict(itertools.islice(sorted_tags.items(), 20, None))
print(tag_dict_20)
print(slice_sorted_tags)
tag_dict_20['other'] = sum(slice_sorted_tags.values())

# re-sorting after calculating other
tag_dict_20 = ((dict(sorted(tag_dict_20.items(), key=lambda item: item[1], reverse=True))))

# resizing figure to make graphs cleaner and more legible
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(13)
plt.bar(tag_dict_20.keys(), tag_dict_20.values(), width=1, bottom=None)
plt.xticks(rotation = 90, fontsize=8)
plt.title("Distribution of Top-20 Post Tags")
plt.show()

########################################################################################################################
# step 3.3
# this function calculates the average number of words and sentences in questions (body and title)
# and answers.
def avg_num_words():
    # computing word count of the collection
    post_count, word_count, sentence_count = 0,0,0
    for post in post_reader.map_questions:
        post_count += 1
        post_text = post_reader.map_questions[post].body + post_reader.map_questions[post].title
        post_text_sent = re.split(r'[.!?]+', post_text)
        sentence_count += len(post_text_sent)
        if (len(post_text_sent) == 0):  # minimum of one sentence for both body and title, likely lacking punctuation
            sentence_count += 2
        tokens = text_processor(post_text, False)
        for token in tokens:
            word_count += 1

    for post in post_reader.map_just_answers:
        post_count += 1
        post_text = post_reader.map_just_answers[post].body
        post_text_sent = re.split(r'[.!?]+', post_text)
        sentence_count += len(post_text_sent)
        if (len(post_text_sent) == 0):  # minimum of one sentence
            sentence_count += 1
        tokens = text_processor(post_text, False)
        for token in tokens:
            word_count += 1

    return (word_count / post_count), (sentence_count / post_count)

avg_words, avg_sentences = avg_num_words()
print("Average number of words :", avg_words, "\nAverage number of sentences :", avg_sentences)
########################################################################################################################
# step 3.4
# the following function computes:
# 1) avg. number of answers
# 2) number of questions with no answer
# 3) number of questions with an accepted answer
def answers_func():
    num_answers, post_count, q_nanswers, q_acptdanswer = 0,0,0,0
    for post in post_reader.map_questions:
        post_count += 1
        if post_reader.map_questions[post].accepted_answer_id != None:
            q_acptdanswer += 1;         # increment count for questions with accepted answers
        if post_reader.map_questions[post].answer_count == 0:
            q_nanswers += 1;            # increment count for questions without answers
        else:
            num_answers += post_reader.map_questions[post].answer_count     # add count of answers for curr. post
    return num_answers / post_count, q_acptdanswer, q_nanswers

x1,x2,x3 = answers_func()
print("Average Number of Answers :", x1, "\nNumber of questions with accepted answers :", x2,
      "\nNumber of questions without answers :", x3)
########################################################################################################################
# step 3.5
# analysis is discussed in report, this function returns a sample of 5 questions without answers.
# this is a manual check, with a way to automatically access them, a personal choice was made here
# not to strip xml tags.
def sample_qnanswers():
    x = 0;
    list_of_post_text = list()
    for post in post_reader.map_questions:
        if (x >= 5):
            return 0
        if post_reader.map_questions[post].answer_count == 0:
            print("Post [", post_reader.map_questions[post].post_id, "]\nTitle:",
                  post_reader.map_questions[post].title, "\nBody :", post_reader.map_questions[post].body)
            x += 1;

sample_qnanswers()
########################################################################################################################
# step 3.6 utility function
# gathers user ids and their corresponding reputation score
import xml.etree.ElementTree as ET
def user_reader():
    user_dict = dict()
    tree = ET.parse("Users.xml")
    root = tree.getroot()
    for child in root:
        attr_dic = child.attrib
        user_id = int(attr_dic['Id'])
        user_rep = int(attr_dic['Reputation'])
        user_dict[user_id] = user_rep
    return user_dict

# step 3.6
# First function returns the following:
# 1) Count of accepted answers where they are the first answer
# Second function returns the following:
# 1) Is there a specific pattern in the reputation score of the person who answered the question?
# Third function turns the following:
# 1) Are always the accepted answers the ones with the highest score? If not, an example ID is returned.
# Fourth function is a caller for the three functions.
def accp_answer_count():
    count = 0
    for post in post_reader.map_questions:
        accepted_answer_id = post_reader.map_questions[post].accepted_answer_id;
        if post_reader.map_questions[post].answer_count != 0:
            if (post_reader.map_answers[post][0]).post_id == accepted_answer_id:
                count += 1;
    return count

def accp_answer_reputation(user_dict_input):
    this_list = list()
    this_list2 = list()
    for post in post_reader.map_questions:
        accepted_answer_id = post_reader.map_questions[post].accepted_answer_id;
        if post_reader.map_questions[post].answer_count != 0:
            if (post_reader.map_answers[post][0]).post_id == accepted_answer_id:
                user_id = (post_reader.map_answers[post][0]).owner_user_id
                if (user_id != None):
                    this_list.append(user_dict_input[user_id])
            else:
                user_id = (post_reader.map_answers[post][0]).owner_user_id
                if (user_id != None):
                    this_list2.append(user_dict_input[user_id])

    this_arr = np.array(this_list)
    this_arr2 = np.array(this_list2)
    x,y = np.array_split(this_arr[:-1], 2)
    # correlation between first answers that are accepted, and those that are not
    # length of accepted answers had to be 'sliced' so the correlation method would work
    print(np.corrcoef(this_arr[:18323], this_arr2))
    # splitting the reputation scores into two arrays, and checking correlation scores.
    print(np.corrcoef(x, y))
    # neither returned any correlative value indicating correlation

def accp_answer_alwaysHighest():
    for post in post_reader.map_questions:
        if (post_reader.map_questions[post].accepted_answer_id != None):
            accepted_answer_id = post_reader.map_questions[post].accepted_answer_id;
            accepted_answer_score = post_reader.map_just_answers[accepted_answer_id].score
            if post_reader.map_questions[post].answer_count > 1:
                for answer in post_reader.map_answers[post]:
                    if (answer.score > accepted_answer_score):
                        print("\nIt is not always the case that the accepted answer has the highest score. Example:\n")
                        print(answer.post_id, "\n Answer Body: ",answer.body)
                        return




count = accp_answer_count()
print("Total count of accepted answers that are the first answers: ", count)
accp_answer_reputation(user_reader())
accp_answer_alwaysHighest()
########################################################################################################################
# Step 3.7
# this function calculates the readability of all questions
def determine_readability():
    readability_list = list()
    numAnswers_list = list()
    for post in post_reader.map_questions:
        text = post_reader.map_questions[post].title + " " + post_reader.map_questions[post].body
        readability_list.append(textstat.textstat.flesch_reading_ease(text))
        if (post_reader.map_questions[post].answer_count == 0):
            numAnswers_list.append(0)
        else:
            numAnswers_list.append(1)

    return readability_list, numAnswers_list

readability_list, numAnswers_list = determine_readability()
readability_arr = np.array(readability_list)
numAnswers_arr = np.array(numAnswers_list)
print(np.corrcoef(readability_arr, numAnswers_arr))
# ~0.07307 score, indicating no correlation between question readability & the chance of the question being answered."""
########################################################################################################################
# Step 3.8
# this function returns the count of all duplicates, and provides one example
def duplicate_count():
    count = 0
    selectOne = False
    count_cmn_terms = 0;
    for post in post_reader.map_questions:
        if len(post_reader.map_questions[post].related_post) != 0:
            count += 1
            if (selectOne == False):
                related_post_id = post_reader.map_questions[post].related_post
                related_set_of_words = text_processor(post_reader.map_questions[related_post_id].title + " "
                                                      + post_reader.map_questions[related_post_id].body, False)
                post_set_of_words = text_processor(post_reader.map_questions[post].title + " "
                                                   + post_reader.map_questions[post].body, False)
                for word in post_set_of_words:
                    if word in related_set_of_words:
                        count_cmn_terms += 1
            selectOne == True
    return count, count_cmn_terms

cnt, cnt_cmn_trms = duplicate_count()
print("Total number of questions with duplicates :", cnt,
      "\nNumber of common terms between a single selected question and its duplicate :", cnt_cmn_trms)
# Interestingly, despite my error checking and checking post history, it appears that there are no strict duplicate
# questions, but there are 'possible duplicates', which do not fit the definition of being duplicates.

########################################################################################################################
# step 3.9 utility function
# reads the comments file
def get_comments():
    comment_dict = dict()
    tree = ET.parse("Comments.xml")
    root = tree.getroot()
    for child in root:
        attr_dic = child.attrib
        post_id = int(attr_dic['PostId'])
        text = (attr_dic['Text'])
        if (post_id not in comment_dict):
            comment_dict[post_id] = list()
        comment_dict[post_id].append(text)
    return comment_dict

# this function returns a list of comments
def get_question_comments(c_dict):
    count = 5
    q_comment_dict = dict()
    for post in post_reader.map_questions:
        curr_post_id = post_reader.map_questions[post].post_id
        if (count == 0):
            return q_comment_dict
        if post_reader.map_questions[post].comment_count > 0:
            if curr_post_id not in q_comment_dict:
                q_comment_dict[curr_post_id] = list()
            q_comment_dict[curr_post_id].append(c_dict[curr_post_id])

            count -= 1

comment_dict = get_comments()
q_comment_d = get_question_comments(comment_dict)
for x in q_comment_d:
    print("PostID", x, "->", q_comment_d[x], "\n")

########################################################################################################################
# 3.10
# Personal choice of collection analysis
# Explanation: A typical sign of a dedicated community, is involvement, I hypothesize that we can utilize individual
# users upvotes & downvotes as a metric of community involvement. For instance, users who are engaging more actively,
# are more likely to give user votes. I believe:
# (number of users with less than the mean votes)/(total number of users) = percentage of users where votes < mean votes
# and we can use this as a metric of vote disparity between the community, indicating member involvement.
import xml.etree.ElementTree as ET
def user_reader_votes():
    user_dict = dict()
    tree = ET.parse("Users.xml")
    root = tree.getroot()

    sum_of_all_votes = 0
    user_count = 0
    for child in root:
        attr_dic = child.attrib
        user_id =  int(attr_dic['Id'])
        user_votes = int(attr_dic['UpVotes']) + int(attr_dic['DownVotes'])
        user_dict[user_id] = user_votes

        sum_of_all_votes += user_votes
        user_count += 1
    mean_votes = sum_of_all_votes / user_count
    count_users_votes_less_mean = 0

    for x in user_dict:
        if user_dict[x] < mean_votes:
            count_users_votes_less_mean += 1

    return (count_users_votes_less_mean / user_count) * 100

print("{:.2f}".format(user_reader_votes()), "%")