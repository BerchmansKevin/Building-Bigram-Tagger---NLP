#!/usr/bin/env python
# coding: utf-8

# # `BERCHMANS KEVIN S`
# 
# 

# ##  Building Bigram Tagger - NLP

# #### EXERCISE - 1

# In[1]:


import nltk


# In[2]:


from nltk.tokenize import sent_tokenize,word_tokenize


# In[3]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[4]:


import nltk
nltk.download('punkt')


# In[5]:


text = word_tokenize("And now for something completely different")
nltk.pos_tag(text)


# #### Write down the expansion for CC,RB,.....JJ in the above output.

# ### Abbreviation-Meaning
# 
# CC  - coordinating conjunction
# 
# CD  - cardinal digit
# 
# DT  - determiner
# 
# EX  - existential there
# 
# FW  - foreign word
# 
# IN  - preposition/subordinating conjunction
# 
# JJ  - This NLTK POS Tag is an adjective (large)
# 
# JJR - adjective, comparative (larger)
# 
# JJS - adjective, superlative (largest)
# 
# LS  - list market
# 
# MD  - modal (could, will)
# 
# NN  - noun, singular (cat, tree)
# 
# NNS - noun plural (desks)
# 
# NNP - proper noun, singular (sarah)
# 
# NNPS- proper noun, plural (indians or americans)
# 
# PDT - predeterminer (all, both, half)
# 
# POS - possessive ending (parent\ ‘s)
# 
# PRP - personal pronoun (hers, herself, him, himself)
# 
# PRP$- possessive pronoun (her, his, mine, my, our )
# 
# RB  - adverb (occasionally, swiftly)
# 
# RBR - adverb, comparative (greater)
# 
# RBS - adverb, superlative (biggest)
# 
# RP  - particle (about)
# 
# TO  - infinite marker (to)
# 
# UH  - interjection (goodbye)
# 
# VB  - verb (ask)
# 
# VBG - verb gerund (judging)
# 
# VBD - verb past tense (pleaded)
# 
# VBN - verb past participle (reunified)
# 
# VBP - verb, present tense not 3rd person singular(wrap)
# 
# VBZ - verb, present tense with 3rd person singular (bases)
# 
# WDT - wh-determiner (that, what)
# 
# WP  - wh- pronoun (who)
# 
# WRB - wh- adverb (how)
# 

# #### EXERCISE - 2

# In[6]:


from nltk.corpus import brown


# In[7]:


nltk.download('brown')


# In[8]:


tagsen = brown.tagged_sents()
tagsen


# #### Prepare data sets
# 

# In[9]:


len(tagsen)


# In[10]:


br_train = tagsen[0:50000]
br_test = tagsen[50000:]
br_test[0]


# #### Build a bigram tagger
# 
# 

# In[11]:


t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(br_train, backoff=t0)
t2 = nltk.BigramTagger(br_train, backoff=t1)


# #### Evaluate

# In[12]:


t2.evaluate(br_test)


# #### Explore

# In[13]:


# 1. 

total_train = [len(l) for l in br_train]
sum(total_train)


# In[14]:


total_test = [len(l) for l in br_test]
sum(total_test)


# In[15]:


# 2.

t1.evaluate(br_test)


# In[16]:


t2.evaluate(br_test)


# In[17]:


# 3.

br_train[0]


# In[18]:


br_train[1277]


# In[19]:


br_train[1277] [11]


# In[20]:


# 4.

br_train_flat = [(word, tag) for sent in br_train for (word, tag) in sent]


# In[21]:


br_train_flat[:40]


# In[22]:


br_train_flat[13]


# In[23]:


# 5. a)

fd = nltk.FreqDist(br_train_flat)
cfd = nltk.ConditionalFreqDist(br_train_flat)


# In[24]:


cfd['cold'].most_common()


# In[25]:


# 5. b)

br_train_2grams = list(nltk.ngrams(br_train_flat, 2))
br_train_cold = [a[1] for (a,b) in br_train_2grams if b[0] == 'cold']
fdist =  nltk.FreqDist(br_train_cold)
[tag for (tag, _) in fdist.most_common()]


# In[26]:


# 5. c)

br_pre = [(w2+"/"+t2, t1) for ((w1,t1),(w2,t2)) in br_train_2grams]
br_pre_cfd =  nltk.ConditionalFreqDist(br_pre)
br_pre


# In[27]:


# 5. d)

br_pre_cfd['cold/NN'].most_common()


# In[28]:


br_pre_cfd['cold/JJ'].most_common()


# In[29]:


# 6.

bigram_tagger = nltk.BigramTagger(br_train)


# In[30]:


# 6. a)

text1 = word_tokenize('I was very cold.')
bigram_tagger.tag(text1)


# In[31]:


# 6. b)

text2 = word_tokenize('I had a cold.')
bigram_tagger.tag(text2)


# In[32]:


# 6. c)

text3 = word_tokenize('I had a severe cold.')
bigram_tagger.tag(text3)


# In[33]:


# 6. d)

text4 = word_tokenize('January was a cold month.')
bigram_tagger.tag(text4)


# In[34]:


# 7.


# In[35]:


# 8. a)

text5 = word_tokenize('I failed to do so.')
bigram_tagger.tag(text5)


# In[36]:


# 8. b)

text6 = word_tokenize('I was happy,but so was my enemy.')
bigram_tagger.tag(text6)


# In[37]:


# 8. c)

text7 = word_tokenize('So, how was the exam?')
bigram_tagger.tag(text7)


# In[38]:


# 8. d)

text8 = word_tokenize('The students came in early so they can get good seats.')
bigram_tagger.tag(text8)


# In[39]:


# 8. e)

text9 = word_tokenize('She failed the exam, so she must take it again.')
bigram_tagger.tag(text9)


# In[40]:


# 8. f)

text10 = word_tokenize('That was so incredible.')
bigram_tagger.tag(text10)


# In[41]:


# 8. g)

text11 = word_tokenize('Wow, so incredible.')
bigram_tagger.tag(text11)


# In[ ]:




