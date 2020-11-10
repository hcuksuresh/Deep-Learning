# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:44:46 2020

@author: sukandulapati
"""

import pandas as pd
guidelines = pd.read_csv('testfile_new3.csv', header=None)
guidelines.columns = ['w_id', 'g_value']

guidelines = guidelines.dropna()
#guidelines = guidelines.drop_duplicates()

import re
def clean(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.split()
    text = " ".join(text)
    return text

guidelines['g_value'] = guidelines['g_value'].astype(str)
guidelines['g_value'] = guidelines['g_value'].apply(lambda x: x.lower())
guidelines['g_value'] = guidelines['g_value'].apply(lambda x: clean(x))

guide_lines = guidelines
noprovision1 = guide_lines[guide_lines.g_value == 'no requirement']
guide_lines1 = guide_lines.loc[guide_lines.index.difference(noprovision1.index), ]
noprovision2 = guide_lines1[guide_lines1.g_value == 'no provision']
guide_lines_new = guide_lines1.loc[guide_lines1.index.difference(noprovision2.index), ]


strict = guide_lines_new[guide_lines_new['g_value'].str.contains(r'(?:\s|^)recommended(?:\s|$)')]
strict['label'] = 'recommended'

after_strict = guide_lines_new.loc[guide_lines_new.index.difference(strict.index), ]

after_strict['label'] = 'required'
noprovision1['label'] = 'not defined'
noprovision2['label'] = 'not defined'


import datetime as dt

frames = [noprovision1, noprovision2, strict, after_strict]
final_df = pd.concat(frames)
final_df = final_df[['w_id', 'label']]
final_df.columns = ['table_w_id', 'label']

final_df['table_name'] = 'covid_data.fact_guidelines'
final_df['algorithm'] = 'Guidelines-factclf'
final_df['label_type'] = 'categorization'
final_df['w_upsert_timestamp'] = pd.Series([dt.datetime.now()] * len(final_df))
    
final_df = final_df[['table_name', 'algorithm', 'table_w_id', 'label_type', 'label', 'w_upsert_timestamp']]
final_df.head()


final_df.to_csv('processedfile.csv', header=None, index=False)




from sklearn.model_selection import train_test_split
train, test = train_test_split(final_df, stratify = final_df['label'], test_size = 0.2, random_state = 0)

train.to_csv('train.csv', header=None, index=False)
test.to_csv('test.csv', header=None, index=False)