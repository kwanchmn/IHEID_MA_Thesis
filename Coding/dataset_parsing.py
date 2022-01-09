import glob
import pandas as pd
import re
import numpy as np

# Parsing the text files of news with regex


def parse_text(path):
    replace_regex = r"^Total.*|^={2,}|^DOCUMENT ID:.*|^Source:.*|^Wisers.*|^Copyright\s\(c\).*|\(?https://.+\)?|^Text\sSnapshot:.*|\(?.+@.+\.[a-z]+\)?"
    with open(path, "r", encoding='utf-8') as file:
        text = [line.strip('\n').strip() for line in file.readlines()]
        text = [*map(lambda line: re.sub(replace_regex, '', line), text)]
        text = [*filter(None, text)]
        text = ''.join([' split ' if re.match(
            r'^-+', line) else line for line in text])
        text = re.split(r'\ssplit\s', text)
    metadata_regex = r"\d+\.\s+(.*)\s+-\s+\((.*)\)\s+(\d{4}-\d{2}-\d{2})\s+(.*)"
    metadata_list = []
    main_text_list = []
    for i in range(0, len(text)-1, 2):
        metadata_list.extend(re.findall(metadata_regex, text[i]))
        main_text_list.append(text[i+1])
    return [*map(lambda x: str(x).strip('()'), metadata_list)], main_text_list

# Making the parsed text into a dataframe


def make_df(metadata, news_text):
    news_df = pd.DataFrame()
    news_df[['Newspaper', 'Title', 'Date', 'Category_data']] = pd.Series(metadata)\
        .str.split(', ', expand=True)\
        .apply(lambda x: x.str.strip("'"))
    news_df['Text'] = news_text
    news_df['Date'] = pd.to_datetime(news_df.Date)
    return news_df

# Creating the dataframe


news_df = pd.DataFrame()
news_text = glob.glob(r'news_articles/*.txt')

for file in news_text:
    parsed_file_metadata, parsed_file_text = parse_text(file)
    file_df = make_df(parsed_file_metadata, parsed_file_text)
    news_df = news_df.append(file_df)

news_df.drop_duplicates(
    subset=['Title'], inplace=True)

# Cleaning the data
news_df['Title'] = news_df['Title'].apply(
    lambda x: ','.join(x)).str.replace(r"[,\\u30]", "", regex=True).str.strip()
metadata = news_df.Category_data.str.split(' ', expand=True)
news_df[['Category', 'Page_number']] = metadata.iloc[:, 0:2]
news_df.drop(columns='Category_data', inplace=True)
news_df.sort_values(
    by=['Date', 'Newspaper'], inplace=True, ignore_index=True)

# Rearranging the columns' order
column_order = ['Newspaper', 'Date', 'Title',
                'Text', 'Category', 'Page_number']
news_df = news_df[column_order]
news_df['Newspaper'] = news_df['Newspaper'].str.replace(
    r"\s\(Chinese\)", "", regex=True)

# Political camps of the news outlets
pro_democracy = set(['Apple Daily', 'Kung Kao Po'])
neutral = set(['Ming Pao Daily News', 'Metro Daily'])
polit_camp_dict = {
    newspaper: 'Pro-democracy' if newspaper in pro_democracy else 'Neutral' if newspaper in neutral else 'Pro-Beijing' for newspaper in news_df.Newspaper.unique()}
news_df['Political_camp'] = news_df.Newspaper.map(
    polit_camp_dict)

# Filtering articles closely related to asylum seekers in HK
asylum_seeker_keywords = r"聲請|尋求庇護|難民|行街紙"
title_mask = news_df.Title.str.contains(asylum_seeker_keywords, regex=True)
text_mask = news_df.Text.str.contains(asylum_seeker_keywords, regex=True)
news_df_filtered = news_df.query(
    "@title_mask or @text_mask").reset_index(drop=True)

# Filtering out marginal articles of non-refoulement claimants
# fake_refugee_once = news_df_filtered[(news_df_filtered.Text.str.count(
#     '假難民') == 1) & (~news_df_filtered.Text.str.contains(r"南亞兵團/假難民近年犯案事件簿"))]
fake_refugee_once_df = pd.read_csv("fake_refugee_once.csv")
fake_refugee_filter_index = fake_refugee_once_df.query("Filter == 1").Title
news_df_filtered = news_df_filtered[~news_df_filtered.Title.isin(
    fake_refugee_filter_index)]

# Creating features
news_df_filtered['Raw_article_length'] = news_df_filtered.Text.apply(len)
news_df_filtered['Title_length'] = news_df_filtered.Title.apply(len)
news_df_filtered['Month'] = news_df_filtered['Date'].dt.month

# Auto coding the racial labels
racial_label_string = r"南亞|非洲|越南|印尼|菲律賓|印度|孟加拉|巴基斯坦|泰國|尼日利亞|斯里蘭卡|印裔|巴裔|非華|哥倫比亞|委內瑞拉"
racial_label_condition = news_df_filtered.Text.str.contains(
    racial_label_string) | news_df_filtered.Title.str.contains(racial_label_string)
news_df_filtered['Racial_label'] = np.where(racial_label_condition, 1, 0)

# Auto coding the sentiments
sentiment_fake_refugee = (news_df_filtered.Title.str.contains(
    "假難民") | news_df_filtered.Text.str.contains("假難民"))
news_df_filtered['Sentiment'] = np.where(sentiment_fake_refugee, 0, "")

# Dropping the news from Hong Kong Government News which are records of government speeches rather than news
non_news_gov = news_df_filtered.Title.str.contains(
    r"立法會.+題|發言|回顧|演辭", regex=True)
news_df_filtered = news_df_filtered[~news_df_filtered.Title.isin(
    news_df_filtered[non_news_gov].Title)].reset_index(drop=True)

# Porting the dataset to csv file
# news_df_filtered.to_csv('asylum_seekers_articles.csv', encoding='utf-8-sig')
