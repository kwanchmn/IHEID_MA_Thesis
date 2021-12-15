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


news_articles_all = pd.DataFrame()
news_text = glob.glob(r"news_articles/*.txt")

for file in news_text:
    parsed_file_metadata, parsed_file_text = parse_text(file)
    file_df = make_df(parsed_file_metadata, parsed_file_text)
    news_articles_all = news_articles_all.append(file_df)

news_articles_all.drop_duplicates(
    subset=['Title'], inplace=True)

# Cleaning the data
news_articles_all['Title'] = news_articles_all['Title'].apply(
    lambda x: ','.join(x)).str.replace(r"[,\\u30]", "", regex=True).str.strip()
metadata = news_articles_all.Category_data.str.split(' ', expand=True)
news_articles_all[['Category', 'Page_number']] = metadata.iloc[:, 0:2]
news_articles_all.drop(columns='Category_data', inplace=True)
news_articles_all.sort_values(
    by=['Date', 'Newspaper'], inplace=True, ignore_index=True)

# Rearranging the columns' order
column_order = ['Newspaper', 'Date', 'Title',
                'Text', 'Category', 'Page_number']
news_articles_all = news_articles_all[column_order]
news_articles_all['Newspaper'] = news_articles_all['Newspaper'].str.replace(
    r"\s\(Chinese\)", "", regex=True)

# Creating features
news_articles_all['Raw_article_length'] = news_articles_all.Text.apply(len)
news_articles_all['Title_length'] = news_articles_all.Title.apply(len)
news_articles_all['Month'] = news_articles_all.Date.dt.month
news_articles_all['Protest'] = np.where(news_articles_all.Month >= 6, 1, 0)
pro_democracy = set(['Apple Daily', 'Kung Kao Po'])
neutral = set(['Ming Pao Daily News', 'Metro Daily'])
polit_camp_dict = {newspaper: 'Pro-democracy' if newspaper in pro_democracy else 'Neutral' if newspaper in neutral else 'Pro-Beijing' for newspaper in news_articles_all.Newspaper.unique()}
news_articles_all['Political_camp'] = news_articles_all.Newspaper.map(
    polit_camp_dict)
news_articles_all.to_csv('all_news_articles.csv', encoding='utf-8')
