import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme(context='paper')

# Articles by media
all_news_articles = pd.read_csv('Coding/all_news_articles.csv')
articles_outlet = all_news_articles.value_counts('Newspaper')
sns.barplot(x=articles_outlet, y=articles_outlet.index)
sns.despine()
plt.gca().set(title='News articles by newspaper on asylum seekers in 2019',
              xlabel='Count', ylabel='')
# plt.yticks(rotation=45)
for idx, value in enumerate(articles_outlet):
    plt.text(value + 5, idx + 0.2, value)
plt.tight_layout()
plt.show()
plt.clf()
