# Preliminaries
import jieba
import zhon.hanzi
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import glob
import os

# Setting directory

os.getcwd()
master_directory = r"C:\Users\kenji\Desktop\Thesis"
os.chdir(master_directory)

# Defining a function for reading text files


def read_txt(path):
  with open(path, "r", encoding="utf-8") as txt:
    file = txt.readlines()
    file = [word.strip("\n") for word in file]
    return file


# Creating traditional Chinese stop word list
stop_words_cantonese = read_txt("stopwords.txt")

# Punctuations
punctuations = [punc for punc in zhon.hanzi.punctuation]

# Combining stop words with punctuations
stop_words_full = list(
    set(stop_words_cantonese + punctuations))

# Adding Hong-Kong-politics-related words into the dictionary for better tokenisation

# Loading the names of the word lists
word_list_path = r"C:\Users\kenji\Desktop\Thesis\HKPolDict-master"
os.chdir(word_list_path)
word_file_list = glob.glob("*.txt")
word_file_list
os.chdir(master_directory)
os.getcwd()

# Defining the parts of speech
word_tag_list = ["ORG", "ORG", "ORG", "PER", "LOC", "nz", "nz"]
word_tag_dict = dict(zip(word_file_list, word_tag_list))

# Adding words to the dictionary

for doc, pos in word_tag_dict.items():
  for word in doc:
    jieba.add_word(word, tag=pos)

# Adding asylum-seeker-related words to the dictionary
asylum_seeker_words = read_txt("Asylum_seeker_words.txt")

for word in asylum_seeker_words:
  jieba.add_word(word, tag="nz")

# Removing digits, punctuations and spaces in the text
def preprocess(doc):
  regex = r"[\d+\s+\n\t]|[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[【】╮╯▽╰╭★→「」]+|[！，❤。～《》：（）【】「」？”“；：、]"
  return re.sub(regex, "", doc)

# Trying to tokenise an article
news = ["""假難民濫用免遣返聲請程序禍港多年，不但拖延留港打黑工及作奸犯科，更連累本港於過去7個財政年度枉花60億元公帑。港府為打擊假難民問題，提出的《2021年入境（修訂）條例》草案已於今年4月獲立法會通過，在5大範疇針對性推出最少10招改善免遣返聲請安排，包括防止假難民拖延留港、改善酷刑聲請上訴委員會的程序、加強羈留、加快遣返及從源頭堵截假難民等，相關法例將於下月1日起生效。有學者認為，今次修例如「曙光」一樣，展示保安局及入境處杜絕假難民問題的決心。""",
        """本港面對「難民」問題除之不去，早於30多年已為越南難民問題賠上11億港元，豈料舊債未清，30多年後新債繼續來。港人要為假難民「埋單」，估算過去7個財政年度，最少狂燒公帑超過60億港元，更甚者是，南亞幫假難民藉酷刑聲請滯留香港長達數年，無惡不作成為本港毒瘤，帶來嚴重治安問題。
假難民禍港狂燒公帑，根據2020/21年度的保安局開支預算，預料出入境管制撥款較2019/20年度的修訂預算，增加27%，包括為免遣返聲請人士提供公費法律支援，預計2020/21年度，由當值律師服務運作的「法律支援免遣返聲請計劃」開支約為9227萬港元，較上一年度上升36%。
根據司法機構的開支預算，去年終審法院處理的上訴許可申請是490多宗，當中與免遣返聲請個案有關的上訴許可申請數目近390宗、即佔去年申請總數的79%。至於較高級別法院需處理的免遣返聲請案件數目激增，其中高等法院上訴法庭，去年涉及免遣返聲請的上訴案件有351宗。另按入境處開支預計，今年提出的免遣返聲請個案有1100宗。而連同之前6個財政年度用於假難民開支，估計高達60億港元。
議員葛珮帆表示，公帑應用於市民所需，更何況假難民根本並非真正難民，他們來港作奸犯科、做黑工，其實是經濟難民；不但濫用公帑，更破壞本港治安，認為政府「不應用石頭砍自己雙腳」，不要濫用港人的仁慈，犧牲港人的利益，必需積極尋求解決方法，盡快處理假難民問題。"""]
news_df = pd.DataFrame(news, columns=["news"])


# Now using jieba with sklearn as the tokenizer
def tokenize_zh(doc):
  return jieba.cut(doc)

# BoW
count = CountVectorizer(tokenizer=tokenize_zh,
                        stop_words=stop_words_full, preprocessor=preprocess)
bow = count.fit_transform(news_df["news"])
tokens_count = pd.DataFrame(
    bow.toarray(), columns=count.get_feature_names_out())
tokens_count

# Tfidf
tfidf = TfidfVectorizer(tokenizer=tokenize_zh,
                        stop_words=stop_words_full, preprocessor=preprocess)
tfidf.fit(news_df["news"])
tfidf_result = tfidf.transform(news_df["news"])
tokens_tfidf = pd.DataFrame(tfidf_result.toarray(),
                            columns=tfidf.get_feature_names_out())
tokens_tfidf
