import pandas as pd
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def load_data(file_path, col):
    folder_dir = 'data'
    path = folder_dir + '/' + file_path
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    ans = df[col].values.copy()

    return ans


# lowercase, remove number + punc, stemming
def clear_text(text):
    # lowercase
    text = text.lower()

    # remove word if len(word)<2
    text = " ".join([word for word in text.split() if len(word) >= 2])

    # remove punctuation
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    # remove number
    text = text.translate(str.maketrans(' ', ' ', string.digits))

    # remove html tags, url
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)

    # remove noun
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    filtered_words = [word for word, tag in tagged_words if tag not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']]
    text = " ".join(filtered_words)

    # stemming and lematization
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = " ".join([ps.stem(word) for word in text.split()])
    text = " ".join([lemmatizer.lemmatize(word, pos="a") for word in text.split()])

    return " ".join(text.split()).strip()


if __name__ == '__main__':
    data = load_data('trainer.csv', col='text')
    data = map(clear_text, data)
    print(max(data, key=len))
