import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Getting necessary modules
snowball = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()
stopword_corpus = set(stopwords.words('english'))

"""
Yang bisa ditambah, menjaga konteks, gimana caranya agar negative adverbs bisa tetep
ke detect sebagai pembuat negasi dari konteks kata setelahnya
"""

def clear_tweet_element_inline(line):
    result = []
    words = line.split()
    for word in words:
        if word[0] != '@' and word[0] != '#':
            result.append(word)
    return " ".join(result)

def clear_line(line):
    # Saving to new var
    line = line
    # Convert line to lower case
    line = line.lower()
    # Remove number, because it is simply irrelevant
    line = re.sub(r'\d+', ' ', line)
    # Remove HTML tags
    line = re.sub('<.*?>', ' ', line)
    # Remove punctuation
    line = re.sub(r'[^\w\s]', ' ', line)
    # Remove extra white spaces
    line = re.sub('\s\s+', ' ', line)

    return line

def stem_words(words):
    result = []
    for word in words:
        word = snowball.stem(word)
        result.append(word)

    return result


def lemmatize_words(words):
    result = []
    for word in words:
        word = lemmatizer.lemmatize(word)
        result.append(word)

    return result


def remove_stopwords(words):
    result = []
    for word in words:
        if(word not in stopword_corpus):
            result.append(word)

    return result

def preprocess(line):
    line = clear_tweet_element_inline(line)
    # Clearing line from unimportant chars
    line = clear_line(line)
    # Splitting line or sentence into list of words
    words = line.split()
    # Stemming list of words
    words = stem_words(words)
    # Lemmatizing list of words
    words = lemmatize_words(words)
    # Removing stopwords
    words = remove_stopwords(words)

    return words


def main():
    line = 'Box A @ExampleTag contains 3 red #ExampleHashtag and 5 white balls, ' \
           'while Box B contains 4 red and 2 blue balls.'
    print("=======================================")
    print(preprocess(line))


if __name__ == "__main__":
    main()
