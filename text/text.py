import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# «агрузка NLTK ресурсов дл€ английского €зыка
nltk.download('punkt')
nltk.download('vader_lexicon')

def read_text(file_path, encoding='latin-1'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print("File not found.")
        return None

def count_words_and_sentences(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return len(words), len(sentences)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    file_path = input("Enter the file path of the text file: ")
    text = read_text(file_path)
    if text:
        word_count, sentence_count = count_words_and_sentences(text)
        sentiment = analyze_sentiment(text)

        print(f"Word count: {word_count}")
        print(f"Sentence count: {sentence_count}")
        print(f"Sentiment of the text: {sentiment}")

if __name__ == "__main__":
    main()
