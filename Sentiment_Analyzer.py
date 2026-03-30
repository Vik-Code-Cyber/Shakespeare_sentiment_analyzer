#Assignment 2: Sentiment Analysis using Shakespeare
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

text = """When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possessed,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,       
(Like to the lark at break of day arising
From sullen earth) sings hymns at heaven's gate;
For thy sweet love remembered such wealth brings
That then I scorn to change my state with kings."""

Pos_words=["fortune", "hope", "rich", "friends", "enjoy", "contented", "haply", "day", "arising", "sings", "hymns", "heaven", "sweet", "love","wealth", "kings"]
Neg_words=["disgrace", "alone", "beweep", "outcast", "trouble", "deaf", "bootless", "cries", "curse", "fate", "desiring", "despising", "sullen"]

def clean_text(text):
    #Convert to lowercase
    text = text.lower()
    #Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    #Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(sentence, pos_words, neg_words):
    cleaned = clean_text(sentence)
    tokens = word_tokenize(cleaned)

    pos_count = sum(1 for token in tokens if token in pos_words)
    neg_count = sum(1 for token in tokens if token in neg_words)
    score = pos_count - neg_count

    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return label, score, pos_count, neg_count


#Preprocessing text
cleaned_text = clean_text(text)
tokens = word_tokenize(cleaned_text)
sentences = sent_tokenize(text)

print("Cleaned Text:\n", cleaned_text)


#Sentence-level predictions (examples)
print("\nSentence-level predictions:")
sentence_results = []
for i, sentence in enumerate(sentences, start=1):
    label, score, pos_count, neg_count = predict_sentiment(sentence, Pos_words, Neg_words)
    sentence_results.append((label, score))
    print(f"{i}. {label} (score={score}, +{pos_count}/-{neg_count})")
    print(f"   \"{sentence}\"")


#Single-line sentiment analysis: line 1
line1 = "When, in disgrace with fortune and men's eyes,"
line_1_label, line_1_score, line_1_pos, line_1_neg = predict_sentiment(line1, Pos_words, Neg_words)

print("\nSelected line sentiment analysis (line 1):")
print(f"Line: \"{line1}\"")
print(f"Result: {line_1_label} (score={line_1_score}, +{line_1_pos}/-{line_1_neg})")


#Single-line sentiment analysis: line 2
line2 = "Wishing me like to one more rich in hope,"
line_2_label, line_2_score, line_2_pos, line_2_neg = predict_sentiment(line2, Pos_words, Neg_words)

print("\nSelected line sentiment analysis (line 2):")
print(f"Line: \"{line2}\"")
print(f"Result: {line_2_label} (score={line_2_score}, +{line_2_pos}/-{line_2_neg})")


#Overall sentiment for the full passage
overall_label, overall_score, overall_pos, overall_neg = predict_sentiment(text, Pos_words, Neg_words)

print("\nOverall Sentiment Output:")
print(f"Label: {overall_label}")
print(f"Score: {overall_score} (+{overall_pos}/-{overall_neg})")


#Short explanation of approach
"""1) Preprocess text by lowercasing, removing punctuation, and normalizing spaces.
2) Tokenize text/sentences using NLTK.
3) Use rule-based scoring: +1 for each positive word match, -1 for each negative word match.
4) Classify sentiment as Positive/Negative/Neutral from final score."""
