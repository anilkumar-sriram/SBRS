import pandas as pd
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.corpus import wordnet

lemmatizer = nltk.stem.WordNetLemmatizer()


def fetch_all_reviews_for_the_given_products(df, product_list):
    df = pd.read_csv(r'./dataset/sample30.csv', low_memory=False)
    common_df = df[df.name.isin(product_list)]
    return common_df


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def clean_text(text):
    text = text.lower()
    pattern = re.compile('[\(\[].*?[\)\]]')
    text = re.sub(pattern, '', text)
    pattern = re.compile('[^\w\s]')
    text = re.sub(pattern, '', text)
    pattern = re.compile('[0-9]')
    text = re.sub(pattern, '', text)
    return text


def apply_pre_processing_on_df(pre_process_df):
    pre_process_df = pre_process_df[["name", "reviews_text", "reviews_title"]]
    pre_process_df = pre_process_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    pre_process_df["reviews_title"].fillna(value="", inplace=True)
    pre_process_df["reviews_text_title"] = pre_process_df["reviews_text"] + " " + pre_process_df["reviews_title"]
    pre_process_df = pre_process_df[["name", "reviews_text_title"]]
    return pre_process_df


def apply_nlp_processing_on_df(nlp_process_df):
    nlp_process_df['reviews_text_title'] = nlp_process_df['reviews_text_title'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
    nlp_process_df['reviews_text_title'] = nlp_process_df['reviews_text_title'].apply(lambda x: clean_text(x))
    nlp_process_df['reviews_text_title'] = nlp_process_df['reviews_text_title'].apply(lambda x: lemmatize_sentence(x))
    return nlp_process_df


def predict_sentiment_on_reviews(word_vectorizer, sentiment_model, reviews):
    ## transforming the train and test datasets
    X_transformed = word_vectorizer.transform(reviews)
    y_pred_test = sentiment_model.predict(X_transformed)
    return y_pred_test


def get_top5_product_list(df):
    gb = df.groupby('name')
    counts = gb.size().to_frame(name='counts')
    gb2 = counts.join(gb.agg({'user_sentiment': 'sum'})).reset_index()
    gb2["percentage"] = gb2["user_sentiment"] / gb2["counts"]
    return list(gb2.sort_values('percentage', ascending=False).head(5)["name"])
