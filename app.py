# This is basically the heart of my flask
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from utils import *
import pickle
import warnings
import time
import sklearn

warnings.filterwarnings("ignore")

app = Flask(__name__)  # intitialize the flaks app  # common
df = pd.read_csv(r'./dataset/sample30.csv', low_memory=False)

#### Load the recommendation, word_vector and sentiment models
sentiment_model = pickle.load(open("./pickle/sentiment_model.pkl", 'rb'))
word_vectorizer = pickle.load(open("./pickle/tfidf_vector.pkl", 'rb'))
user_recommendation_df = pd.read_pickle("./pickle/user_recommendation.pkl")

#user_input = "00sab00"
#top20_product_list = list(user_recommendation_df.loc[user_input].sort_values(ascending=False)[0:20].index)
#print(top20_product_list)
#top20_df = fetch_all_reviews_for_the_given_products(df, top20_product_list)
#top20_df = apply_pre_processing_on_df(top20_df)
#print("\n PRE processing completed \n")

#print("\n ######### TO BE REMOVED ########## \n")
#top20_df = top20_df.head(10)

#top20_df = apply_nlp_processing_on_df(top20_df)
#print("\n NLP processing completed \n")
#top20_df["user_sentiment"] = predict_sentiment_on_reviews(word_vectorizer, sentiment_model, top20_df["reviews_text_title"].tolist())
#top5_product_list = get_top5_product_list(top20_df)
#print(top5_product_list)
#exit()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        print("You are in GET method")
        return render_template('sample.html')
    else:
        user_input = str(request.form.get('user_name'))
        print("You are in PUT method", user_input)
        top20_product_list = list(user_recommendation_df.loc[user_input].sort_values(ascending=False)[0:20].index)
        print("TOP 20 Product list:")
        print(top20_product_list)
        top20_df = fetch_all_reviews_for_the_given_products(df, top20_product_list)
        start_time = time.time()
        print("Pre processing started: ", start_time)
        top20_df = apply_pre_processing_on_df(top20_df)
        print("Pre processing Ended: ", time.time() - start_time)
        start_time = time.time()
        print("NLP processing started: ", start_time)
        top20_df = apply_nlp_processing_on_df(top20_df)
        print("NLP processing Ended: ", time.time() - start_time)
        start_time = time.time()
        print("Prediction processing started: ", start_time)

        top20_df["user_sentiment"] = predict_sentiment_on_reviews(word_vectorizer, sentiment_model,
                                                                  top20_df["reviews_text_title"].tolist())
        top5_product_list = get_top5_product_list(top20_df)
        print("Prediction Ended: ", time.time() - start_time)

        return render_template('sample.html', prod1=top5_product_list[0],
                               prod2=top5_product_list[1],
                               prod3=top5_product_list[2],
                               prod4=top5_product_list[3],
                               prod5=top5_product_list[4])


# If request.method == ‘GET’:
# return render_template("view.html”)
# Any HTML template in Flask App render_template
if __name__ == '__main__':
    app.run(debug=True)  # this command will enable the run of your flask app or api
    # ,host="0.0.0.0")
