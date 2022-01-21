# This is basically the heart of my flask
from flask import Flask, render_template, request, redirect, url_for
from utils import *
import pickle
import warnings



warnings.filterwarnings("ignore")

app = Flask(__name__)  # intitialize the flaks app  # common
df = pd.read_csv(r'./dataset/sample30.csv', low_memory=False)

#### Load the recommendation, word_vector and sentiment models
sentiment_model = pickle.load(open("./pickle/sentiment_model.pkl", 'rb'))
word_vectorizer = pickle.load(open("./pickle/tfidf_vector.pkl", 'rb'))
user_recommendation_df = pd.read_pickle("./pickle/user_recommendation.pkl")




@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        user_input = str(request.form.get('user_name'))
        top20_product_list = list(user_recommendation_df.loc[user_input].sort_values(ascending=False)[0:20].index)
        print(top20_product_list)
        top20_df = fetch_all_reviews_for_the_given_products(df, top20_product_list)
        top20_df = apply_pre_processing_on_df(top20_df)
        top20_df = apply_nlp_processing_on_df(top20_df)
        top20_df["user_sentiment"] = predict_sentiment_on_reviews(word_vectorizer, sentiment_model,
                                                                  top20_df["reviews_text_title"].tolist())
        top5_product_list = get_top5_product_list(top20_df)


        return render_template('index.html', prod1=top5_product_list[0],
                               prod2=top5_product_list[1],
                               prod3=top5_product_list[2],
                               prod4=top5_product_list[3],
                               prod5=top5_product_list[4])


if __name__ == '__main__':
    app.run(debug=True)  # this command will enable the run of your flask app or api

