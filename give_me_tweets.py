from numpy import result_type
import pandas as pd
import credentials
import tweepy


auth = tweepy.OAuthHandler(credentials.API_Key, credentials.API_Secret_Key)
auth.set_access_token(credentials.Acces_token, credentials.Acces_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets(hastag):
    tuits_list = []

    filter_hastag = "#" + hastag + " -filter:retweets"

    # Usamos el Cursor dentro de un for
    for tweet in tweepy.Cursor(api.search_tweets,
                               q= filter_hastag,
                               lang="es",
                               tweet_mode="extended",
                               result_type='recent').items(250): #Cantidad de Tuits por busqueda

        # Agregamos el texto, fecha, likes, retweets y hashtags al array
            tuits_list.append([tweet.full_text,
                               tweet.user.screen_name, 
                               tweet.created_at, 
                               tweet.favorite_count, 
                               tweet.retweet_count, 
                               [h["text"] for h in tweet.entities["hashtags"]]])

        # Convertimos el array en un DataFrame y nombramos las columnas
    tuits_list = pd.DataFrame(tuits_list, columns=["Text", "User", "Created at", "Likes", "Retweets", "Hashtags"])

    likes = tuits_list[['Text','User','Hashtags','Likes']]
    max_likes = likes.iloc[1:750].max()
    max_likes['Text'] = max_likes['Text'].replace('#'," ").replace('\n',"")
    text = max_likes['Text']
    user = max_likes['User']
    likes = max_likes['Likes']
      
    return text, user, likes


















    

    