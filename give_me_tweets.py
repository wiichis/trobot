from numpy import result_type
import pandas as pd
import credentials
import tweepy


auth = tweepy.OAuthHandler(credentials.API_Key, credentials.API_Secret_Key)
auth.set_access_token(credentials.Acces_token, credentials.Acces_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets(hastag):
    tuits_list = []

        # Usamos el Cursor dentro de un for
    for tweet in tweepy.Cursor(api.search_tweets,
                               q=hastag, #-filter:retweets
                               lang="es",
                               tweet_mode="extended",
                               result_type='recent').items(750): #Cantidad de Tuits por busqueda

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
    #print(max_likes["Text"])

    rts = tuits_list[['Text','User','Hashtags','Retweets']]
    max_rts = rts.iloc[1:750].max()


    return(max_likes["Text"], max_rts["Text"])




















    

    