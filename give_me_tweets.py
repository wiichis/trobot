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
                               result_type='mixed').items(50): #Cantidad de Tuits por busqueda

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
    likes = likes.astype({'Text':'string',likes:'int32'}).dtypes
    max_likes = likes.iloc[1:750].max()

    rts = tuits_list[['Text','User','Hashtags','Retweets']]
    max_rts = rts.iloc[1:750].max()


    return(max_likes, max_rts)

#en lugar de guardar el tuit en un CSV, solo selecciono el que tenga mas RTs el que tenga mas likes y lo devuelvo con return

















# def get_tweets(hastag):
#     id = None
#     count = 0
#     while count <= 750:
#         tweets = api.search_tweets(q=hastag, lang='es', tweet_mode='extended', max_id=id)
#         for tweet in tweets:
#             if tweet.full_text.startswith('RT'):
#                 count += 1
#                 continue
#             f = open('./archivos/data_tweets/'+hastag+'.txt', 'a', encoding='utf-8')
#             f.write(tweet.full_text + '\n')
#             f.close
#             count += 1
#         id = tweet.id

    

    

    