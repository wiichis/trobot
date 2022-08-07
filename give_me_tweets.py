import credentials
import tweepy


auth = tweepy.OAuthHandler(credentials.API_Key, credentials.API_Secret_Key)
auth.set_access_token(credentials.Acces_token, credentials.Acces_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

def get_tweets(hastag):
    id = None
    count = 0
    while count <= 1000:
        tweets = api.search_tweets(q=hastag, lang='es', tweet_mode='extended', max_id=id)
        for tweet in tweets:
            if tweet.full_text.startswith('RT'):
                count += 1
                continue
            f = open('./archivos/data_tweets/'+hastag+'.txt', 'a', encoding='utf-8')
            f.write(tweet.full_text + '\n')
            f.close
            count += 1
        id = tweet.id

    

    

    