import pandas as pd

# 0XSlGDXj7e7vF2gErYmXUr8nG
# 010Jn7QPg8NL1s3w21nKx3diDnn4jR6GZmzbLl04HXJGKXkOPC
# AAAAAAAAAAAAAAAAAAAAAHKReQEAAAAAxP1H7LJ6QrzBfvb3AJu%2BTVSxTBE%3D9RoBwsmgm41GFJlSuobf4MdG96opusVY4vHGMx4hlRt2SJQJHp

# https://www.kaggle.com/code/erdeq1024/bitcoin-price-analysis-by-tweets
# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

def main():
    df = pd.read_csv('./bitcoin_tweets.csv', header = 0, chunksize = 1024*1024).read()
    print(df.iloc[-1].date)

if __name__ == '__main__':
    main()