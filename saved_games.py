import pickle as pkl
from src.guesswhat.data_provider.guesswhat_dataset import dump_samples_into_dataset
from src.guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    mypath = './saved_games/'
    fname = 'train_new' # 'deceptive'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    games = []
    tokenizer = GWTokenizer('./data/dict.json')
    for file in onlyfiles:
        with open(join(mypath, file), 'rb') as f:
            games.append(pkl.load(f)[0])
    dump_samples_into_dataset(games, f'./data/guesswhat.{fname}.jsonl.gz', tokenizer)