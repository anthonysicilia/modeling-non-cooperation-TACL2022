import json
import pickle
from unicodedata import category
import guesswhat
import generic
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size' : 16})

from collections import defaultdict

OURS = False

if __name__ == '__main__':

    # # our games
    #######################################
    if OURS:
        games = os.listdir('saved_games')
    #######################################
    # their games
    #######################################
    else:
        games = []
        files = ['src/guesswhat.train.jsonl',
            'src/guesswhat.valid.jsonl',
            'src/guesswhat.test.jsonl']
        for file in files:
            for obj in open(file, 'r').readlines():
                games.append(json.loads(obj))
    #######################################

    images = defaultdict(int)
    objects = defaultdict(int)
    ocats = defaultdict(int)
    qwords = defaultdict(int)
    awords = defaultdict(int)
    questions = defaultdict(int)
    num_questions = []
    spam = 0
    success = 0
    num_answers = 0

    for i, g in enumerate(games):

        if OURS:
            g = f'saved_games/{g}'
            g = pickle.load(open(g, 'rb'))
        
            if g[0]['success']:
                success += 1
            game = g[0]['game']

            num_questions.append(len(game.questions))
            if all([a == 'No' for a in game.answers]) \
            or all([a == 'Yes' for a in game.answers]) \
            or all(a == 'N/A' for a in game.answers):
                spam += 1
            images[game.image.url.split('/')[-1]] += 1
            objects[game.object.id] += 1
            ocats[game.object.category] += 1
            for q in game.questions:
                questions[q.lower()] += 1
                for w in q.strip('?').split():
                    qwords[w.lower()] += 1
            for a in game.answers:
                awords[a] += 1
                num_answers += 1
        else:
            if g['status'] == 'success':
                success += 1

            gquestions = [x['question'] for x in g['qas']]
            ganswers = [x['answer'] for x in g['qas']]
            num_questions.append(len(gquestions))
            if all([a == 'No' for a in ganswers]) \
            or all([a == 'Yes' for a in ganswers]) \
            or all(a == 'N/A' for a in ganswers):
                spam += 1
            images[g['image']['file_name']] += 1
            objects[g['object_id']] += 1
            cat = None
            for obj in g['objects']:
                if obj['id'] == g['object_id']:
                    cat = obj['category']
                    break
            if cat is None:
                raise Exception('Cat is none.')
            ocats[cat] += 1
            for q in gquestions:
                questions[q.lower()] += 1
                for w in q.strip('?').split():
                    qwords[w.lower()] += 1
            for a in ganswers:
                awords[a] += 1
                num_answers += 1

    pwrap = lambda x: f'{x * 100:.2f}%'
    
    print('Num unique images:', len(images))
    print('Num unique objects:', len(objects))
    print('Num unique object cats:', len(ocats))
    print('Percent Yes:', pwrap(awords['Yes'] / num_answers))
    print('Percent No:', pwrap(awords['No'] / num_answers))
    print('Percent N/A:', pwrap(awords['N/A'] / num_answers))
    print('Percent Success:', pwrap(success / len(games)))
    print('Percent Spam:', pwrap(spam / len(games)))
    print('Num words (+1):', len(qwords) + len(awords))
    qwords = [k for k,v in qwords.items() if v >= 3]
    awords = [k for k,v in awords.items() if v >= 3]
    print('Num words (+3):', len(qwords) + len(awords))
    print('Total questions:', sum(num_questions))
    print('Unique questions:', len(questions))
    print('Avg num questions', f'{sum(num_questions) / len(games):.2f}')
    print('Num games (dialogues):', len(games))

    fig, ax = plt.subplots(1, 1, figsize=(17, 4.1))
    x = [k for k,_ in sorted(ocats.items(), key=lambda t: -t[1])]
    y = [v for _,v in sorted(ocats.items(), key=lambda t: -t[1])]
    ax.bar(x, y, color='b')
    ax.set_yscale('log')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count (log scale)')
    ax.set_title('Object Distribution')
    if OURS:
        ax.set_yticks([10,100,1000])
    else:
        ax.set_yticks([10,100,1000, 10_000])
    ax.set_xticklabels(x, rotation=90)
    plt.tight_layout()
    plt.savefig('odist' if OURS else 'odist-original')

    fig, ax = plt.subplots(1, 1, figsize=(5,4.1))
    if OURS:
        print('Max #qs', max(num_questions))
    else:
        print('Outliers qs', sum([num > 27 for num in num_questions]))
        num_questions = [num for num in num_questions if num <= 27]
    ax.hist(num_questions, color='g', bins=14)
    if OURS:
        ax.axvline(4.99, c='b', ls='--', lw=4, label='Average')
    else:
        ax.axvline(5.11, c='y', ls='--', lw=4, label='Average')
    ax.legend()
    ax.set_xlabel('Number')
    ax.set_ylabel('Count')
    ax.set_title('Question Distribution')
    plt.tight_layout()
    plt.savefig('qdist' if OURS else 'qdist-original')


