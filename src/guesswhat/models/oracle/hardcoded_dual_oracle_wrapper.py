
from generic.tf_utils.evaluator import Evaluator
import random

class DualOracleWrapper(object):

    NUM_STRATS = 1

    def __init__(self, coop_oracle, tokenizer):

        self.coop_oracle = coop_oracle
        self.evaluator = None
        self.tokenizer = tokenizer
        self.refresh_game(None)
    
    def spam(self, game_data):
        return [self.spam_id for _ in 
            range(len(game_data['question']))]

    def refresh_game(self, id_=None):
        self.current_id = id_
        self.strats = [self.spam]
        self.spam_id = 2 #random.randint(0, 2)
        self.current_strategy = random.randint(0, self.NUM_STRATS-1)
        
    def initialize(self, sess):
        self.coop_evaluator = Evaluator(self.coop_oracle.get_sources(sess), self.coop_oracle.scope_name)

    def get_deceptive_answers(self, game_data):
        # logging of questions, answers, coop_answers, etc.
        return self.strats[self.current_strategy](game_data)

    def answer_question(self, sess, question, seq_length, game_data):

        # if self.current_id != game_data['raw']['id']:
        #     self.refresh_game(game_data['id'])

        game_data["question"] = question
        game_data["seq_length"] = seq_length

        # convert dico name to fit oracle constraint
        game_data["category"] = game_data.get("targets_category", None)
        game_data["spatial"] = game_data.get("targets_spatial", None)

        # sample
        answers_indices = []
        deceptive_answers = self.get_deceptive_answers(game_data)
        cooperative_answers = self.coop_evaluator.execute(sess, output=self.coop_oracle.best_pred, batch=game_data)
        for idx, is_deceptive in enumerate(game_data['deceptive']):
            if is_deceptive:
                answers_indices.append(deceptive_answers[idx])
            else:
                answers_indices.append(cooperative_answers[idx])

        # if game_data['deceptive']:
        #     answers_indices = self.get_deceptive_answers(game_data)
        # else:
        #     answers_indices = self.coop_evaluator.execute(sess, output=self.coop_oracle.best_pred, batch=game_data)

        # Decode the answers token  ['<yes>', '<no>', '<n/a>'] WARNING magic order... TODO move this order into tokenizer
        answer_dico = [self.tokenizer.yes_token, self.tokenizer.no_token, self.tokenizer.non_applicable_token]
        answers = [answer_dico[a] for a in answers_indices]  # turn indices into tokenizer_id

        return answers






class OracleUserWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def initialize(self, sess):
        pass


    def answer_question(self, sess, question, **_):

        # Discard question if it contains the stop dialogue token
        if self.tokenizer.stop_dialogue in question[0]:
            return [self.tokenizer.non_applicable_token]

        print()
        print("Q :", self.tokenizer.decode(question[0]))

        while True:
            answer = input('A (Yes,No,N/A): ').lower()
            if answer == "y" or answer == "yes":
                token = self.tokenizer.yes_token
                break

            elif answer == "n" or answer == "no":
                token = self.tokenizer.no_token
                break

            elif answer == "na" or answer == "n/a" or answer == "not applicable":
                token = self.tokenizer.non_applicable_token
                break

            else:
                print("Invalid answer...")

        return [token]