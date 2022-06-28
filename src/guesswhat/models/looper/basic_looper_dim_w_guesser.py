from tqdm import tqdm
import numpy as np


from guesswhat.models.looper.tools import clear_after_stop_dialogue, list_to_padded_tokens


def print_dialog(dialogue, tokenizer):
    start  = 1
    for k, word in enumerate(dialogue):
        if word == tokenizer.yes_token or \
                word == tokenizer.no_token or \
                word == tokenizer.non_applicable_token:
            q = tokenizer.decode(dialogue[start:k - 1])
            a = tokenizer.decode([dialogue[k]])
            with open('train-dialog-output.txt', 'a') as out:
                out.write('Q: ' + ' '.join(q) + '\n')
                out.write('A: ' + ' '.join(a) + '\n')
            start = k + 1


class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, d_guesser_wrapper, tokenizer, batch_size,
            train_on_original_task=False, train_only_on_task=False):
        self.storage = []

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']
        self.max_depth = config['loop']['max_depth']
        self.k_best = config['loop']['beam_k_best']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.d_guesser = d_guesser_wrapper
        self.qgen = qgen_wrapper
        self.train_on_original_task = train_on_original_task
        self.train_only_on_task = train_only_on_task

    def process(self, sess, iterator, mode, optimizer=dict(), store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)
        self.d_guesser.initialize(sess)


        self.storage = []
        score, found_score, total_elem = 0, 0, 0
        num_games = 0
        num_coop_games = 0
        coop_score = 0

        for game_data in tqdm(iterator):

            # initialize the dialogue
            full_dialogues = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_answers = full_dialogues

            prob_objects = []

            no_elem = len(game_data["raw"])
            total_elem += no_elem

            # Step 1: generate question/answer
            self.qgen.reset(batch_size=no_elem)
            try:
                self.oracle.refresh_game()
            except:
                pass
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                padded_questions, questions, seq_length = \
                    self.qgen.sample_next_question(sess, prev_answers, game_data=game_data, mode=mode)
                
                game_data['history'], game_data['hist_seq_length'] = \
                    list_to_padded_tokens(full_dialogues, self.tokenizer)
                
                # Step 1.2: Answer the question
                answers = self.oracle.answer_question(sess,
                                                      question=padded_questions,
                                                      seq_length=seq_length,
                                                      game_data=game_data)

                # Step 1.3: store the full dialogues
                for i in range(self.batch_size):
                    full_dialogues[i] = np.concatenate((full_dialogues[i], questions[i], [answers[i]]))

                # Step 1.4 set new input tokens
                prev_answers = [[a]for a in answers]

                # Step 1.5 Compute the probability of finding the object after each turn
                if store_games:
                    padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
                    _, softmax, _ = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
                    prob_objects.append(softmax)

                # Step 1.6 Check if all dialogues are stopped
                has_stop = True
                for d in full_dialogues:
                    has_stop &= self.tokenizer.stop_dialogue in d
                if has_stop:
                    break

            # Step 2 : clear question after <stop_dialogue>
            full_dialogues, _ = clear_after_stop_dialogue(full_dialogues, self.tokenizer)
            padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
            
            # Step 3 : Find the object
            found_object, _, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            found_score += np.sum(found_object)
            num_games += len(found_object)

            coop_games = found_object[(1-np.array(game_data['deceptive'])).astype(bool)]
            coop_score += np.sum(coop_games)
            num_coop_games += len(coop_games)

            # Step 3 : Find the object
            #found_object, _, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            found_deception, _, softmax = self.d_guesser.is_deceptive(sess, padded_dialogue, seq_length, game_data)
            score += np.sum(found_deception)


            # TODO fix this function, even though its not called, for metrics later
            if store_games:
                prob_objects = np.transpose(prob_objects, axes=[1,0,2])
                for i, (d, g, t, f, go, po) in enumerate(zip(full_dialogues, game_data["raw"], game_data["targets_index"], found_object, id_guess_objects, prob_objects)):
                    self.storage.append({"dialogue": d,
                                         "game": g,
                                         "object_id": g.objects[t].id,
                                         "success": f,
                                         "guess_object_id": g.objects[go].id,
                                         "prob_objects" : po} )

            if len(optimizer) > 0:
                if self.train_only_on_task:
                    final_reward = found_object + 0  # +1 if found otherwise 0
                else:
                    final_reward = found_deception + 0  # +1 if found otherwise 0
                    if self.train_on_original_task:
                        final_reward = final_reward + found_object
            for key in optimizer:
                if key != 'guesser':
                    optimizers = optimizer[key]
                    self.apply_policy_gradient(sess,
                                           final_reward=final_reward,
                                           padded_dialogue=padded_dialogue,
                                           seq_length=seq_length,
                                           game_data=game_data,
                                           optimizer=optimizers)
                elif key == 'guesser':
                    guesser = self.d_guesser.guesser
                    sess.run(optimizer[key], feed_dict = {
                        guesser.dialogues: padded_dialogue,
                        guesser.seq_length: seq_length,
                        guesser.obj_mask: game_data['obj_mask'],
                        guesser.obj_cats: game_data['obj_cats'],
                        guesser.obj_spats: game_data['obj_spats'],
                        guesser.deceptive: game_data['deceptive']
                    })
                else:
                    raise NotImplementedError()

            # with open('train-dialog-output.txt', 'a') as out:
            #     out.write('Deceptive: ' + str(game_data['deceptive'][0]) + '\n')
            #     out.write('Game Info: ' + str(game_data['debug'][0]) + '\n')
            # print_dialog(full_dialogues[0], self.tokenizer)
            # with open('train-dialog-output.txt', 'a') as out:
            #     out.write('Found Deception: ' + str(found_deception[0]) + '\n')
            #     out.write('End Game' + str(num_games) + '\n')

        score = 1.0 * score / num_games #iterator.n_examples
        found_score = 1.0 * found_score / num_games #iterator.n_examples
        coop_score = 1.0 * coop_score / num_coop_games

        return score, found_score, coop_score

    def get_storage(self):
        return self.storage

    def apply_policy_gradient(self, sess, final_reward, padded_dialogue, seq_length, game_data, optimizer):

        # Compute cumulative reward TODO: move into an external function
        cum_rewards = np.zeros_like(padded_dialogue, dtype=np.float32)
        for i, (end_of_dialogue, r) in enumerate(zip(seq_length, final_reward)):
            cum_rewards[i, :(end_of_dialogue - 1)] = r  # gamma = 1

        # Create answer mask to ignore the reward for yes/no tokens
        answer_mask = np.ones_like(padded_dialogue)  # quick and dirty mask -> TODO to improve
        answer_mask[padded_dialogue == self.tokenizer.yes_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.no_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.non_applicable_token] = 0

        # Create padding mask to ignore the reward after <stop_dialogue>
        padding_mask = np.ones_like(padded_dialogue)
        padding_mask[padded_dialogue == self.tokenizer.padding_token] = 0
        # for i in range(np.max(seq_length)): print(cum_rewards[0][i], answer_mask[0][i],self.tokenizer.decode([padded_dialogue[0][i]]))

        # Step 4.4: optim step
        qgen = self.qgen.qgen  # retrieve qgen from wrapper (dirty)

        sess.run(optimizer,
                 feed_dict={
                     qgen.images: game_data["images"],
                     qgen.dialogues: padded_dialogue,
                     qgen.seq_length: seq_length,
                     qgen.padding_mask: padding_mask,
                     qgen.answer_mask: answer_mask,
                     qgen.cum_rewards: cum_rewards,
                 })
