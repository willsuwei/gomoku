# -*- coding: utf-8 -*-

import random
import numpy as np
import os
import shutil
import time
from mpi4py import MPI
from collections import defaultdict, deque
from game_board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet
from policy_value_net_tensorlayer import PolicyValueNet as PolicyValueNet2
import sys
from datetime import datetime

CURRENT_MODEL_DIR = 'training/model_current/'
CURRENT_MODEL_PATH = CURRENT_MODEL_DIR + 'policy.model'
BEST_MODEL_DIR = 'training/model_best/'
BEST_MODEL_PATH = BEST_MODEL_DIR + 'policy.model'
EVALUATION_MODEL_DIR = 'training/model_evaluation/'
EVALUATION_MODEL_PATH = EVALUATION_MODEL_DIR + 'policy.model'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class TrainPipeline():
    def __init__(self, init_model=None, transfer_model=None):
        self.BOARD_WIDTH = 15
        self.BOARD_HEIGHT = 15
        self.N_IN_ROW = 5
        
        self.MAX_TRAINING_EPOCH = 100000
        
        self.EVALUATE_GAMES = 10

        self.MIN_NEW_GAME_COUNT = 20 # Play this number of games for each epoch

        self.AVERAGE_MOVES_PER_GAME = self.BOARD_WIDTH * self.BOARD_HEIGHT // 5 # define the average number of moves per game
        self.AVERAGE_DATA_PER_GAME = self.AVERAGE_MOVES_PER_GAME * 8 # define the average number of moves per game
        self.BUFFER_SIZE = 1000 * self.AVERAGE_DATA_PER_GAME
        self.BATCH_SIZE = 480 # 480 is the max to avoid out of memory

        self.resnet_block = 19  # num of block structures in resnet
        # params of the board and the game
        self.board = Board(width=self.BOARD_WIDTH,
                           height=self.BOARD_HEIGHT,
                           n_in_row=self.N_IN_ROW)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 1e-3
        self.N_PLAYOUTS = 1 + self.BOARD_WIDTH * self.BOARD_HEIGHT * 2  # num of simulations for each move. Minimum 2
        self.c_puct = 5

        self.best_win_ratio = 0.5
        
        self.data_buffer = deque(maxlen=self.BUFFER_SIZE)
            
        self.CUDA_NODES = 3
        self.cuda = rank < self.CUDA_NODES
        
        # cuda = True
        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, init_model=init_model, cuda=self.cuda)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # start training from a pre-trained policy-value net
            self.policy_value_net = PolicyValueNet(self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, transfer_model=transfer_model, cuda=self.cuda)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, cuda=self.cuda)

    def get_equivalent_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.BOARD_HEIGHT, self.BOARD_WIDTH)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def get_play_data(self, player1, player2, start_player = 0):
        winner, play_data = self.game.start_training_play(
            player1=player1, 
            player2=player2,
            start_player=start_player,
            rank=rank)
        
        play_data = list(play_data)[:]
        play_data = self.get_equivalent_data(play_data)
        return winner, play_data

    def policy_update(self, print_out):
        tmp_buffer = np.array(self.data_buffer)
        np.random.shuffle(tmp_buffer)
        steps = len(tmp_buffer)//self.BATCH_SIZE
        if print_out:
            print('tmp buffer: {}/{}, steps: {}'.format(len(tmp_buffer), self.BUFFER_SIZE, steps))
        for i in range(steps):
            mini_batch = tmp_buffer[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            old_probs, old_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)

            loss, entropy = self.policy_value_net.train_step(state_batch,
                                                             mcts_probs_batch,
                                                             winner_batch,
                                                             self.learn_rate)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
            )

            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))

            if print_out and (steps < 10 or (i % 20 == 0)):
                print('Rank {}: '.format(rank),
                    'batch:{}/{}, length:{}. kl:{:.5f}, loss:{}, entropy:{}, explained_var_old:{:.3f}, explained_var_new:{:.3f}'.format(
                    i, steps, len(mini_batch), kl,loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy

    def policy_evaluate(self):
        mcts_player = MCTSPlayer(
            self.policy_value_net,
            model=CURRENT_MODEL_PATH,
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.N_PLAYOUTS,
            is_selfplay=False)
            
        best_mcts_player = MCTSPlayer(
            self.policy_value_net,
            model=BEST_MODEL_PATH,
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.N_PLAYOUTS,
            is_selfplay=False)
    
        evaluation_mcts_player = MCTSPlayer(
            self.policy_value_net,
            model=EVALUATION_MODEL_PATH,
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.N_PLAYOUTS,
            is_selfplay=False)

        mcts_pure_player = MCTS_Pure(c_puct=5, n_playout=self.N_PLAYOUTS)

        win_cnt = defaultdict(int)
        for i in range(self.EVALUATE_GAMES):
            print('Rank {}: '.format(rank), 'Evaluating... Game:{}'.format(i))

            winner, _ = self.get_play_data(mcts_player, evaluation_mcts_player, start_player = i%2)

            win_cnt[winner] += 1
            win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / self.EVALUATE_GAMES # win for 1，tie for 0.5
            lost_ratio = 1.0 * (win_cnt[2] + 0.5 * win_cnt[-1]) / self.EVALUATE_GAMES # lost for 1，tie for 0.5

            print('Rank {}: '.format(rank), 'Win: {},  Lose: {},  Tie:{}, Win Ratio:{}, Lost Ratio:{}'.format(win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio, lost_ratio))
            
            # if (lost_ratio >= 0.5):
            #     print('Rank {}: '.format(rank), 'lost ratio reached 0.5')
            #     break

            # if (win_ratio > 0.5):
            #     print('Rank {}: '.format(rank), 'Win ratio over 0.5')
            #     break
        
        return win_ratio, win_cnt[1], win_cnt[2], win_cnt[-1]

    def model_trainer(self):
        start_time = time.time()
        epoch = 0
        game_count = 0
        while True:
            # Collecting games
            new_game_count = 0
            while (new_game_count < self.MIN_NEW_GAME_COUNT):
                time.sleep(3)
                dir_kifu_new = os.listdir('training/kifu_new')
                for file in dir_kifu_new:
                    if file == ".DS_Store":
                        continue
                    
                    try:
                        shutil.move('training/kifu_new/'+file, 'training/kifu_train/'+file)
                    except:
                        print('Rank {}: '.format(rank), '{} is being written now...'.format(file))

                dir_kifu_train = os.listdir('training/kifu_train')
                for file in dir_kifu_train:
                    if file == ".DS_Store":
                        continue
                    
                    try:
                        data = np.load('training/kifu_train/'+file, allow_pickle=True)
                        self.data_buffer.extend(data.tolist())
                        shutil.move('training/kifu_train/'+file, 'training/kifu_old/'+file)
                        new_game_count += 1
                    except:
                        print('Rank {}: '.format(rank), '{} is being written now...'.format(file))
                print("game count: {}. new game count: {}/{}".format(game_count, new_game_count, self.MIN_NEW_GAME_COUNT))
            game_count += new_game_count
            
            # Training
            epoch += 1
            print('Rank {}: '.format(rank), 'Training... Epoch: {}. Total games: {}. Data buffer length: {}. Now time: {}. {}'.format(epoch, game_count, len(self.data_buffer), (time.time() - start_time) / 3600, datetime.now()))
            loss, entropy = self.policy_update(print_out=True)
            print('Rank {}: '.format(rank), 'Loss: {} Entropy: {}'.format(loss, entropy))

            # Save
            while True:
                try:
                    self.policy_value_net.save_model(CURRENT_MODEL_PATH)
                    self.policy_value_net.save_model(CURRENT_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')
                    break
                except:
                    print('Rank {}: '.format(rank), "Model saving error. Try again in a moment...")
                    time.sleep(3)

            # Evaluate
            win_ratio, win, lose, tie = self.policy_evaluate()
            
            print('Rank {}: '.format(rank), "Evaluated!")
            
            # threshold = 0.5
            if win_ratio > self.best_win_ratio: # compare with self.best_win_ratio or threshold
                print("New best policy!!!!!!!!")
                self.best_win_ratio = win_ratio
                while True:
                    try:
                        self.policy_value_net.save_model(BEST_MODEL_PATH)
                        self.policy_value_net.save_model(BEST_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')
                        
                        f = open(BEST_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + "/info.txt", "w")
                        f.write("Win ratio: {}\n".format(win_ratio))
                        f.write("Win: {}\n".format(win))
                        f.write("Lose: {}\n".format(lose))
                        f.write("Tie: {}\n".format(tie))
                        f.close()

                        break
                    except:
                        print('Rank {}: '.format(rank), "Model saving error. Try again in a moment...")
                        time.sleep(3)

    def kifu_generator(self):
        mcts_player = MCTSPlayer(
            self.policy_value_net,
            model=CURRENT_MODEL_PATH,
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.N_PLAYOUTS,
            is_selfplay=True)

        while True:
            while True:
                try:
                    self.policy_value_net.restore_model(CURRENT_MODEL_PATH)
                    # print('Rank {}: '.format(rank), 'Load model "{}" successfully'.format(CURRENT_MODEL_PATH))
                    break
                except:
                    print('Rank {}: '.format(rank), 'Cannot load model. Will try again in a moment...')
                    time.sleep(3)

            winner, play_data = self.get_play_data(mcts_player, mcts_player)
            print('Rank {}: '.format(rank), 'Game finished. Winner is: {}. Moves: {}'.format(winner, len(play_data) // 8))
            np.save('training/kifu_new/rank_'+str(rank)+'_date_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.npy', np.array(play_data))

    def run(self):
        if (rank == 0):
            if not os.path.exists('training'):
                os.makedirs('training')
            if not os.path.exists('training/model_current'):
                os.makedirs('training/model_current')
            if not os.path.exists('training/model_best'):
                os.makedirs('training/model_best')
            if not os.path.exists('training/model_evaluation'):
                os.makedirs('training/model_evaluation')
            if not os.path.exists('training/kifu_new'):
                os.makedirs('training/kifu_new')
            if not os.path.exists('training/kifu_train'):
                os.makedirs('training/kifu_train')
            if not os.path.exists('training/kifu_old'):
                os.makedirs('training/kifu_old')

            dir_kifu_new = os.listdir('training/kifu_old')
            for file in dir_kifu_new:
                if file == ".DS_Store":
                    continue
                
                try:
                    shutil.move('training/kifu_old/'+file, 'training/kifu_new/'+file)
                except:
                    print('Rank {}: '.format(rank), '{} is being written now...'.format(file))

            self.policy_value_net.save_model(CURRENT_MODEL_PATH)
            self.policy_value_net.save_model(CURRENT_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')
            
            if (not os.path.exists(BEST_MODEL_PATH + '.index')):
                self.policy_value_net.save_model(BEST_MODEL_PATH)
                self.policy_value_net.save_model(BEST_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')

            if (not os.path.exists(EVALUATION_MODEL_PATH + '.index')):
                self.policy_value_net.save_model(EVALUATION_MODEL_PATH)
                self.policy_value_net.save_model(EVALUATION_MODEL_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')

        comm.Barrier()

        if (rank == 0 and ("--no_trainer" not in sys.argv)):
            self.model_trainer()
        else:
            self.kifu_generator()

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model=CURRENT_MODEL_PATH)
    # training_pipeline = TrainPipeline(transfer_model='training/policy.model')
    training_pipeline.run()
