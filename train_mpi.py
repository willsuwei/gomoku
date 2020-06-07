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
import sys
from datetime import datetime

# import sys
# sys.stdout.flush()
# or just
# mpiexec -np 43 python -u train_mpi.py

# MPI setting
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()  # processing ID


class TrainPipeline():
    def __init__(self, init_model=None, transfer_model=None):
        self.BOARD_WIDTH = 15
        self.BOARD_HEIGHT = 15
        self.N_IN_ROW = 5
        
        self.MAX_TRAINING_EPOCH = 100000
        
        self.TRAINING_GAME_COUNT = 16 # Play this number of games for each epoch

        self.game_count = 0  # count total game have played

        self.resnet_block = 19  # num of block structures in resnet
        # params of the board and the game
        self.board = Board(width=self.BOARD_WIDTH,
                           height=self.BOARD_HEIGHT,
                           n_in_row=self.N_IN_ROW)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 1e-3
        self.N_PLAYOUTS = 1 + 400  # num of simulations for each move. Minimum 2
        self.c_puct = 5

        self.AVERAGE_MOVES_PER_GAME = 30 # define the average number of moves per game
        self.BATCH_SIZE = 16 * self.AVERAGE_MOVES_PER_GAME # 512 default. 480 for 15x15
        self.BUFFER_SIZE = 4000 * self.AVERAGE_MOVES_PER_GAME
            # memory size, should be larger with bigger board
            # in paper it can stores 500,000 games, here 500000 with 11x11 board can store only around 2000 games. (25 steps per game)
        self.data_buffer = deque(maxlen=self.BUFFER_SIZE)
            
        self.CUDA_NODES = 1
        self.cuda = rank < self.CUDA_NODES

        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, init_model=init_model, cuda=self.cuda)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # start training from a pre-trained policy-value net
            self.policy_value_net = PolicyValueNet(
                self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, transfer_model=transfer_model, cuda=self.cuda)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                self.BOARD_WIDTH, self.BOARD_HEIGHT, block=self.resnet_block, cuda=self.cuda)

        self.mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                      action_fc=self.policy_value_net.action_fc_test,
                                      evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                      c_puct=self.c_puct,
                                      n_playout=self.N_PLAYOUTS,
                                      is_selfplay=True)


    def policy_update(self, print_out):
        #play_data: [(state, mcts_prob, winner_z), ..., ...]
        # train an epoch

        tmp_buffer = np.array(self.data_buffer)
        np.random.shuffle(tmp_buffer)
        steps = len(tmp_buffer)//self.BATCH_SIZE
        if print_out:
            print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer), steps))
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
                # print some information, not too much
                print('rank {}:'.format(rank), 
                      'batch: {}/{}, length: {} '
                      'kl:{:.5f}, '
                      'loss:{}, '
                      'entropy:{}, '
                      'explained_var_old:{:.3f}, '
                      'explained_var_new:{:.3f}'.format(i,
                                                        steps,
                                                        len(mini_batch),
                                                        kl,
                                                        loss,
                                                        entropy,
                                                        explained_var_old,
                                                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=10, num=0, model1='tmp/current_policy.model', model2='model/best_policy.model'):
        # mcts_player = self.mcts_player
        mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                        action_fc=self.policy_value_net.action_fc_test,
                                        evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout,
                                        is_selfplay=False)

        # test_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        test_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                        action_fc=self.policy_value_net.action_fc_test,
                                        evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout,
                                        is_selfplay=False)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            print('rank {}: '.format(rank), 
                'Evaluating... epoch:{}, game:{}'.format(num, i)
            )

            winner, _ = self.game.start_training_play(
                player1=mcts_player, 
                player2=test_player,
                start_player=i%2,
                rank=rank,
                isEvaluate=True,
                model1=model1,
                model2=model2,
                policy_value_net=self.policy_value_net,
                show_play=False,
                show_probs_value=False,
                show_play_UI=True,
                calculate_performance=False)

            win_cnt[winner] += 1
            win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games # win for 1，tie for 0.5
            lost_ratio = 1.0*(win_cnt[2] + 0.5*win_cnt[-1]) / n_games # lost for 1，tie for 0.5

            print('rank {}: '.format(rank), 
                'Win: {},  Lose: {},  Tie:{}, Win Ratio:{}, Lost Ratio:{}'.format(
                win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio, lost_ratio)
            )
            
            if (lost_ratio >= 0.5):
                print('rank {}: '.format(rank), 
                    'lost ration reached 0.5'
                )
                break
        
        return win_ratio, win_cnt[1], win_cnt[2], win_cnt[-1]

    def mymovefile(self, srcfile, dstfile):
        '''
        move file to another dirs
        '''
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            shutil.move(srcfile, dstfile)

    def mycpfile(self, srcfile, dstfile):
        '''
        copy file to another dirs
        '''
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            shutil.copy(srcfile, dstfile)

    def get_equivalent_data(self, play_data):
        '''
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        '''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                # rotate counterclockwise 90*i
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.BOARD_HEIGHT, self.BOARD_WIDTH)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            # print('rank: {}, n_games: {}, start playing...'.format(
            #     rank, i))
            
            winner, play_data = self.game.start_training_play(
                self.mcts_player,
                self.mcts_player,
                rank=rank,
                show_play=False,
                show_probs_value=False,
                show_play_UI=True,
                calculate_performance=True)
            
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equivalent_data(play_data)
            self.data_buffer_tmp.extend(play_data)
            
            # print('rank: {}, n_games: {}, data length: {}, winner: {}'.format(
            #     rank, i, self.episode_len, winner))

    def play(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass

    def run(self):
        if (rank == 0):
            # Create folders
            if not os.path.exists('training'):
                os.makedirs('training')
            if not os.path.exists('training/model_current'):
                os.makedirs('training/model_current')
            if not os.path.exists('training/model_best'):
                os.makedirs('training/model_best')
            if not os.path.exists('training/kifu_new'):
                os.makedirs('training/kifu_new')
            if not os.path.exists('training/kifu_train'):
                os.makedirs('training/kifu_train')
            if not os.path.exists('training/kifu_old'):
                os.makedirs('training/kifu_old')

            # Save current model    
            self.policy_value_net.save_model('training/model_current/policy.model')
            self.policy_value_net.save_model('training/model_current/' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')
            
            if (not os.path.exists('training/model_best/policy.model.index')):
                self.policy_value_net.save_model('training/model_best/policy.model')
                self.policy_value_net.save_model('training/model_best/' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/policy.model')

        comm.Barrier()
        
        for num in range(self.MAX_TRAINING_EPOCH):
            self.play()
            self.train()
            self.evaluate()

            # print('rank {}: '.format(rank), 
            #     'Now is {}'.format(datetime.now())
            # )

            continue

            if rank == 0:
                before = time.time()
                
                while True:
                    dir_kifu_new = os.listdir('play_history/kifu_new')
                    for file in dir_kifu_new:
                        if file == ".DS_Store":
                            continue
                        
                        try:
                            self.mymovefile('play_history/kifu_new/'+file,
                                            'play_history/kifu_train/'+file)
                        except:
                            print('rank {}: '.format(rank), 
                                '{} is being written now...'.format(file)
                            )

                    dir_kifu_train = os.listdir('play_history/kifu_train')
                    for file in dir_kifu_train:
                        if file == ".DS_Store":
                            continue
                        
                        try:
                            data = np.load('play_history/kifu_train/'+file, allow_pickle=True)
                            self.data_buffer.extend(data.tolist())
                            self.mymovefile('play_history/kifu_train/'+file, 'play_history/kifu_old/'+file)
                            os.remove('play_history/kifu_old/'+file) # don't save history
                            self.game_count += 1
                        except:
                            print('rank {}: '.format(rank), 
                                '{} is being written now...'.format(file)
                            )

                    print('rank {}: '.format(rank), 
                        'Train epoch: {}. Total games: {}'.format(num, self.game_count)
                    )

                    if len(self.data_buffer) < self.batch_size * 5:
                        print('rank {}: '.format(rank), 
                            'Buffer size: {}. Minimum: {}'.format(len(self.data_buffer), self.batch_size * 5)
                        )
                        time.sleep(10)
                        continue
                    else:
                        break
                
                # Training
                print('rank {}: '.format(rank),
                    'Training... Epoch: . Data buffer length: {}. Now time: {}'.format(len(self.data_buffer), (time.time()-start_time)/3600)
                )
                loss, entropy = self.policy_update(print_out=True)
                print('rank {}: '.format(rank),
                    'Loss: {} Entropy: {}'.format(loss, entropy)
                )

                # Save
                while True:
                    try:
                        self.policy_value_net.save_model('tmp/current_policy.model')
                        self.policy_value_net.save_model('tmp/' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/current_policy.model')
                        break
                    except:
                        print("!" * 100, "Model saving error. Try again in 3 seconds")
                        time.sleep(3)

                # Evaluate
                evaluate_start_time = time.time()
                
                threshold = 0.5
                win_ratio, win, lose, tie = self.policy_evaluate(
                    n_games=10, 
                    num=num, 
                    model1='tmp/current_policy.model',
                    model2='model/best_policy.model'
                    # model2='archieve/15x15/best_policy.model'
                )
                
                print("Evaluated!!!!!!!!", "-"*400)
                evaluate_time += time.time() - evaluate_start_time
                if win_ratio > self.best_win_ratio: # compare with self.best_win_ratio or threshold
                    print("New best policy!!!!!!!!", '!'*400)
                    self.best_win_ratio = win_ratio
                    
                    while True:
                        try:
                            self.policy_value_net.save_model('model/best_policy.model')
                            self.policy_value_net.save_model('model/' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '/best_policy.model')
                            break
                        except:
                            print("!" * 100, "Model saving error. Try again in 3 seconds")
                            time.sleep(3)
                    
                    f = open("model/win_ratio.txt", "w")
                    f.write(str(self.best_win_ratio) + '\n')
                    f.write(str(win) + '\n')
                    f.write(str(lose) + '\n')
                    f.write(str(tie) + '\n')
                    f.close()
                    
                # Keep 10 mins training interval
                # after = time.time()
                # if after - before < 60 * 10:
                #     print('rank {}: '.format(rank), 
                #         'Now is {}. Sleep for {} seconds'.format(datetime.now(), 60*10-after+before)
                #     )
                    # time.sleep(60*10-after+before)

            else:
                #　self-play to collect data
                if os.path.exists('model/best_policy.model.index'):
                    while True:
                        try:
                            retore_model_start_time = time.time()
                            self.policy_value_net.restore_model('model/best_policy.model')
                            retore_model_time += time.time()-retore_model_start_time
                            # print("rank", rank, ":", 'best model loaded...')
                            break
                        except:
                            # the model is under written
                            print("rank", rank, ":", 'cannot load model...')
                            time.sleep(3)
                else:
                    while True:
                        try:
                            retore_model_start_time = time.time()
                            self.policy_value_net.restore_model('tmp/current_policy.model')
                            retore_model_time += time.time()-retore_model_start_time
                            # print("rank", rank, ":", 'current model loaded...')
                            break
                        except:
                            # the model is under written
                            print("rank", rank, ":", 'cannot load model...')
                            time.sleep(3)

                # tmp buffer to collect self-play data
                self.data_buffer_tmp = []

                # collect self-play data
                collect_data_start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                collect_data_time += time.time()-collect_data_start_time

                # save data to file
                save_data_satrt_time = time.time()
                np.save('play_history/kifu_new/rank_'+str(rank)+'_date_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.npy', np.array(self.data_buffer_tmp))
                save_data_time += time.time()-save_data_satrt_time

                # print('rank {}: '.format(rank), 'now time : {}'.format((time.time() - start_time) / 3600))
                # print('rank {}: '.format(rank), 
                #     'rank : {}, restore model time : {}, collect_data_time : {}, save_data_time : {}'
                #     .format(rank, retore_model_time/3600, collect_data_time/3600, save_data_time/3600)
                # )

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='training/model_current/policy.model', transfer_model=None)
    # training_pipeline = TrainPipeline(init_model=None, transfer_model='model/best_policy.model')
    # training_pipeline = TrainPipeline()
    training_pipeline.run()
