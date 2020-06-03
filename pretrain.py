import os
import fast_fgvr_semi_train
import neptune

neptune.init('piperod/Leaves')

#neptune.set_project('piperod/Leaves')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    print('here')
    num_trials = 1
    arg_db_name = 'pretrain'
    arg_net = 'resnet50_leaves'
    arg_train_mode = 'vanilla'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '4',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.5',
            '--batch_size','45',
            '--caffe_iter_size', '5',
            '--logging_threshold', '5',
            '--train_iters', '10000',
            '--test_interval','5',
            '--learning_rate', lr,
            '--Triplet_K','5',
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_ptretraining_' + str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
        ]


        fast_fgvr_semi_train.main(args,neptune)