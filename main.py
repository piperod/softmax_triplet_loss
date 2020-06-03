import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    num_trials = 1
    arg_db_name = 'leaves_fossils'
    arg_net = 'resnet50_leaves'
    arg_train_mode = 'hard_anchor'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '6',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.5',
            '--batch_size','45',
            '--caffe_iter_size', '10',
            '--logging_threshold', '10',
            '--train_iters', '10000',
            '--test_interval','10',
            '--learning_rate', lr,
            '--Triplet_K','3',
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_anchor_max' + str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
        ]


        fast_fgvr_semi_train.main(args)
