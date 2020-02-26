import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    num_trials = 2
    arg_db_name = 'leaves_fossils'
    arg_net = 'resnet50_leaves'
    arg_train_mode = 'hard'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '1',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.2',
            '--batch_size','40',
            '--caffe_iter_size', '10',
            '--logging_threshold', '10',
            '--train_iters', '10000',
            '--test_interval','10',
            '--learning_rate', lr,
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_debug_threshold_3_resize_fixed_299_' + str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
        ]


        fast_fgvr_semi_train.main(args)
