import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    num_trials = 3
    arg_db_name = 'pnas'
    arg_net = 'resnet50'
    arg_train_mode = 'cntr'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '4',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.2',
            '--batch_size','40',
            '--caffe_iter_size', '1',
            '--logging_threshold', '100',
            '--train_iters', '40000',
            '--test_interval','100',
            '--learning_rate', lr,
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_final_lm1_aug_img_fixed_299_' + str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
        ]


        fast_fgvr_semi_train.main(args)
