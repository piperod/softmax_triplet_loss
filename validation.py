import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':


    #gpu_number = 3
  
    num_trials = 1
    username = 'irodri15_validation'
    arg_db_name = 'validation_pnas'
    arg_net = 'resnet50_leaves_pretrained'
    arg_train_mode = 'hard'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '6',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.1',
            '--batch_size','45',
            '--caffe_iter_size', '10',
            '--logging_threshold', '5',
            '--train_iters', '15000',
            '--test_interval','5',
            '--learning_rate', lr,
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username', username,

            '--checkpoint_suffix', '_validation_5050_pretrained_2_'+ str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
            ]
        

        fast_fgvr_semi_train.main(args) 
            
            