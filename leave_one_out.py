import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':


    gpu_number = 3
    for m in [0.05,0.2,0.5]:
        l=0
        num_trials = 1
        arg_db_name = 'leaves_fossils_out'
        arg_net = 'resnet50_leaves'
        arg_train_mode = 'hard_anchor'
        lr = '0.01'
        for idx in range(num_trials):
            args = [
            '--gpu', '0',
            '--db_name', arg_db_name,
            '--label_out',str(l),
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', str(m),
            '--batch_size','40',
            '--caffe_iter_size', '5',
            '--logging_threshold', '5',
            '--train_iters', '2000',
            '--test_interval','5',
            '--learning_rate', lr,
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_debug_leave_out_fossils_' +str(l)+'_'+ str(idx)+'_'+str(m)

            # These flags are used for different experiments
            # '--frame_size','299',
             ]


            fast_fgvr_semi_train.main(args) 