import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import neptune 
neptune.set_project('Leaves/leaves')

if __name__ == '__main__':


    gpu_number = 3
    #,,
    for l in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        #l=0
        num_trials = 1
        arg_db_name = 'leaves_fossils_out'
        arg_net = 'resnet50_leaves'
        arg_train_mode = 'hard_anchor_fossils'
        lr = '0.01'
        for idx in range(num_trials):
            args = [
            '--gpu', '1',
            '--db_name', arg_db_name,
            '--label_out',str(l),
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.4',
            '--batch_size','40',
            '--caffe_iter_size', '5',
            '--logging_threshold', '5',
            '--train_iters', '2500',
            '--test_interval','5',
            '--Triplet_K','4',
            '--learning_rate', lr,
            #'--training_mode_debug','True',
            '--aug_style', 'img',
            '--username','irodri15p1',

            '--checkpoint_suffix', '_leave_out_new_loss_test_neptune_' +str(l)+'_'+ str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
             ]
        

            fast_fgvr_semi_train.main(args,neptune) 
            