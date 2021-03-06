import os
import time
import json
import numpy as np
import tensorflow as tf
from pydoc import locate
from utils import user_io
import constants as const
from utils import os_utils
from utils import log_utils
from ranking import center_loss
from ranking import triplet_semi
from ranking import triplet_hard
import utils.tb_utils as tb_utils
import utils.tf_utils as tf_utils
from utils.log_utils import classification_report_csv
from nets.conv_embed import ConvEmbed
from data_sampling.quick_tuple_loader import QuickTupleLoader
from data_sampling.triplet_tuple_loader import TripletTupleLoader
from config.base_config import BaseConfig
from sklearn.metrics import classification_report

def main(argv):
    
    cfg = BaseConfig().parse(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    img_generator_class = locate(cfg.db_tuple_loader)
    args = dict()
    args['db_path'] = cfg.db_path
    args['tuple_loader_queue_size'] = cfg.tuple_loader_queue_size
    args['preprocess_func'] = cfg.preprocess_func
    args['batch_size'] = cfg.batch_size
    args['shuffle'] = False
    args['img_size'] = const.max_frame_size
    args['gen_hot_vector'] = True

    args['batch_size'] = cfg.batch_size
    args['csv_file'] = cfg.test_csv_file
    test_iter = img_generator_class(args)

    
    test_imgs, test_lbls = test_iter.imgs_and_lbls()
    cfg = BaseConfig().parse(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    model_dir = cfg.checkpoint_dir
    print(model_dir)
    log_file = os.path.join(cfg.checkpoint_dir, cfg.log_filename + '_test.txt')
    logger = log_utils.create_logger(log_file)

    with tf.Graph().as_default():
        #meta_file = os.path.join(model_dir,'model.ckptbest.meta')
        #saver = tf.train.import_meta_graph(meta_file)
        #ckpt_file = os.path.join(model_dir,'model.ckptbest')
        #saver.restore(sess,ckpt_file)
        
        #print('Model Path {}'.format(ckpt_file))
        #load_model_msg = model.load_model(model_dir, ckpt_file, sess, saver, load_logits=True)
        #logger.info(load_model_msg)
        #graph = tf.get_default_graph()
        #print(graph.get_operations())

        
        test_dataset = QuickTupleLoader(test_imgs, test_lbls,cfg, is_training=False,repeat=False).dataset
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, test_dataset.output_types, test_dataset.output_shapes)
        images_ph, lbls_ph = iterator.get_next()

        network_class = locate(cfg.network_name)
        model = network_class(cfg,images_ph=images_ph, lbls_ph=lbls_ph)
        validation_iterator = test_dataset.make_initializable_iterator()
        
        
        sess = tf.InteractiveSession()
        validation_handle = sess.run(validation_iterator.string_handle())
        ckpt_file = tf.train.latest_checkpoint(model_dir)
        print(ckpt_file)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        load_model_msg = model.load_model(model_dir, ckpt_file, sess, saver, load_logits=True )
        print(load_model_msg)

        ckpt_file = os.path.join(model_dir, cfg.checkpoint_filename)


        val_loss = tf.summary.scalar('Val_Loss', model.val_loss)
        val_acc_op = tf.summary.scalar('Batch_Val_Acc', model.val_accuracy)
        model_acc_op = tf.summary.scalar('Split_Val_Accuracy', model.val_accumulated_accuracy)

       
                
        run_metadata = tf.RunMetadata()
        tf.local_variables_initializer().run()
        sess.run(validation_iterator.initializer)

        _val_acc_op = 0
        feat =[]
        label =[]
        pooling =[]
        while True:
            try:
                # Eval network on validation/testing split                            
                feed_dict = {handle: validation_handle}
                features, labels= sess.run([model.val_end_features,model.val_features_labels], feed_dict)
                
                print(labels.shape)
                feat.append(features['resnet_v2_50/block4'])
                pooling.append(features['global_pool'])
                label.append(labels)
                print('___________________')
            except tf.errors.OutOfRangeError:

                path = model_dir
                f_folder = os.path.join(model_dir,'features')
                os.makedirs(f_folder,exist_ok=True)
                p_file = os.path.join(f_folder,'pooling.npy')
                f_file = os.path.join(f_folder,'features.npy')
                l_file = os.path.join(f_folder,'labels.npy')
                print('pooling')
                pooling = np.concatenate(pooling)
                pooling = pooling.reshape(pooling.shape[0],-1)
                np.save(p_file,pooling)
                print('7x7')
                np.save(f_file,feat)
                print('labels')
                np.save(l_file,np.array(label))
                break
        #sess.close()


if __name__ == '__main__':
    import os 
    import json 
    parent_folder = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints'
    path = os.path.join(parent_folder,
                    'leaves_fossils_resnet50_leaves_lr0.01_B45_caf10_iter10K_lambda1_trn_mode_hard_anchor__anchor_max0/')
   
    args_file = os.path.join(path,'args.json')
    with open(args_file,'r') as f:
        arguments = json.load(f)
    print(arguments)
    num_trials = 1
    username = arguments['username']
    arg_db_name = arguments['db_name']
    print(arg_db_name)
    arg_net = arguments['net']
    arg_train_mode = arguments['train_mode']
    lr = str(arguments['learning_rate'])#'0.01'
    for idx in range(num_trials):
         args = [
             '--gpu', '6',
             '--db_name', arg_db_name,
             '--net', arg_net,
             '--train_mode', arg_train_mode,
             '--margin', str(arguments['margin']),
             '--batch_size',str(arguments['batch_size']),
             '--caffe_iter_size',str(arguments['caffe_iter_size']),
             '--logging_threshold', str(arguments['logging_threshold']),
             '--train_iters', str(arguments['train_iters']),
             '--test_interval',str(arguments['test_interval']),
             '--learning_rate', lr,
             '--aug_style', 'img',
             '--username',username,
             '--checkpoint_suffix',arguments['checkpoint_suffix']#'_validation_5050_pretrained_log_'+str(idx)#'_debug_anchor_resize_fixed_299_' + str(idx)

             # These flags are used for different experiments
             # '--frame_size','299',
         ]


         main(args)


