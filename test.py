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
        gts=[]
        preds=[]
        pred_3=[]
        pred_5=[]
        while True:
            try:
                # Eval network on validation/testing split                            
                feed_dict = {handle: validation_handle}
                gt,preds_raw,predictions,acc_per_class,val_loss_op, batch_accuracy, accuracy_op, _val_acc_op, _val_acc, c_cnf_mat,macro_acc = sess.run(
                                [model.val_gt,model.val_preds,model.val_class_prediction,model.val_per_class_acc_acc,val_loss, model.val_accuracy, model_acc_op, val_acc_op, model.val_accumulated_accuracy,
                                 model.val_confusion_mat,model.val_per_class_acc_acc], feed_dict)
                gts+=list(gt)
                preds+=list(predictions)

                for g,p in zip(gt,preds_raw):
                    preds_sort_3= np.argsort(p)[-3:]
                    preds_sort_5= np.argsort(p)[-5:]
                    if g in preds_sort_3:
                        pred_3+=[g]
                    else:
                        pred_3+=[preds_sort_3[-1]]

                    if g in preds_sort_5:
                        pred_5+=[g]
                    else:
                        pred_5+=[preds_sort_5[-1]]

                #print('Acc per class:',acc_per_class)
                #print('batch:',batch_accuracy)
                #print('Confusion Matrix:',c_cnf_mat)
                #print('gt:',gt)
                #print('preds:',preds_raw)
                #print('predictions:',predictions)
            #logger.info('Val Acc {0}, Macro Acc: {1}'.format(_val_acc,macro_acc))
            except tf.errors.OutOfRangeError:
            #    logger.info('problem:')
            #    logger.info('Val Acc {0}, Macro Acc: {1}'.format(_val_acc,macro_acc))
                logger.info('____ Clasification Report Top 1 ____')
                report = classification_report(gts,preds,output_dict=True)
                csv_pd = classification_report_csv(report)
                csv_pd.to_csv(os.path.join(model_dir,'Classification_Report_top1.csv'))
                logger.info(report)
                logger.info('____ Clasification Report Top 2 ____')
                report = classification_report(gts,pred_3,output_dict=True)
                csv_pd = classification_report_csv(report)
                csv_pd.to_csv(os.path.join(model_dir,'Classification_Report_top2.csv'))
                logger.info(report)
                logger.info('____ Clasification Report Top 3 ____')
                report = classification_report(gts,pred_5,output_dict=True)
                csv_pd = classification_report_csv(report)
                csv_pd.to_csv(os.path.join(model_dir,'Classification_Report_top3.csv'))
                logger.info(report)

                break

                   
        #sess.close()


if __name__ == '__main__':
    

    num_trials = 1
    arg_db_name = 'leaves_fossils'
    arg_net = 'resnet50_leaves'
    arg_train_mode = 'hard'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '2',
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
            '--aug_style', 'img',
            '--checkpoint_suffix','_debug_threshold_3_18_classes_'+str(idx)#'_debug_anchor_resize_fixed_299_' + str(idx)

            # These flags are used for different experiments
            # '--frame_size','299',
        ]


        main(args)


