from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np 
import os 
features_path = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/validation_pnas_resnet50_leaves_pretrained_lr0.01_B45_caf10_iter15K_lambda1_trn_mode_hard__validation_5050_pretrained_log_0/features/'

train_poolin_path = os.path.join(features_path,'train_pooling.npy')
train_labels_path = os.path.join(features_path,'train_labels.npy')
test_poolin_path = os.path.join(features_path,'pooling.npy')
test_labels_path = os.path.join(features_path,'labels.npy')

x_train = np.load(train_poolin_path,allow_pickle=True)
y_train = np.concatenate(np.load(train_labels_path,allow_pickle=True))

x_test = np.load(test_poolin_path,allow_pickle=True)
y_test = np.concatenate(np.load(test_labels_path,allow_pickle=True))
print('fitting')
)
clf = svm.SVC()
clf.fit(x_train,y_train)
print('predicting')
preds = clf.predict(x_test)

print(classification_report(y_test,preds))