# -*- coding:utf-8 -*-
## *************** main training code for the BI-LSTM-CRF model.*******************************

import torch

import torch.optim as optim
import pickle
import os
import argparse
from model.bi_lstm_crf import BiLSTM_CRF
from Utils.data_load import DatasetLoader
from torch.backends import cudnn
import numpy as np


torch.manual_seed(1)


def load_dictionary(dict_path):
    with open(dict_path, 'rb') as f:
        dic= pickle.load(f, encoding='latin1')
        f.close()

    return dic

def save_network(network,optimizer, epoch_label):
    if not os.path.isdir(args.load_check_point):
        os.makedirs(args.load_check_point)
    save_filename = 'net_epoch_%s_id.pth' % (epoch_label)
    save_path = os.path.join(args.load_check_point, save_filename)

    torch.save({'epoch':epoch_label,'state_dict':network.state_dict()}, save_path)
    print('saved net: %s' % save_path)

    save_filename = 'opt_epoch_%s_id.pth' % (epoch_label)
    save_path = os.path.join(args.load_check_point, save_filename)
    torch.save({'epoch': epoch_label, 'state_dict': optimizer.state_dict()}, save_path)
    print('saved net: %s' % save_path)

def eval_sequence(pred_s, target, mask):
    """
    :param pred_s: (batch_size, sequence_length)
    :param target: (batch_size, sequence_length)
    :param mask: (batch_size,) a number
    :return:  precison, recall and f1_score
    """
    pre_sum, re_sum,f1_sum=0.0,0.0,0.0
    target=target.cpu().numpy()
    mask=mask.cpu().numpy()
    batch_size=len(mask)
    ns=0
    for n in range(batch_size):
        length=mask[n]
        pred_y=np.array(pred_s[n])
        true_y=target[n][1:length+1]

        assert len(pred_y)==len(true_y)
        if np.sum(true_y)==0:
            ns+=1
        else:
            TP=sum(np.logical_and(pred_y , true_y))
            FN=np.sum(true_y)-TP
            TN=length-np.sum(np.logical_or(pred_y, true_y))
            FP=np.sum(pred_y)-TP
            if TP+FP==0:
                pre=0
            else:
                pre=TP/(TP+FP)
            pre_sum+=pre
            if TP+FN==0:
                re=0
            else:
                re=TP/(TP+FN)
            re_sum+=re
            if re+pre==0:
                f1_sum+=0
            else:
                f1_sum+=2*re*pre/(re+pre)

    return pre_sum/(batch_size-ns), re_sum/(batch_size-ns), f1_sum/(batch_size-ns)

def display_evaluation(validate_data_loaders,model,epoch,mode):
    results=np.zeros((10000,3))
    iter=0
    for data_loader in validate_data_loaders:
        score_val = 0
        precision_val = 0
        recall_val = 0
        f1_val = 0

        for i, [data_ev,mask] in enumerate(data_loader):  ## data_ev:(batch_size, 2, length)

            bt_size =len(mask)
            score, pred_seqs = model.predict_vb(data_ev[:,0],mask)
            score_val += score
            # for j, seq in enumerate(pred_seqs):   ## j :0 --  batch_size
            precision, recall, f1 = eval_sequence(pred_seqs, data_ev[:,1],mask)

            print("precision: %f, recall: %f, f1_score: %f." % (precision, recall, f1))
            results[iter,0]=precision
            results[iter,1]=recall
            results[iter,2]=f1
            iter += 1
    if mode=='val':
        with open('/home/sh_jun/Bi_LSTM_CRF_test_sent/intermediate/validae_res_sent.pkl', 'wb') as f:
            pickle.dump(results, f)
            f.close()
    elif mode=='train':
        with open('/home/sh_jun/Bi_LSTM_CRF_test_sent/intermediate/train_res_sent.pkl', 'wb') as f:
            pickle.dump(results, f)
            f.close()
    else:
        with open('/home/sh_jun/Bi_LSTM_CRF_test_sent/intermediate/test_res_sent.pkl', 'wb') as f:
            pickle.dump(results, f)
            f.close()

    return score_val




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with BLSTM-CRF')
    parser.add_argument('--root', default='./Datasets/KP_sentence/', help='root path to data set')
    parser.add_argument('--train_file', default='train_data.pkl', help='path to training file')
    parser.add_argument('--valid_file', default='valid_data.pkl', help='path to development file')
    parser.add_argument('--test_file', default='test_data.pkl', help='path to test file')
    parser.add_argument('--dict_file', default='dictionary.pkl', help='path to development file')
    parser.add_argument('--pretrained_embedding', default='Embedding.pkl', help='pretrained word embedding')

    parser.add_argument('--dataset_name', default='KP', choices=['KP', 'WWW', 'KDD'])
    parser.add_argument('--mode', default='train', choices=['train', 'valid', 'test'])

    parser.add_argument('--embedding_dim', type=int, default=100, help='dimension for word embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--unk', default='*unknown*', help='unknow-token in pre-trained embedding')
    parser.add_argument('--start', default='*start*', help='start-token in pre-trained embedding')
    parser.add_argument('--pad', default='*pad*', help='padding-token in pre-trained embedding')
    parser.add_argument('--end', default='*end*', help='end-token in pre-trained embedding')
    parser.add_argument('--START_TAG', default='start', help='start-tag in tag dictionary')
    parser.add_argument('--STOP_TAG', default='end', help='end-tag in tag dictionary')


    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size (10)')
    parser.add_argument('--checkpoint', default='./checkpoint/sentence', help='path to checkpoint prefix')
    parser.add_argument('--epoch', type=int, default=1, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    # parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-4, help='decay ratio of learning rate') ## 0.05 ??
    parser.add_argument('--drop_out', type=float, default=0.55, help='dropout ratio')

    parser.add_argument('--load_check_point', default='./checkpoint/sent', help='path to load pretrained model')
    parser.add_argument('--load_epoch', default=-1, help='load optimizer from ')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer method')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')

    args = parser.parse_args()

    cudnn.benchmark = True

    torch.cuda.set_device(args.gpu)   ## only use one GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ## construct dataset
    train_data_loaders= DatasetLoader(args,mode='train')
    validate_data_loaders= DatasetLoader(args,mode='valid')
    test_data_loaders=DatasetLoader(args,mode='test')



    ## load dictioanry
    dict_path=os.path.join(args.root,args.dict_file)
    word2id, id2word = load_dictionary(dict_path)
    tag_to_ix = {"nkp": 0, "kp": 1, "start": 2, 'pad':3, 'end': 4} ## tag_to_idx dictionary for final label

    ##  Build model
    model = BiLSTM_CRF(args, len(word2id), tag_to_ix, args.embedding_dim, args.hidden_dim)

    model.cuda()
    model.eval()


    if args.update == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lr_decay,
                                   momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)

    ## load_model
    if args.load_epoch>=0:

        if os.path.isfile(model.load_model_path):
            print("loading checkpoint: '{}'".format(model.load_model_path))
            device = torch.device("cuda")
            checkpoint_file = torch.load(model.load_model_path, map_location=device)

            args.start_epoch = checkpoint_file['epoch']
            model.load_state_dict(checkpoint_file['state_dict'])

        if os.path.isfile(model.load_opt_path):
            print("loading checkpoint: '{}'".format(model.load_opt_path))
            checkpoint_file = torch.load(model.load_opt_path, map_location=device)
            optimizer.load_state_dict(checkpoint_file['state_dict'])

    else:
        pass

    ## test model
    print("test training start")
    display_evaluation(train_data_loaders, model, 0, 'train')

    ## test model
    print("Validation start")
    display_evaluation(validate_data_loaders, model, 0,'val')

    print("Test start")
    display_evaluation(test_data_loaders, model, 0,'test')





