import argparse
import os
import sys

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import ssl

import myutils
from model import Model
import scipy.io
import numpy as np

from torch.autograd import Variable

ssl._create_default_https_context = ssl._create_unverified_context

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # out = torch.cat([out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], dim=0)
        # [8*B, 8*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
       

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top2, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.target, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)
            print("feature")
            print(feature.shape)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            print(feature_labels.size())
            print(feature_labels.expand(data.size(0),-1))

            sim_indices = sim_indices.type(torch.int64)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_labels = sim_labels.type(torch.int64)
            sim_weight = (sim_weight / temperature).exp()
            print('sim_label_shape')
            print(sim_labels.shape)


            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            print('one_hot_label-shape')
            print(one_hot_label.shape)

            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            print(pred_labels)

            with open('preds_256bt_1000ep.csv','a') as fd:
                fd.write( ','.join(map(str, pred_labels.detach().tolist())) + '\n')
                fd.write( ','.join(map(str, target.unsqueeze(dim=-1).detach().tolist())) + '\n')
                fd.write( ','.join(map(str, '-------------------------------------------------------------------------------------'+'\n')))
            # test_bar.set_description('Test pred_labels{%s}'.format(str(pred_labels[:,:2])))
            # test_bar.set_description('Test targets{%s}'.format(str(target.unsqueeze(dim=-1))))
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top2 += torch.sum((pred_labels[:, :2] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@2:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top2 / total_num * 100))

    return total_top1 / total_num * 100, total_top2 / total_num * 100


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    print('loop')
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=2, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch') #512
    parser.add_argument('--epochs', default=2, type=int, help='Number of sweeps over the dataset to train')
    
    x = Variable(torch.rand(2,3).float())
    print(x.data[0])
    #root_dir = 'C://Users//camalas.DEACNET//Project//Other_Dataset//'    
    root_dir = '//deac//csc//paucaGrp//camalas//DataSets//Pond32//'

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    print(sys.path)

    # data prepare
    # train_data = myutils.CIFAR10Pair(root='data', 
    #                                train=True, 
    #                                transform=myutils.train_transform, 
    #                                download=False) #download=True)
    #Xn_s2 = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Planet_S1S2_training_data//S2_P_32_lands///train_data_s2_all.mat');#train_data.mat');
    #Xn_p = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Planet_S1S2_training_data//S2_P_32_lands//train_data_p_all.mat');#train_data_swnr.mat');
    #Yn = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Planet_S1S2_training_data//S2_P_32_lands//train_label_all.mat');#train_label.mat');
    Xn_s2 = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//train_data_s2_all.mat');
    Xn_p = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//train_data_p_all.mat');
    Yn = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//train_label_all.mat');
    
    Xtrain_s2 = Xn_s2['train_data_s2'];
    Xtrain_p = Xn_p['train_data_p'];
    Ytrain = Yn['train_label'].reshape((-1,));
    #Xtrain = Xtrain[0:16,:];
    #Xtrain_sw = Xtrain_sw[0:16,:];
    #Ytain = Ytrain[0:16];
    train_data = myutils.PondPair( root_dir, Xtrain_s2,Xtrain_p,Ytrain,train=True,transform=myutils.train_transform)  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                              drop_last=True)
    
    print(Ytrain.shape)
    print(Xtrain_s2.shape)
    # memory_data = myutils.CIFAR10Pair(root='data', train=True, transform=myutils.test_transform, download=True)
    #Xv = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_data_new_256.mat');
    #Xv_sw = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_data_swnr_new_256.mat');
    #Yv = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_label_new_256.mat');
    Xv_s2 = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//val_data_s2_256.mat');
    Xv_p = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//val_data_p_256.mat');
    Yv = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//val_label_256.mat');

    Xval_s2 = Xv_s2['val_data_s2_256'];
    Xval_p = Xv_p['val_data_p_256'];
    Yval = Yv['val_label_256'].reshape((-1,));
    memory_data = myutils.PondPair(root_dir, Xval_s2,Xval_p,Yval,train=True, transform=myutils.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print(Yval.shape)
    print(Xval_s2.shape)

    # test_data = myutils.CIFAR10Pair(root='data', train=False, transform=myutils.test_transform, download=True)
    #Xt = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_data_new_256.mat');#test_data.mat');
    #Xt_sw = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_data_swnr_new_256.mat');#test_data_swnr.mat');
    #Yt = scipy.io.loadmat('C://Users//camalas.DEACNET//Project//Other_Dataset//val_label_new_256.mat');#test_label.mat');
    Xt_s2 = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//test_data_s2_all.mat');
    Xt_p = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//test_data_p_all.mat');
    Yt = scipy.io.loadmat('//deac//csc//paucaGrp//camalas//DataSets//S2_P_32_lands//test_label.mat');

    Xtest_s2 = Xt_s2['test_data_s2'];
    Xtest_p = Xt_p['test_data_p'];
    Ytest = Yt['test_label'].reshape((-1,));
    test_data = myutils.PondPair(root_dir, Xtest_s2,Xtest_p,Ytest,train=False, transform=myutils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print(Ytest.shape)
    print(Xtest_s2.shape)

    
    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 4, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    #c = len(memory_data.classes)
    c = len(np.unique(memory_data.target))

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@2': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_2 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@2'].append(test_acc_2)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics_rgb_1000ep.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model_rgb_500ep.pth'.format(save_name_pre))
