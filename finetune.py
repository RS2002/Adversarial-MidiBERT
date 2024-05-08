import numpy as np
import tqdm
import torch
import torch.nn as nn
from model import MidiBert, TokenClassification, SequenceClassification
import argparse
from transformers import BertConfig,AdamW
import pickle
import os
from torch.utils.data import DataLoader
from dataset import FinetuneDataset


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('--task', type=str, required=True)
    ### dataset & data root ###
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default=None)
    ### path setup ###
    parser.add_argument('--dict_file', type=str,default='./Data/Octuple.pkl')
    parser.add_argument('--model_path', default='./midibert_pretrain.pth')
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int,
                        default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")
    ### cuda ###
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda_devices", type=int, nargs='+',
                        default=[5, 7], help="CUDA device ids")
    args = parser.parse_args()

    # check args
    if args.class_num is None:
        if args.task == 'melody':
            args.class_num = 4
        elif args.task == 'velocity':
            args.class_num = 7
        elif args.task == 'composer':
            args.class_num = 8
        elif args.task == 'emotion':
            args.class_num = 4

    return args

def load_data(dataset, data_root=None):
    if data_root is None:
        data_root = 'Data/finetune/others'

    X_train = np.load(os.path.join(
        data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(
        data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(
        data_root, f'{dataset}_test.npy'), allow_pickle=True)
    print('X_train: {}, X_valid: {}, X_test: {}'.format(
        X_train.shape, X_val.shape, X_test.shape))

    y_train = np.load(os.path.join(
        data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(
        data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(
        data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)
    print('y_train: {}, y_valid: {}, y_test: {}'.format(
        y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test

def iteration(model,midibert,optim,data_loader,task,device,train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    acc_list=[]
    loss_list=[]

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x,label in pbar:
        x=x.to(device)
        label=label.to(device)
        x = x.long()
        label = label.long()
        attn_mask = (x[..., 0] != midibert.bar_pad_word).float().to(device)
        y=model(x,attn=attn_mask)
        output = torch.argmax(y,dim=-1)
        if task=="emotion" or task=="composer":
            loss_func = nn.CrossEntropyLoss()
            loss=loss_func(y,label)
            acc = torch.mean((output==label).float())
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            loss = loss_func(y,label)
            loss = torch.sum(loss*attn_mask)/torch.sum(attn_mask)
            acc = torch.sum((output == label).float()*attn_mask)/torch.sum(attn_mask)

        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

        acc_list.append(acc.item())
        loss_list.append(loss.item())

    return np.mean(loss_list),np.mean(acc_list)

def main():
    args=get_args()
    cuda_devices=args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device=torch.device(device_name)
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type='relative_key_query',
                               hidden_size=args.hs)
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e).to(device)
    if not args.nopretrain:
        midibert.load_state_dict(torch.load(args.model_path))
    task = args.task
    if task=="composer" or task=="emotion":
        model = SequenceClassification(midibert, args.class_num, args.hs).to(device)
    else:
        model = TokenClassification(midibert, args.class_num+1, args.hs).to(device)
    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.dataset, args.dataroot)
    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(test_loader))
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_acc=0
    best_loss=1e8
    acc_epoch=0
    loss_epoch=0
    result_acc=None
    j = 0
    while True:
        j+=1
        loss,acc=iteration(model, midibert, optim, train_loader, task, device, train=True)
        log = "Epoch {:} | Train Loss {:06f} Train Acc {:06f} | ".format(j,loss,acc)
        with open(args.task+"_"+args.dataset+".txt",'a') as file:
            file.write(log)
        print(log)
        loss,acc=iteration(model, midibert, optim, valid_loader, task, device, train=False)
        log = "Valid Loss {:06f} Valid Acc {:06f} | ".format(loss,acc)
        with open(args.task+"_"+args.dataset+".txt",'a') as file:
            file.write(log)
        print(log)
        test_loss,test_acc=iteration(model, midibert, optim, test_loader, task, device, train=False)
        log = "Test Loss {:06f} Test Acc {:06f}".format(test_loss,test_acc)
        with open(args.task+"_"+args.dataset+".txt",'a') as file:
            file.write(log+"\n")
        print(log)

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), args.task+"_"+args.dataset+".pth")
            result_acc = test_acc
        if acc >= best_acc:
            best_acc = acc
            acc_epoch = 0
        else:
            acc_epoch += 1
        if loss < best_loss:
            best_loss = loss
            loss_epoch = 0
        else:
            loss_epoch += 1
        if acc_epoch >= args.epochs and loss_epoch >= args.epochs:
            break
        print("Acc Epoch {:}, Loss Epcoh {:}, Result Acc {:}".format(acc_epoch, loss_epoch,result_acc))

if __name__ == '__main__':
    main()