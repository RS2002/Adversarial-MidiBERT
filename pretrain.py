from model import MidiBert,MidiBertLM,Masker,Discriminator
import argparse
from transformers import BertConfig,AdamW
import torch
import pickle
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import MidiDataset
import tqdm
import random
import copy
import math

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['EMOPIA', 'Pianist8', 'POP1K7', 'POP909'])
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    return args

def load_data(datasets):
    to_concat = []
    #root = '../../Data/output'
    root = './Data/output_pretrain'
    # for dataset in datasets:
    #     data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    #     print(f'   {dataset}: {data.shape}')
    #     to_concat.append(data)
    for dataset in datasets:
        '''data_train = np.load(os.path.join(root, dataset, 'midi_train_split.npy'), allow_pickle = True)
        data_test = np.load(os.path.join(root, dataset, 'midi_test_split.npy'), allow_pickle = True)
        data_valid = np.load(os.path.join(root, dataset, 'midi_valid_split.npy'), allow_pickle = True)'''
        data_train = np.load(os.path.join(root, dataset, dataset+'_train_split.npy'), allow_pickle=True)
        data_test = np.load(os.path.join(root, dataset, dataset+'_test_split.npy'), allow_pickle=True)
        data_valid = np.load(os.path.join(root, dataset, dataset+'_valid_split.npy'), allow_pickle=True)
        data = np.concatenate((data_train, data_test, data_valid), axis = 0)
        print(f'   {dataset}: {data.shape}')
        to_concat.append(data)

    training_data = np.vstack(to_concat)
    print('   > all training data:', training_data.shape)
    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data) * 0.85)
    X_train, X_val = training_data[:split], training_data[split:]
    return X_train, X_val

def get_mask_ind(mask_ind=None,mask_percent=None,max_seq_len=1024):
    if mask_ind is None:
        Lseq = [i for i in range(max_seq_len)]
        mask_ind = random.sample(Lseq, round(max_seq_len * mask_percent))
    else:
        mask_ind=mask_ind.cpu().tolist()
    mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
    left = list(set(mask_ind)-set(mask80))
    rand10 = random.sample(left, round(len(mask_ind)*0.1))
    cur10 = list(set(left)-set(rand10))
    return mask80, rand10, cur10


def get_weight(acc_list):
    sum=np.sum(acc_list)
    acc_list=torch.tensor(acc_list)
    weight_list=sum/acc_list
    weight_list/=torch.sum(weight_list)
    return weight_list

def iteration(midibert,model,masker,discriminator,optim_midibert,optim_model,optim_masker,optim_discriminator,mask_book,data_loader,device,epoch,train=True,last_acc=None):
    if train:
        midibert.train()
        model.train()
        masker.train()
        # discriminator.train()
        torch.set_grad_enabled(True)
    else:
        midibert.eval()
        model.eval()
        torch.set_grad_enabled(False)
    pbar = tqdm.tqdm(data_loader, disable=False)
    acc_list=[[],[],[],[],[],[],[],[]]
    loss_list=[]

    for x,index in pbar:
        x=x.to(device)
        batch,seq_len,_ = x.shape
        input_ids = copy.deepcopy(x)
        loss_mask = torch.zeros(batch, seq_len).to(device)
        real_mask = torch.zeros(batch, seq_len).to(device)
        attn_mask = (x[..., 0] != midibert.bar_pad_word).float().to(device)
        mask_num_list = []
        if train:
            mask_percent = masker(x,attn=attn_mask)
            current_mask_book = mask_book[index]
            mask_percent = mask_percent * current_mask_book
            for b in range(batch):
                # mask_num = int(random.uniform(0.1, 0.7)*torch.sum(current_mask_book[b]))
                mask_num = min(int(random.uniform(0.1, 0.3)*torch.sum(current_mask_book[b])),int(seq_len*0.15))

                mask_num_list.append(mask_num)
                _,mask_ind = torch.topk(mask_percent[b],mask_num)
                mask80, rand10, cur10 = get_mask_ind(mask_ind=mask_ind)
                for i in mask80:
                    mask_word = torch.tensor(midibert.mask_word_np).to(device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                    real_mask[b][i]=1
                for i in rand10:
                    rand_word = torch.tensor(midibert.get_rand_tok()).to(device)
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                    real_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1
        else:
            for b in range(batch):
                # mask80, rand10, cur10 = get_mask_ind(mask_percent=random.uniform(0.1, 0.4))
                mask80, rand10, cur10 = get_mask_ind(mask_percent=0.15)
                for i in mask80:
                    mask_word = torch.tensor(midibert.mask_word_np).to(device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                for i in rand10:
                    rand_word = torch.tensor(midibert.get_rand_tok()).to(device)
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1


        y = model.forward(input_ids, attn_mask)
        generation=None
        loss_cls = nn.CrossEntropyLoss(reduction="none")
        loss_mse = nn.MSELoss()
        loss = 0
        loss_mask=loss_mask.reshape([batch*seq_len])
        loss_total=torch.zeros_like(loss_mask).to(device)
        if last_acc is not None:
            weight=get_weight(last_acc)
        else:
            weight=[0.125]*8
        for i in range(8):
            predict=y[i].reshape([batch*seq_len,-1])
            gt=x[...,i].reshape([batch*seq_len])
            loss_current=loss_cls(predict,gt.long()) * weight[i]
            loss_total+=loss_current
            loss += torch.sum(loss_current*loss_mask)/torch.sum(loss_mask)
            loss += torch.mean(loss_current)
            output = torch.argmax(predict,dim=-1)
            acc = torch.sum((output==gt).float()*loss_mask)/torch.sum(loss_mask)
            acc_list[i].append(acc.item())
            output=output.reshape(batch,seq_len,1)
            if generation is None:
                generation=output
            else:
                generation=torch.concat([generation,output],dim=-1)
        loss_list.append(loss.item())
        if train:
            # loss_mask=loss_mask.reshape([batch,seq_len])
            loss_mask=real_mask.reshape([batch,seq_len])
            loss_total=loss_total.reshape([batch,seq_len])

            loss_masker=0
            for b in range(batch):
                select_num = int(0.3*mask_num_list[b])
                loss_total[b][loss_mask[b] == 0] = 0
                _,index1=torch.topk(loss_total[b],select_num)
                gt1=torch.ones_like(mask_percent[b,index1]).to(device)
                loss1=loss_mse(mask_percent[b,index1],gt1)
                loss_total[b][loss_mask[b]==0]=1e8
                loss_total=-loss_total
                _,index0=torch.topk(loss_total[b],select_num)
                gt0=torch.zeros_like(mask_percent[b,index0]).to(device)
                loss0=loss_mse(mask_percent[b,index0],gt0)
                loss_masker=loss_masker+(loss1+loss0)/batch


            # alpha=2.0/(1.0+math.exp(-10*epoch/100))-1
            # generation=generation.to(device)
            #
            # # result=discriminator(generation,attn=attn_mask,alpha=alpha,x=y)
            # # real = real_mask.reshape(-1).long()
            # # result = result.reshape(-1,2)
            # # loss_discriminator=torch.mean(loss_cls(result,real))
            #
            # real_hat = discriminator(x,attn=attn_mask,alpha=alpha)
            # false_hat = discriminator(generation,attn=attn_mask,alpha=alpha,x=y)
            # real = torch.ones(x.shape[0]).long().to(device)
            # false = torch.zeros(x.shape[0]).long().to(device)
            # loss_discriminator=torch.mean(loss_cls(real_hat,real))+torch.mean(loss_cls(false_hat,false))
            loss_discriminator = 0

            loss=loss+loss_discriminator+loss_masker

            model.zero_grad()
            masker.zero_grad()
            # discriminator.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            nn.utils.clip_grad_norm_(masker.parameters(), 3.0)
            # nn.utils.clip_grad_norm_(discriminator.parameters(), 3.0)
            optim_midibert.step()
            optim_model.step()
            optim_masker.step()
            # optim_discriminator.step()
    acc=[]
    for a in acc_list:
        acc.append(np.mean(a))
    return np.mean(loss_list),acc,np.mean(acc)


def update_maskbook(midibert,masker,mask_book,train_loader,device):
    midibert.eval()
    masker.eval()
    torch.set_grad_enabled(False)
    pbar = tqdm.tqdm(train_loader, disable=False)
    for x,index in pbar:
        x=x.to(device)
        batch = x.shape[0]
        attn_mask = (x[..., 0] != midibert.bar_pad_word).float().to(device)
        percentage=masker(x,attn=attn_mask)
        current_mask_book=mask_book[index]
        percentage=percentage*current_mask_book
        for b in range(batch):
            freeze_num = int(random.uniform(0.1, 0.3) * torch.sum(current_mask_book[b]))
            _, freeze_ind = torch.topk(percentage[b], freeze_num)
            mask_book[index[b],freeze_ind]=0
            unfreeze_prop = random.uniform(0, 0.1)
            rand = torch.rand(x.shape[1])
            mask_book[index[b]][rand<unfreeze_prop]=1


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
    model = MidiBertLM(midibert).to(device)
    masker = Masker(midibert,hs=args.hs).to(device)
    # discriminator = Discriminator(midibert,hs=args.hs).to(device)
    discriminator = None
    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)
        masker = nn.DataParallel(masker, device_ids=cuda_devices)
        # discriminator = nn.DataParallel(discriminator, device_ids=cuda_devices)
    optim_midibert = AdamW(midibert.parameters(), lr=args.lr, weight_decay=0.01)
    optim_model = AdamW(set(model.parameters())-set(midibert.parameters()), lr=args.lr, weight_decay=0.01)
    optim_masker = AdamW(set(masker.parameters())-set(midibert.parameters()), lr=args.lr, weight_decay=0.01)
    # optim_discriminator = AdamW(set(discriminator.parameters())-set(midibert.parameters()), lr=args.lr, weight_decay=0.01)
    optim_discriminator = None

    X_train, X_val = load_data(datasets=args.datasets)
    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    mask_book = torch.ones([X_train.shape[0],args.max_seq_len]).to(device)
    # mask_book[X_train[...,0]==midibert.bar_pad_word]=0

    best_acc=0
    best_loss=1e8
    acc_epoch=0
    loss_epoch=0

    j = 0
    acc_list=None
    while True:
        j+=1
        loss,acc_list,acc=iteration(midibert,model, masker, discriminator, optim_midibert, optim_model, optim_masker, optim_discriminator, mask_book, train_loader,
                  device, j, train=True, last_acc=acc_list)
        log = "Epoch {:} | Train Acc {:06f} Test Loss {:06f} ".format(j,acc,loss)
        print(log)
        with open("pretrain.txt",'a') as file:
            file.write(log)
        log = "Acc Detail [{:06f},{:06f},{:06f},{:06f},{:06f},{:06f},{:06f},{:06f}] | ".format(acc_list[0],acc_list[1],acc_list[2],acc_list[3],acc_list[4],acc_list[5],acc_list[6],acc_list[7])
        print(log)
        with open("pretrain.txt",'a') as file:
            file.write(log)
        loss,acc_list,acc=iteration(midibert,model, masker, discriminator, optim_midibert, optim_model, optim_masker, optim_discriminator, mask_book, test_loader,
                          device, j, train=False)
        log = "Test Acc {:06f} Test Loss {:06f} ".format(acc,loss)
        print(log)
        with open("pretrain.txt",'a') as file:
            file.write(log)
        log = "Acc Detail [{:06f},{:06f},{:06f},{:06f},{:06f},{:06f},{:06f},{:06f}]".format(acc_list[0],acc_list[1],acc_list[2],acc_list[3],acc_list[4],acc_list[5],acc_list[6],acc_list[7])
        print(log)
        with open("pretrain.txt",'a') as file:
            file.write(log+"\n")
        if j%5==0 :
            update_maskbook(midibert, masker, mask_book, train_loader, device)
        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), "pretrain_model.pth")
            torch.save(midibert.state_dict(), "midibert_pretrain.pth")
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
        if (acc_epoch >= 30 and loss_epoch >= 30) or j>args.epochs:
            break
        print("Acc Epoch {:}, Loss Epcoh {:}".format(acc_epoch, loss_epoch))

if __name__ == '__main__':
    main()