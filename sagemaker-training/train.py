import argparse
import math
import time
import numpy as np
import random
import os
import json
import sys
 
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import RandomSampler

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

import sentencepiece as sp

from prepare_data import data_chunk_iter,prepare_dataloader

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def decoder_gold_splitter(trg):
    # shifts decoder input to the right
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def train_epoch(model, training_data_chunks,validation_data_chunks, optimizer, args, device):
    model.train()
                
    total_samples = len(training_data_chunks)
    
    # restore training from checkpoint
    chunk_num = 1
    process_samples_used = 0
    total_samples_used = 0
    total_loss = 0
    n_word_correct = 0
    n_word_total = 0
    min_val_loss = None
    checkpoint_file = os.path.join(args.checkpoint_dir,'best_checkpoint.pt')
    if os.path.exists(checkpoint_file) and args.curr_epoch == 1:
        checkpoint_info = torch.load(checkpoint_file)
        assert len(checkpoint_info["processes_info"]) == args.world_size
        chunk_num = checkpoint_info["chunk_num"]
        total_samples_used = checkpoint_info["total_samples_used"]
        total_loss = checkpoint_info["total_loss"]
        n_word_correct = checkpoint_info["n_word_correct"]
        n_word_total = checkpoint_info["n_word_total"]
        min_val_loss = checkpoint_info["val_loss"]
        for process_info in checkpoint_info["processes_info"]:
            rank, num_samples_used = process_info
            if args.host_rank == rank:
                process_samples_used = num_samples_used

        model.load_state_dict(checkpoint_info["state_dict"])
        if args.host_rank == 0:
            print("starting from best checkpoint") 
    training_data_chunks.skip_chunks(chunk_num-1)
    if args.checkpoint_samples_num is not None:
        checkpoint_threshold = (total_samples_used//args.checkpoint_samples_num + 1)*(args.checkpoint_samples_num)
    
    start = time.time()
    samples_used_in_current_execution = 0
    
    for chunk in training_data_chunks:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(chunk) if args.is_distributed else RandomSampler(chunk)
        )
        training_data = prepare_dataloader(chunk,args.batch_size,sampler,args.bos_id,args.eos_id,args.pad_id,process_samples_used)
        
        for src_batch,trg_batch in training_data:
            src_seq = src_batch.to(device)
            trg_seq, gold = map(lambda x: x.to(device), decoder_gold_splitter(trg_batch)) 

            # forward
            optimizer.zero_grad()
            pred = model(src_seq, trg_seq)

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(
                pred, gold, args.pad_id) 
            loss.backward()
            if args.is_distributed:
                average_gradients(model)
            optimizer.step_and_update_lr()

            # performance note-keeping
            if args.is_distributed:
                n_correct,n_word = torch.tensor(n_correct),torch.tensor(n_word)
                dist.all_reduce(n_correct,op=dist.ReduceOp.SUM)
                dist.all_reduce(n_word,op=dist.ReduceOp.SUM)
                dist.all_reduce(loss,op=dist.ReduceOp.SUM)
                n_correct,n_word = n_correct.item(),n_word.item()
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

            # save best checkpoint only
            process_batch_samples = src_batch.size()[0]
            process_samples_used += process_batch_samples
            total_batch_samples = torch.tensor(process_batch_samples)
            if args.is_distributed:
                dist.all_reduce(total_batch_samples, op=dist.ReduceOp.SUM) 
            total_samples_used+= total_batch_samples.item()
            samples_used_in_current_execution += total_batch_samples.item()
            if args.checkpoint_samples_num is not None:
                if total_samples_used >= checkpoint_threshold or total_samples_used==total_samples:
                    checkpoint_threshold += args.checkpoint_samples_num

                    process_info = (args.host_rank,process_samples_used)
                    if args.is_distributed:
                        processes_info = [None for _ in range(args.world_size)]
                        dist.gather_object(
                            process_info,
                            processes_info if args.host_rank == 0 else None,
                            dst=0
                        )
                    else:
                        processes_info = [process_info]
                    val_loss,val_accuracy = eval_epoch(model, validation_data_chunks, device, args)
                    val_ppl = math.exp(min(val_loss,100))
                    if args.host_rank == 0:
                        # save checkpoint
                        best_checkpoint_file = os.path.join(args.checkpoint_dir,"best_checkpoint.pt")
                        if total_samples_used==total_samples:
                            checkpoint_info = {"training_done":True}
                        else:
                            checkpoint_info = {
                                # data use information
                                "processes_info": processes_info,
                                "chunk_num": chunk_num,
                                "total_samples_used": total_samples_used,
                                "training_done": False,

                                # metric tracking
                                "total_loss": total_loss,
                                "n_word_total": n_word_total,
                                "n_word_correct": n_word_correct,
                            }
                        
                        if min_val_loss is None:
                                min_val_loss = val_loss
                                checkpoint_info.update({
                                    "val_loss": min_val_loss,
                                    "state_dict":model.state_dict()
                                    })
                        elif val_loss <= min_val_loss:
                            min_val_loss = val_loss
                            checkpoint_info.update({
                                    "val_loss": min_val_loss,
                                    "state_dict":model.state_dict()
                                    })  
                        else:
                            prev_checkpoint = torch.load(best_checkpoint_file)
                            prev_checkpoint.update(checkpoint_info)
                            checkpoint_info = prev_checkpoint
                        with open(best_checkpoint_file,'wb') as f:
                            torch.save(checkpoint_info,f)
                        
                        # log performance
                        train_loss = total_loss/n_word_total
                        train_ppl = math.exp(min(train_loss,100))
                        train_accuracy = n_word_correct/n_word_total
                        with open(args.train_log_file,'r') as f:
                            checkpoint_num = len(f.readlines())
                        with open(args.train_log_file,'a') as log_file:
                            log_file.write('{checkpoint},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                                    checkpoint=checkpoint_num, loss=train_loss,
                                    ppl=train_ppl, accu=100*train_accuracy))
                        with open(args.val_log_file,'a') as log_file:
                            log_file.write('{checkpoint},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                                    checkpoint=checkpoint_num, loss=val_loss,
                                    ppl=val_ppl, accu=100*val_accuracy))
                        
            # print progess
            if args.host_rank == 0:
                rate = round(samples_used_in_current_execution/(time.time()-start),2)
                samples_left = total_samples - total_samples_used
                print(f"-(training) samples: {total_samples_used}/{total_samples} | {rate} samples/sec | estimated_time_left: {(samples_left/rate)/3600} hrs") 
        chunk_num+=1
        process_samples_used = 0
    
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
     
    return loss_per_word, accuracy

def eval_epoch(model, validation_data_chunks, device, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss, n_word_total, n_word_correct = 0, 0, 0
    process_samples_used = 0
    total_samples_used = 0
    total_samples = len(validation_data_chunks)
    start = time.time()
    for chunk in validation_data_chunks:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(chunk) if args.is_distributed else RandomSampler(chunk)
            )
        val_data = prepare_dataloader(chunk,args.batch_size*2,sampler,args.bos_id,args.eos_id,args.pad_id)
        with torch.no_grad():
            for src_batch,trg_batch in val_data:
                src_seq = src_batch.to(device)
                trg_seq, gold = map(lambda x: x.to(device), decoder_gold_splitter(trg_batch)) 

                # forward
                pred = model(src_seq, trg_seq)
                loss, n_correct, n_word = cal_performance(
                    pred, gold, args.pad_id) 

                # performance note-keeping
                if args.is_distributed:
                    n_correct,n_word = torch.tensor(n_correct),torch.tensor(n_word)
                    dist.all_reduce(n_correct,op=dist.ReduceOp.SUM)
                    dist.all_reduce(n_word,op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss,op=dist.ReduceOp.SUM)
                    n_correct,n_word = n_correct.item(),n_word.item()
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

                # print progress
                process_batch_samples = src_batch.size()[0]
                process_samples_used += process_batch_samples
                total_batch_samples = torch.tensor(process_batch_samples)
                if args.is_distributed:
                    dist.all_reduce(total_batch_samples, op=dist.ReduceOp.SUM) 
                total_samples_used+= total_batch_samples.item()
                if args.host_rank == 0:
                    rate = round(total_samples_used/(time.time()-start),2)
                    samples_left = total_samples - total_samples_used
                    time_left = (samples_left / rate) // 60
                    print(f"-(eval) samples: {total_samples_used}/{total_samples} | {rate} samples/sec | {time_left} mins left")
                if total_samples_used > 10000: break
        if total_samples_used > 10000: break
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data_chunks, validation_data_chunks, optimizer, device, args):

    # create log files
    if args.checkpoint_samples_num is not None:
        args.train_log_file = os.path.join(args.checkpoint_dir, 'train.log')
        args.val_log_file = os.path.join(args.checkpoint_dir, 'val.log')
        if not os.path.exists(args.train_log_file):
            log_files = [args.train_log_file,args.val_log_file]
            for log_file in log_files:
                    with open(log_file,'w') as f:
                        f.write('checkpoint,loss,ppl,accuracy\n')

    for epoch_i in range(args.epoch):
        args.curr_epoch = epoch_i + 1
        train_epoch(model,training_data_chunks,validation_data_chunks,optimizer,args,device)

    
def main():
    parser = argparse.ArgumentParser()
    
    # sentencepiece
    parser.add_argument('--sp_model_file', type=str) # model file name

    # data loading
    parser.add_argument('--load_amount',type=int,default=100000)
    parser.add_argument('--max_seq_len',type=int,default=100)
    parser.add_argument('--train_files_prefix',type=str)
    parser.add_argument('--val_files_prefix',type=str)

    # training parameters
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--steps_ckpt', type=int, default=10)
    parser.add_argument('--src_word_dropout', default=0.2)

    # model hyperparameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embs_share_weight', type=bool, default = True)
    parser.add_argument('--proj_share_weight', type=bool, default=True)
    parser.add_argument('--scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('--backend',type=str,default=None)
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    # checkpoint info
    parser.add_argument("--checkpoint_samples_num", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints")
    
    args = parser.parse_args() 
    
    # for reproducibility
    torch.manual_seed(0)
    
    # set up distributed training
    args.host_rank, args.world_size = 0,1 
    args.is_distributed = args.backend is not None and len(args.hosts)>1
    if args.is_distributed:
        args.world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        args.host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=args.host_rank, world_size=args.world_size)

    # sentencepiece 
    sp_model_path = os.path.join(args.data_dir,args.sp_model_file)
    sp_model = sp.SentencePieceProcessor(sp_model_path)
    args.pad_id = sp_model.PieceToId('<pad>')
    args.bos_id = sp_model.PieceToId('<s>')
    args.eos_id = sp_model.PieceToId('</s>')

    # model definition    
    args.d_word_vec = args.d_model
    VOCAB_SIZE = sp_model.vocab_size()
    device = torch.device('cpu')
    model = Transformer(
                VOCAB_SIZE,
                VOCAB_SIZE,
                src_pad_idx=args.pad_id,
                trg_pad_idx=args.pad_id,
                trg_emb_prj_weight_sharing=args.proj_share_weight,
                emb_src_trg_weight_sharing=args.embs_share_weight,
                d_k=args.d_k,
                d_v=args.d_v,
                d_model=args.d_model,
                d_word_vec=args.d_word_vec,
                d_inner=args.d_inner_hid,
                n_layers=args.n_layers,
                n_head=args.n_head,
                dropout=args.dropout,
                n_position=args.max_seq_len+2,
                scale_emb_or_prj=args.scale_emb_or_prj).to(device)
    
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.lr_mul, args.d_model, args.n_warmup_steps)            
    
    # setting up data iterators
    
    train_files_prefix = os.path.join(args.data_dir,args.train_files_prefix)
    val_files_prefix = os.path.join(args.data_dir,args.val_files_prefix)

    train_src_file,train_trg_file = [train_files_prefix+ext for ext in [".src",".trg"]]
    training_data_chunks = data_chunk_iter(train_src_file,train_trg_file,sp_model,args.max_seq_len,args.load_amount,args.src_word_dropout)

    val_src_file,val_trg_file = [val_files_prefix+ext for ext in [".src",".trg"]]
    validation_data_chunks = data_chunk_iter(val_src_file,val_trg_file,sp_model,args.max_seq_len,args.load_amount,args.src_word_dropout)

    train(model, training_data_chunks, validation_data_chunks, optimizer, device, args)

if __name__ == '__main__':
    main()

