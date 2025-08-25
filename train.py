import os
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import networkx as nx

import numpy as np
import re
from tqdm import tqdm
from retrieve import GraphRetrieval


def main(args):

    seed_everything(seed=args.seed)
    print(args)
    #set training set
    train_data_path: str = f"dataset/ML1M/10000_data_id_20.json"
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    retrieval_model=GraphRetrieval(model_name='sbert', path='dataset/fb')



    gold_train = []
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)[:9000]#Take the first 9000 as training set as an example
        inputs_train = [_['input'] for _ in train_data]
        questions_train = [_['questions'] for _ in train_data]
        gold_train = [0 if _['output']=="A" else 1 if _['output']=="B" else 2 if _['output']=="C" else 3 if _['output']=="D" else 4 if _['output']=="E" else 5 if _['output']=="F" else 6 if _['output']=="G" else 7 if _['output']=="H" else 8 if _['output']=="I" else 9 if _['output']=="J" else 10 if _['output']=="K" else 11 if _['output']=="L" else 12 if _['output']=="M" else 13 if _['output']=="N" else 14 if _['output']=="O" else 15 if _['output']=="P" else 16 if _['output']=="Q" else 17 if _['output']=="R" else 18 if _['output']=="S" else 19 for _ in train_data]
        sequence_ids_train = [json.loads(_.get('sequence_ids', '[]')) for _ in train_data]
        target_all = [_['output'] for _ in train_data]

        def train(inputs, questions=None, gold=None, targets=None, sequence_ids=None):
            id=[]
            query=[]
            graph=[]
            label=[]
            sample = {'id':id,'graph':graph, 'question':query, 'label':label}
            d = 0
            for input, question, target, sequence_id in zip(inputs, questions, targets, sequence_ids):
                d = d+1
                id.append(f'query{d}')
                      
                query.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  
                            ### Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only".""")                    
                label.append(target)
                retrieve_movies_list=[]
                retrieve_movies_list = retrieval_model.whether_retrieval(sequence_id,args.adaptive_ratio*len(sequence_id))
                graph.append(retrieval_model.retrieval_topk(input, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers))
            loss = model.forward(sample)
            return loss
        
    
        def batch(list, batch_size=args.batch_size):
                chunk_size = (len(list) - 1) // batch_size + 1
                for i in range(chunk_size):
                    yield list[batch_size * i: batch_size * (i + 1)]
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
        )
        model.train()
        for epoch in range(args.num_epochs):
            for i, batch_prompt in tqdm(enumerate(zip(batch(inputs_train), batch(questions_train), batch(gold_train), batch(target_all), batch(sequence_ids_train)))):
                inputs, questions, golds, targets, sequence_ids = batch_prompt
                optimizer.zero_grad()
                loss = train(inputs, questions, golds, targets, sequence_ids)
                loss.backward()
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                optimizer.step()
                print(f'{i}-th LOSS:', loss.item())
                print("Epoch %s: Training Process is %s/9000" % (epoch, i*5))
            print("Epoch %s is finished"%(epoch))
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            print("Save checkpoint")


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)