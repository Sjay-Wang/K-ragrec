import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from sklearn.metrics import classification_report

import numpy as np
import re
from torch_geometric.data import Data
import networkx as nx
from retrieve import GraphRetrieval
import time

def main(args):

    seed_everything(seed=args.seed)
    print(args)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    try:
        model = _reload_best_model(model, args)
    except:
        pass
    model.eval()
    retrieval_model=GraphRetrieval(model_name='sbert', path='dataset/fb')
    test_data_path: str = "dataset/ML1M/10000_data_id_20.json"

    def recall_at_k(actual, predicted, k):
        index=0
        for x in predicted[:k]:
            index=index+1
            if x in actual:
                if index <=1:
                    return 1,1,1,1
                elif index <=3:
                    return 0,1,1,1
                elif index <=5:
                    return 0,0,1,1
                elif index <=10:
                    return 0,0,0,1
        return 0,0,0,0

    def evaluate(inputs, questions=None,gold=None, sequence_ids=None):
        id=[]
        query=[]
        graph=[]
        label=[]

        sample = {'id':id,'graph':graph, 'question':query, 'label':label}
        d = 0
        for input, question, sequence_id in zip(inputs, questions, sequence_ids):
            d = d+1
            #attack
            retrieve_movies_list=[]
            retrieve_movies_list = retrieval_model.whether_retrieval(args.adaptive_ratio*sequence_id, 5)
            graph.append(retrieval_model.retrieval_topk(input, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers))
            id.append(f'query{d}')
                
            query.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  
                        ## Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only".""")    
  
            label.append('')

        output = model.inference(sample).tolist()

        return output

    
    from tqdm import tqdm
    gold = []
    pred = []
    recalls_1 = []
    recalls_3 = []
    recalls_5 = []
    recalls_10 = []
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)[9000:10000]
        inputs = [_['input'] for _ in test_data]
        questions = [_['questions'] for _ in test_data]
        gold = [0 if _['output']=="A" else 1 if _['output']=="B" else 2 if _['output']=="C" else 3 if _['output']=="D" else 4 if _['output']=="E" else 5 if _['output']=="F" else 6 if _['output']=="G" else 7 if _['output']=="H" else 8 if _['output']=="I" else 9 if _['output']=="J" else 10 if _['output']=="K" else 11 if _['output']=="L" else 12 if _['output']=="M" else 13 if _['output']=="N" else 14 if _['output']=="O" else 15 if _['output']=="P" else 16 if _['output']=="Q" else 17 if _['output']=="R" else 18 if _['output']=="S" else 19 for _ in test_data]
        sequence_ids = [json.loads(_.get('sequence_ids', '[]')) for _ in test_data]
        def batch(list, batch_size=args.eval_batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(inputs), batch(questions), batch(gold), batch(sequence_ids)))):
            inputs, questions, golds, sequence_ids = batch
            output = evaluate(inputs, questions, golds, sequence_ids)
            pred.extend(output)
            start_index = len(pred) - args.eval_batch_size
            ground_truth=gold[start_index:start_index + args.eval_batch_size]
            for ind in range(args.eval_batch_size):
                recall_1,recall_3,recall_5,recall_10 = recall_at_k([ground_truth[ind]], output[ind], 10)
                recalls_1.append(recall_1)
                recalls_3.append(recall_3)
                recalls_5.append(recall_5)
                recalls_10.append(recall_10)
            print(f"Recall@1: {sum(recalls_1) / len(recalls_1)}, Recall@3: {sum(recalls_3) / len(recalls_3)}, Recall@5: {sum(recalls_5) / len(recalls_5)}, Recall@10: {sum(recalls_10) / len(recalls_10)}")

    # print(f'Final ACC: ', accuracy_score(gold, pred))
    
            
if __name__ == "__main__":
    args = parse_args_llama()

    main(args)