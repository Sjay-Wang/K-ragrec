# K-RagRec

**This is the pytorch implementation for our ACL paper "Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation".**

## Environment

- Python==3.9
- numpy==1.23.4
- torch==2.4.1
- cuda==11.8.89
- transformers==4.45.2
- networkx==2.8.7
- peft==0.12.0

## Dataset

We provide three datasets: MovieLens-1M, MovieLens-20M and Amazon Book.

Due to space limitations, we only processed MovieLens-1M and put the KG and datasets at https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view?usp=sharing. Other datasets we provided raw data that can be processed as in the paper.

## KG

We process the knowledge graph and provide the corresponding knowledge vector databases.

## An example to run K-RagRec on MovieLens-1M

To run K-RagRec on MovieLens-1M with threshold $p=50%$, retrieve knowledge sub-graphs numbers  $k=3$, and re-ranking knowledge sub-graphs numbers $K=3$, respectively.

**For Training:**

```
python train.py --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5 
```

**For evaluation:**

```
python evaluate.py  --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5 
```

or you can run:

```
bash run.sh
```

## **Extend:**

- If you want to run K-RagRec on other datasets, you should first process the datasets, and change the datasets name. We provide 3 datasets in our environment.
- Change the **llm_model_name** (e.g., 8b) to adopt other llm model.
- Our experiments are performed on two NVIDIA A6000-48G GPUs. If you have only one GPU, please use *graph_llm_for_one_GPU*.



### 🌹Please Cite Our Work If Helpful:



**Thanks! / 谢谢! / ありがとう! / merci! / 감사! / Danke! / спасибо! / gracias! ...**



```
@article{wang2025knowledge,
            title={Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation},
            author={Wang, Shijie and Fan, Wenqi and Feng, Yue and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei},
            journal={arXiv preprint arXiv:2501.02226},
            year={2025}
          }
```

