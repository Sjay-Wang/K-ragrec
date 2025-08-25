# # run training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5
echo "Training completed."

# run evaluation
echo "Starting evaluation..."
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py  --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5
echo "Evaluation completed."
