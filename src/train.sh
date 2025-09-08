
# 修改 bash1.sh
#!/bin/bash
set -x

d="$1"
context="$2"
k_contexts="$3"
n_layers="$4"
train_history_len="$5"
train_lr="$6"




echo "Received d: $d"
echo "Received context: $context"
echo "Received k_contexts: $k_contexts"
echo "Received n_layers: $n_layers"
echo "Received train_history_len: $train_history_len"
echo "Received train_lr: $train_lr"



#  ./train.sh > 3.28_K_7_train.log 2>&1
# export CUDA_VISIBLE_DEVICES=1
python main.py \
    -d $d \
    --context $context \
    --k_contexts $k_contexts \
    --hypergraph_ent \
    --n_layers_hypergraph_ent 1 \
    --hypergraph_rel \
    --n_layers_hypergraph_rel 1 \
    --score_aggregation hard \
    --encoder rgcn \
    --n_layers $n_layers \
    --n_hidden 200 \
    --self_loop \
    --layer_norm \
    --train_history_len $train_history_len \
    --lr $train_lr \
    --wd 1e-6 \
