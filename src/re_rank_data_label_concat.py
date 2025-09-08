import argparse
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
sys.path.append("..")
from src import utils
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import warnings
warnings.filterwarnings("ignore")
debug_pth = "cross_day_graphs_debug"



def deal_data(args):
    save_data_pth = "../data/" + args.dataset + "/train_re_rank_data"
    # save_data_pth = "../data/EG/train_re_rank_data/get_score_test"
    # 从第三十天开始算起  构造数据
    gold_tensor_dict = {}
    get_score_dict = {}
    related_top100_dict = {}
    top_N_indices_all_dict = {}
    repeat_history_dict = {}
    
    for i in tqdm(range(args.split_flag)):
        # (all_history_back_cont_embed, train_sample_num, num_rels, find_cross_day, top_n):
        print("---------------------", i, "---------------------")
        with open(save_data_pth + '/related_top100_' + str(i+1) + '.pkl', 'rb') as f:
            related_top100 = pickle.load(f)  # 以天为单位的图
            related_top100_dict = {**related_top100_dict, **related_top100}
        
        with open(save_data_pth + '/gold_tensor_' + str(i+1) + '.pkl', 'rb') as f:
            gold_tensor = pickle.load(f)  # 以天为单位的图
            gold_tensor_dict = {**gold_tensor_dict, **gold_tensor}
            

        with open(save_data_pth + '/get_score_' + str(i+1) + '.pkl', 'rb') as f:
            get_score = pickle.load(f)  # 以天为单位的图
            get_score_dict = {**get_score_dict, **get_score}

        with open(save_data_pth + '/top_N_indices_' + str(i+1) + '.pkl', 'rb') as f:
            top_N_indices = pickle.load(f)  # 以天为单位的图
            top_N_indices_all_dict = {**top_N_indices_all_dict, **top_N_indices}
            
        with open(save_data_pth + '/repeat_history_' + str(i+1) + '.pkl', 'rb') as f:
            repeat_history = pickle.load(f)  # 以天为单位的图
            repeat_history_dict = {**repeat_history_dict, **repeat_history}


    with open(save_data_pth + '/related_top100_all.pkl', 'wb') as file:
        pickle.dump(related_top100_dict, file)
    with open(save_data_pth + '/gold_tensor_all.pkl', 'wb') as file:
        pickle.dump(gold_tensor_dict, file) 
    with open(save_data_pth + '/get_score_all.pkl', 'wb') as file:
        pickle.dump(get_score_dict, file)   
    with open(save_data_pth + '/top_N_indices_all.pkl', 'wb') as file:
        pickle.dump(top_N_indices_all_dict, file)   
    with open(save_data_pth + '/repeat_history_all.pkl', 'wb') as file:
        pickle.dump(repeat_history_dict, file)   
    print("finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='re_rank_data_label')
    parser.add_argument("-d", "--dataset", type=str, default='EG',
                        help="which country's dataset to use: EG/ IR/ IS")

    parser.add_argument("--split_flag", type=int, default=7,
                        help="which country's dataset to use: EG/ IR/ IS")
    args = parser.parse_args()
    print(args)

    deal_data(args)
    sys.exit()
    

    # python re_rank_data_label_concat.py --dataset EG --split_flag 7