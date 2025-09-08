# ITHI
This is the code and data for History does Not Simply Repetition: Integrating Three Types of Historical Information for Context-aware Event Forecasting

## Environment

The code is tested to be runnable under the environment with
python=3.10.4; pytorch=2.1.0; cuda=12.2; dgl=2.3.0+cu121 ...



## Dataset

The Datasets files need to be unzipped first:
```
unzip data.zip
unzip data_disentangled.zip
```

The folder `./data` contains the original event data and the structured data.  



where d_name is `EG` or `IR` or `IS`.
## Structured Data Generation
    ```
	cd get_history

	python get_back_embed.py  --c d_name
	python get_graph_current.py  --datapath  ../data/d_name
	python get_history_dict.py  
  	
	
    ```

## Model Training Commands：

 	python ITHI.py --dataset d_name

## Model Test Command：
 python test.py --dataset d_name
  