# ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations

This repository is the official implementation of our paper ["ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations"](http://arxiv.org/abs/2506.02818) by Ekaterina Grishina, Mikhail Gorbunov and Maxim Rakhuba.

OPT and Llama2 HuggingFace models are supported.

## Installation

Clone and navigate to the repository
```bash
git clone https://github.com/GrishKate/ProcrustesGPT.git
```
Install requirements.txt

```bash
pip install -r requirements.txt
```

## How To Use
Fill the configs for compression of the weight matrices. For examples of configs, please, see ```/configs``` folder. Provide ```tmp_path``` folder to save orthogonal matrices. 

1. Firstly, compress the model in Frobenius norm:
```bash
python run_procrustes_gpt.py --model_name 'facebook/opt-125m'\ # 'facebook/opt-...' and 'meta-llama/Llama-2-...-hf' are supported 
                             --model_path '/path/to/model' \ # optionally if model is stored locally
                             --cfg_for_compression_path './configs/compression_frobenius.yaml' \ # path to config
                             --cfg_for_layers_path './configs/k_layers_opt_125m.yaml' # path to config with specified sizes of decompositions
                             --skip_connections 'cayley' \ # optionally compress skip connections ('cayley' or 'exponent')
                             --save True \ # save the resulting model
                             --save_path 'path/to/save/model' \ # where to save
                             --filename 'opt_125m_compressed.pt' \ # filename to save
```
2. Secondly, change the compression config and run compression in the weighted norm:
```bash
python run_procrustes_gpt.py --model_name 'facebook/opt-125m'\
                             --model_path '/path/to/model' \ # optionally if model is stored locally
                             --cfg_for_compression_path './configs/compression_weighted.yaml' \
                             --cfg_for_layers_path './configs/k_layers_opt_125m.yaml'
                             --skip_connections 'cayley' \ # compress skip connections ('cayley' or 'exponent')
                             --save True \ # save the resulting model or not
                             --save_path 'path/to/save/model' \ # where to save
                             --filename 'opt_125m_compressed.pt' \ # filename to save
```

3. To evaluate the perplexity:

```bash
python run_lm_eval.py --model 'facebook/opt-125m' \
                      --tokenizer_path 'facebook/opt-125m' \ # optionally provide path to tokenizer, if saved locally
                      --weights_path 'path/to/save/model/opt_125m_compressed.pt'\ # path to saved compressed model
                      --no-wandb
```

4. To evaluate the zero-shot performance:

```bash
python run_ppl_eval.py --model_name 'facebook/opt-125m'\
                       --tokenizer_path 'facebook/opt-125m' \ # optionally provide 
                       --weights_path '/kaggle/working/opt_125m_compressed.pt' # path to saved compressed model
```

## Credits

This code is based on [SliceGPT](https://github.com/microsoft/TransformerCompression.git) repository.

