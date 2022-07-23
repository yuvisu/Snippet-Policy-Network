# Snippet Policy Network

This repository is the official implementation of [Snippet Policy Network for Multi-class Varied-length ECG Early Classification](https://ieeexplore.ieee.org/document/9741533), published at IEEE Transactions on Knowledge and Data Engineering. Feel free to contact me via this email (yuvisu.cs04g@nctu.edu.tw) if you get any problems.

### Guideline:

1. Clone this repository.
2. Run get_data.sh
3. Run the main program to train a model.
4. Use the pipeline_inference to evaluate the model.

We provide a well-trained model (the model of the first fold in 10-fold cross-validation) to validate the performance in our paper.

### If you find this code helpful, feel free to cite our paper:
```
@ARTICLE{Huang2022SPN,
  author={Huang, Yu and Yen, Gary G and Tseng, Vincent S.},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Snippet Policy Network for Multi-class Varied-length ECG Early Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2022.3160706}}
```