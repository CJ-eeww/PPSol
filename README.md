# PPSol
Enhancing Protein Solubility Prediction through Pre-trained Language Models and Graph Convolutional Neural Networks

## 1. Dependencies
- torch==1.8.0+cu111
- scikit-learn==0.22
- tqdm==4.26.0
- fair-esm==2.0.0
- python=3.7.2
- torch-geometric==2.0.0
- torch-cluster==1.5.9
- torch-scatter==2.0.8
- torch-sparse==0.6.9

## 2. Software and database 
To run the full & accurate version of PPSol, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[SPIDER3](https://sparks-lab.org/server/spider3/)

PHY and BLOSUM62 is in ./Data

## 3. Run PPSol for prediction  

Run the following python script and it will take about 1 hour to train the model.

```
$ python predict.py
```



## 4. Dataset and model  
We provide the datasets, the pre-predicted structures, and the pre-trained models here for those interested in reproducing our paper.  
The datasets used in this study (eSol, scerevisiae) are stored in ./Data/ in csv format.  
Everything else can be found here(https://drive.google.com/drive/folders/1hMJ0p-0ZdNWNP8utBiJwFuVf5SRy9C37?usp=sharing)
