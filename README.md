ARGNet
======
A deep nueral network for robust identification and annotation of antibiotic resistance genes.

The input can be long amino acid sequences(full length/contigs), long nucleotide sequences, 
short amino acid reads (30-50aa), short nucleotide reads (100-150nt) in fasta format.
If your input is short reads you should assign 'argnet-s' model, or if your input is full-length/contigs
you should assign 'argnet-l' to make the predict. </br>

![alt text](https://github.com/patience111/ARGNet/blob/main/pics/ARGNet_workflow.png)</br>


Installation
------------
clone the program to your local machine:</br>
git clone https://github.com/patience111/ARGNet

**1. Setting up environment**</br></br>
   **1.1 Installation with conda**</br>
   
   1.1.1 For **CPU** inference, you could install the program with conda YAML file in the installation directory with the following commands:</br>
       cd ./installation </br>
       conda env create -f ARGNet-CPU.yml -n ARGNet-cpu </br>
       conda activate ARGNet-cpu </br>
   
   (This was tested on Ubuntu 16.04)</br>
   ![alt text](https://github.com/patience111/ARGNet/blob/main/pics/argnet_conda_cpu_trial.png)</br>
   
   
   1.1.2 For **GPU** inference, you could install the program with conda YAML file in the installation directory with the following commands:</br>
        cd ./installation</br>
        conda env create -f ARGNet-GPU.yml -n ARGNet-gpu</br>
        conda activate ARGNet-gpu</br>
      
    (This was tested on Ubuntu 16.04, cuda 10.1, Driver Version: 430.64)</br>
    ![alt text](https://github.com/patience111/ARGNet/blob/main/pics/argnet_conda_gpu_trial.png)</br>

   **1.2 Or, if you prefer installing dependencies manually**, you might find this information useful:</br>
      The program was tested with the following package version, you can install exactly the same version or other compatible versions.</br>

      Biopython:  1.79 </br>
      tensorflow:  2.2.0 </br> 
      cuda: 10.2 (for GPU using)</br> 
      cudnn: 7.6.5.32 (for GPU using)</br>
      numpy: 1.18.5</br>
      scikit-learn: 0.24.1</br>
      tqdm: 4.56.0</br>

**2. Getting trained models**<br>
   
     cd ./model </br>
     bash get-models.sh

Quickstart Guide
----------------
  ***for full-length or contigs***</br>
      **python**&nbsp;&nbsp; argnet.py **--input**&nbsp;&nbsp;input_path_data&nbsp;&nbsp;**--type**&nbsp;&nbsp; aa/nt&nbsp;&nbsp;**--model**&nbsp;&nbsp; argnet-l&nbsp;&nbsp;  **--outname**&nbsp;&nbsp; output_file_name </br></br>
  ***for short reads***</br>
      **python**&nbsp;&nbsp;argnet.py **--input**&nbsp;&nbsp;input_path_data&nbsp;&nbsp;**--type**&nbsp;&nbsp; aa/nt&nbsp;&nbsp;**--model**&nbsp;&nbsp; argnet-s&nbsp;&nbsp;  **--outname**&nbsp;&nbsp; output_file_name </br>
    
**general options:**</br>
     --input/-i&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the test file as input </br>
     --type/-t &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;molecular type of your test data (aa for amino acid, nt for nucleotide)</br>
     --model/-m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the model you assign to make the prediction (argnet-l for long sequences, argnet-s for short reads) </br>
     --outname/-on&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the output file name </br>

**optional arguments:**</br>
  -h, --help            show this help message and exit</br></br>
  ![alt text](https://github.com/patience111/ARGNet/blob/main/pics/ARGNet_help.jpeg)</br>
  -i INPUT, --input INPUT </br>
                        the test data as input </br></br>
  -t {aa,nt}, --type {aa,nt} </br>
                        molecular type of your input file </br></br>
  -m {argnet-s,argnet-l}, --model {argnet-s,argnet-l} </br>
                        the model to make the prediction </br></br>
  -on OUTNAME, --outname OUTNAME </br>
                        the name of results output </br></br>
  
 
Example
----------
if we predict the long amino acid contigs by using ARGNet-L model, we could use command line (if you are in ARGNet dirctory):</br></br>
**python3**&nbsp;&nbsp;./scripts/argnet.py&nbsp;&nbsp;**-i**&nbsp;&nbsp;./tests/aa/long/arg100p.fasta&nbsp;&nbsp;**-t**&nbsp;&nbsp;aa&nbsp;&nbsp;-m&nbsp;&nbsp;argnet-l&nbsp;&nbsp;**-on**&nbsp;&nbsp;argnet_lsaa_test.txt </br>

**output** will be like: </br>
![alt text](https://github.com/patience111/ARGNet/blob/main/pics/lsaa_prediction.png)</br>
the first column **test_id** is the sequence label of the test sequnece.</br>
the second column **ARG_prediction** is the "ARG" or "non-ARG" prediction of the input sequence.</br>
the third column **resistance_category** is the classifition of the 36 antibiotics categories of the input sequence resisting to.</br>
the last column **probability** is the classifition probability of the antibiotic predition of the input sequence by the model.

Contribute
----------

If you'd like to contribute to ARGNet, check out https://github.com/patience111/argnet</br>
Hope you enjoy ARGNet journey, any problem please contact scpeiyao@gmail.com
