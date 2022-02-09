ARGNet
======
a deep nueral network for robust identification and annotation of antibiotic resistance genes.

The input can be long amino acid sequences(full length/contigs), long nucleotide sequences, 
short amino acid reads (30-50aa), short nucleotide reads (100-150nt) in fasta format.
If your input is short reads you should assign 'argnet-s' model, or if your input is full-length/contigs
you should assign 'argnet-l' to make the predict.

Installation
------------

  To install with git, run:

      git clone https://github.com/patience111/ARGNet

**Requirements:**
---------------

Biopython:  1.79
tensorflow:  2.2.0, cuda10.2, cudnn7.6.5.32
numpy:  1.18.5
sklearn:  0.24.1
tqdm:  4.56.0

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
  -i INPUT, --input INPUT </br>
                        the test data as input </br></br>
  -t {aa,nt}, --type {aa,nt} </br>
                        molecular type of your input file </br></br>
  -m {argnet-s,argnet-l}, --model {argnet-s,argnet-l} </br>
                        the model to make the prediction </br></br>
  -on OUTNAME, --outname OUTNAME </br>
                        the name of results output </br></br>

Hope you enjoy ARGNet journey, any problem please contact scpeiyao@gmail.com </br>

Contribute
----------

If you'd like to contribute to ARGNet, check out https://github.com/patience111/argnet
