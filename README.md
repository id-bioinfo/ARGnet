ARGNet
======
a deep nueral network for robust identification and annotation of antibiotic resistance genes.

The input can be long amino acid sequences(full length/contigs), long nucleotide sequences, 
short amino acid reads (30-50aa), short nucleotide reads (100-150nt) in fasta format.
If your input is short reads you should assign 'argnet-s' model, or if your input is full-length/contigs
you should assign 'argnet-l' to make the predict.

Installation
------------

  To install with pip, run:

      git clone https://github.com/patience111/ARGNet

Quickstart Guide
----------------
  for full-length or contigs</br>
      python argnet.py --input input_path_data --type aa/nt --model argnet-l  --outname output_file_name </br>
  for short reads</br>
      python argnet.py --input input_path_data --type aa/nt --model argnet-s  --outname output_file_name </br>
    
**general options:**</br>
     --input/-i		the test file as input </br>
     --type/-t		molecular type of your test data (aa for amino acid, nt for nucleotide)</br>
     --model/-m		the model you assign to make the prediction (argnet-l for long sequences, argnet-s for short reads) </br>
     --outname/-on	the output file name </br>

**optional arguments:**</br>
  -h, --help            show this help message and exit</br>
  -i INPUT, --input INPUT </br>
                        the test data as input </br>
  -t {aa,nt}, --type {aa,nt} </br>
                        molecular type of your input file </br>
  -m {argnet-s,argnet-l}, --model {argnet-s,argnet-l} </br>
                        the model to make the prediction </br>
  -on OUTNAME, --outname OUTNAME </br>
                        the name of results output </br>

Hope you enjoy ARGNet journey, any problem please contact scpeiyao@gmail.com </br>

Contribute
----------

If you'd like to contribute to ARGNet, check out https://github.com/patience111/argnet
