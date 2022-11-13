import argparse
import textwrap
#import argnet_lsaa as lsaa
#import argnet_lsnt as lsnt
#import argnet_ssaa as ssaa
#import argnet_ssnt as ssnt
import sys
parser = argparse.ArgumentParser(
prog='ARGNet',
formatter_class=argparse.RawDescriptionHelpFormatter,
description=textwrap.dedent("""\
    ARGNet: a deep nueral network for robust identification and annotation of antibiotic resistance genes.
   --------------------------------------------------------------------------------------------------------
    The standlone program is at https:...
    The online service is at https:...
    
    The input can be long amino acid sequences(full length/contigs), long nucleotide sequences, 
    short amino acid reads (30-50aa), short nucleotide reads (100-150nt) in fasta format.
    If your input is short reads you should assign 'argnet-s' model, or if your input is full-length/contigs
    you should assign 'argnet-l' to make the predict.
    
    USAGE:
        for full-length or contigs
            python argnet.py --input input_path_data --type aa/nt --model argnet-l  --outname output_file_name
        for short reads
            python argnet.py --input input_path_data --type aa/nt --model argnet-s  --outname output_file_name
    
    general options:
        --input/-i    the test file as input
        --type/-t     molecular type of your test data (aa for amino acid, nt for nucleotide)
        --model/-m    the model you assign to make the prediction (argnet-l for long sequences, argnet-s for short reads) 
        --outname/-on  the output file name
    """

),
epilog='Hope you enjoy ARGNet journey, any problem please contact scpeiyao@gmail.com')

parser.print_help()
#parser.parse_args()
parser.add_argument('-i', '--input', required=True, help='the test data as input')
parser.add_argument('-t', '--type', required=True, choices=['aa', 'nt'], help='molecular type of your input file')
parser.add_argument('-m', '--model', required=True, choices=['argnet-s', 'argnet-l'], help='the model to make the prediction')
parser.add_argument('-on', '--outname', required=True, help='the name of results output')

args = parser.parse_args()


## for AESS_aa -> classifier
if args.type == 'aa' and args.model == 'argnet-s':
    import argnet_ssaa_chunk as ssaa
    ssaa.argnet_ssaa(args.input, args.outname)

# for AESS_nt -> classifier
if args.type == 'nt' and args.model == 'argnet-s':
    import  argnet_ssnt_new as ssnt
    ssnt.argnet_ssnt(args.input, args.outname)

# for AELS_aa -> classifier
if args.type == 'aa' and args.model == 'argnet-l':
    import argnet_lsaa_speed_sgpu as lsaa
    lsaa.argnet_lsaa(args.input, args.outname)

# for AELS_nt -> classifier
if args.type == 'nt' and args.model == 'argnet-l':
    import argnet_lsnt as lsnt
    lsnt.argnet_lsnt(args.input, args.outname)


