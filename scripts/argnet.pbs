#!/bin/sh
#PBS -N argnet_ssnt_test
#PBS -q gpuq1
#PBS -l walltime=120:00:00
#PBS -l select=1:ncpus=12:ngpus=1:mem=375gb:host=hpc-gn004
##Request 1 node 8 core,4 GPU and 64gb ram
#PBS -o argnet_MAY19IFF.out
#PBS -e argnet_MAY19IFF.err 
#PBS -V
# User Directives
#module load python3 cuda11.0/toolkit/11.0.3 cudnn8.0-cuda11.0/8.0.5.39
module load tensorflow2-py37-cuda10.2-gcc/2.2.0
#module load python37_tf22
#python ~/wlm/PBS_tf/tensorflow_test.py
JID=`echo ${PBS_JOBID}| sed "s/.hpc25-mgt.hku.hk//"`
echo Job ID : ${JID}
echo ${NPROCS} CPUs allocated: `cat $PBS_NODEFILE` 1>&2
echo This PBS script is running on host `hostname` 1>&2
echo Working directory is $PBS_O_WORKDIR  1>&2
echo ============== ${PBS_JOBNAME} : ${NPROCS} CPUs ====================
echo "Job Start  Time is `date "+%Y/%m/%d -- %H:%M:%S"`"
export PYTHONPATH=$PYTHONPATH:/home/d24h_prog2/.conda/envs/argnet/lib/python3.7/site-packages/
echo $PYTHONPATH
/usr/bin/time -o /home/d24h_prog2/pypy/ARG_MGE_platform/outputs/argnet_ssnt_MAY19IFF.txt -f "%E %M" python3 /home/d24h_prog2/pypy/ARG_MGE_platform/ARGNet/scripts/argnet.py -i /home/d24h_prog2/pypy/ARG_MGE_platform/data/reads/MAY19IFF_short_all_trimmed.fas_renamed -t nt -m argnet-s -on argnet_ssnt_MAY19IFF.txt
#End of script

 echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"
