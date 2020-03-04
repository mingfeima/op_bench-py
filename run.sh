### run script for the operator benchmark

#source activate pytorch-mingfei
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so


if [ $# -lt 1 ]; then
  echo "usage: ./run.sh [xxx.py] for inference"
  echo "       ./run.sh [xxx.py] --train for training"
  exit
fi

INPUT_FILE=$1

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

if [ $# -ge 2 ]; then
  export $KMP_SETTING
  export KMP_BLOCKTIME=$KMP_BLOCKTIME

  echo -e "\n### using $KMP_SETTING"
  echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"
  echo -e "\n### using OMP_NUM_THREADS=$TOTAL_CORES"
  OMP_NUM_THREADS=$TOTAL_CORES $PREFIX python -u $INPUT_FILE --train
  exit
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

### single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=$CORES $PREFIX python -u $INPUT_FILE

### single thread test
echo -e "\n### using OMP_NUM_THREADS=1"
PREFIX="numactl --physcpubind=0 --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=1 $PREFIX python -u $INPUT_FILE
