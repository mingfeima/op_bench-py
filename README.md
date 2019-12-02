# op_bench-py
performance benchmark for pytorch operators


### usage
* benchmarking

```bash
the benchmark will run with TWO modes by defaults:
#1: use single socket
#2: use single core
./run.sh [xxx.py]
```

* profile via vtune

```bash
#the record will be generated under folers like r000ah
./profile.sh [xxx.py]
```
* archive

```bash
#you may collect on linux and review the record on windows
#but have to archive the record on linux first
./archive.sh [r000ah]
```
