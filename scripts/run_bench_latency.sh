ipython benchmarks/bench_latency.py -- \
    --backends cpu_pytorch cuda_pytorch cuda_kernel \
    --K 1024 4096 16384 --H 40 80 \
    --warmup 5 --measure 30
