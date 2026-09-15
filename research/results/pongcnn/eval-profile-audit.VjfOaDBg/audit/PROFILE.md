# Native timing and profiler coverage

Native values below are the last logging interval divided by its recorded rollout count. One rollout is 2,048 decisions. These are coarse local intervals, not sums of the downsampled history. Training model includes forward/backward, loss, updates and optimizer work.

| Cell | Rollouts in interval | Rollout ms | Inference ms | Environment ms | Copy ms | Train model ms | Train misc ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| a-flex_quality | 63 | 2.9577 | 1.2830 | 0.3533 | 1.1111 | 1.1152 | 0.0150 |
| a-nature_cnn | 63 | 3.2684 | 1.5428 | 0.3358 | 1.1994 | 1.1702 | 0.0151 |
| a-impala_cnn | 3 | 15.3960 | 12.4259 | 0.3857 | 2.3617 | 48.9844 | 0.0198 |
| a-impoola_cnn | 3 | 14.3032 | 12.3116 | 0.3422 | 1.4708 | 47.4997 | 0.0174 |
| b-flex_quality | 63 | 3.0060 | 1.2843 | 0.3365 | 1.1429 | 3.4236 | 0.0152 |
| b-nature_cnn | 63 | 3.3926 | 1.5443 | 0.3378 | 1.2482 | 3.6250 | 0.0153 |
| b-impala_cnn | 3 | 15.0848 | 12.3801 | 0.3394 | 2.1864 | 148.6936 | 0.0202 |
| b-impoola_cnn | 3 | 14.7241 | 12.3120 | 0.3224 | 1.8777 | 146.6671 | 0.0179 |

## CUDA trace

Times sum traced kernel durations across streams; they are not elapsed wall time or GPU utilization. Matrix-multiply kernels are shared across convolution forward/input-gradient/weight-gradient, projection/core and optimizer, so those callers remain unresolved. Named kernels and all CUDA API summaries are preserved in analysis.json.

| Cell | Kernel executions | Kernel seconds | Shared GEMM % | Patch % | Bias gradient % | Input gather % | Pool % | Capture API ms | After-capture kernel seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a-flex_quality | 53504 | 0.1494 | 65.15 | 10.48 | 6.06 | 0.94 | 0.00 | 4.788 | 0.1482 |
| a-nature_cnn | 65088 | 0.1692 | 58.99 | 14.57 | 5.37 | 3.08 | 0.00 | 4.702 | 0.1678 |
| a-impala_cnn | 191872 | 3.9751 | 62.61 | 21.77 | 5.41 | 4.30 | 1.81 | 10.905 | 3.9630 |
| a-impoola_cnn | 191808 | 3.9312 | 61.36 | 22.35 | 5.69 | 4.57 | 1.83 | 12.205 | 3.9192 |
| b-flex_quality | 77056 | 0.2985 | 68.39 | 8.33 | 9.10 | 1.42 | 0.00 | 5.991 | 0.2973 |
| b-nature_cnn | 95424 | 0.3225 | 59.74 | 12.79 | 8.47 | 4.84 | 0.00 | 6.650 | 0.3211 |
| b-impala_cnn | 283264 | 10.4289 | 64.23 | 19.45 | 6.20 | 5.11 | 1.86 | 18.950 | 10.4168 |
| b-impoola_cnn | 283072 | 10.4020 | 63.58 | 19.92 | 6.36 | 5.14 | 1.87 | 20.533 | 10.3900 |

Capture API time sums cudaStreamBeginCapture/EndCapture, cudaGraphInstantiate and cudaGraphDestroy CPU calls; it does not measure the complete first-use capture region. After-capture excludes kernels starting before the final graph-instantiation return. First and later NVTX host-range durations are retained separately in analysis.json and do not equal asynchronous GPU execution time.

Nsight Systems 2025.5.2 captured CUDA node activity and NVTX under the existing base.profile CUDA profiler hooks. CPU sampling and context-switch tracing were disabled. No Nsight Compute hardware-counter replay was run. No occupancy, cache-bandwidth, per-layer GEMM attribution or isolated resource-query/logging CPU time is claimed. Five-second GPU/CPU samples may miss brief jobs and memory peaks. Instrumented full process time includes trace export overhead. Unprofiled controls remain separate.
