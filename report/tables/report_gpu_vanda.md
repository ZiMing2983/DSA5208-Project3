| configuration | 2D 8192² | 2D 16384² | 3D 256³ | 3D 512³ |
|---|---|---|---|---|
| 1 CPU core (serial) | 270.7 | 1088.3 | 71.3 | 663.5 |
| 1 CPU node (p = 72) | 15.6 | 64.0 | 3.64 | 34.7 |
| 1 A40, CUDA-aware | 14.8 | 59.3 | 3.58 | 31.4 |
| 2 A40 in 1 node, CUDA-aware | 7.40 | 29.3 | 2.41 | 17.4 |
| 2 A40 in 1 node, host-staged | 7.36 | 29.2 | 1.97 | 15.7 |
| 2 A40 on 2 nodes, CUDA-aware | 7.48 | 29.6 | 2.23 | 16.5 |
| 2 A40 on 2 nodes, host-staged | 7.40 | 29.5 | 2.26 | 16.5 |
