# Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph

An implementation of `K Nearest Neighbors Search based on KNN-Graph` and `Optimized Product Quantization`

## Tools
>Eigen 3<br/>
>Intel MKL<br/>
>Opencv 3.4.5<br/>

## Datasets

  SIFT-1M & GIST-1M : 
  >[Download](http://corpus-texmex.irisa.fr)<br/>
  >Then put all `.fvecs` and `.ivecs` files under `'./dataset'`.<br/>
  
  Self-made binary files for KNN Search with OPQ : 
  >[Download](https://pan.baidu.com/s/1EeZ1uQQ8P7j1n9Y_agqg_A) from BaiduPan with key `6raq`<br/>
  >Then put them all under `'./dataset/256bits_100iter_8bits'`<br/>
  >Also, you could choose to encode you results (codebook, centroids, lookuptable, matrix R) of Optimized Product Quantization into such kind of files using `encoder.m`

# K Nearest Neighbors Search based on KNN-Graph

## How to run it
details are provided in the demo `main.cpp`<br/>

## Performance
QPS-Precision on SIFT-1M and GIST-1M with 100NN-Graph<br/>
![QPS-Precision on SIFT-1M and GIST-1M with 100NN-Graph](https://github.com/QinghuiXing/Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph/figure/qps-precision_no_outliers.png)
<br/>

QPS-Precision on SIFT-1M with XNN-Graph<br/>
![QPS-Precision on SIFT-1M with XNN-Graph](https://github.com/QinghuiXing/Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph/figure/qps-precision_XNNGraph_v2.png)
<br/>
and its memory usage<br/>
![memory usage](https://github.com/QinghuiXing/Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph/figure/memory_XNNGraph.png)
<br/>

# Optimized Product Quantization
## How to run it
Details are provided in the demo `main.cpp`<br/>

## Performance
Distortion with different code length on SIFT-1M<br/>
![Distortion with different code length on SIFT-1M](https://github.com/QinghuiXing/Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph/figure/distortion-codelength_mycode.png)

