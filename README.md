# Optimized-Product-Quantization-and-KNN-Search-based-on-KNN-Graph

An implementation of 'K Nearest Neighbors Search based on KNN-Graph'<br/>
and 'Optimized Product Quantization'

# Tools
>Eigen 3<br/>
>Intel MKL<br/>
>Opencv 3.4.5<br/>

# Datasets

  SIFT-1M & GIST-1M : 
  >[download](http://corpus-texmex.irisa.fr).<br/>
  >Then put all '.fvecs' and '.ivecs' files under ['./dataset']().<br/>
  
  Self-made binary files for KNN Search with OPQ : 
  >[download](https://pan.baidu.com/s/1EeZ1uQQ8P7j1n9Y_agqg_A) with key `6raq`.<br/>
  >Then put them all under ['./dataset/256bits_100iter_8bits']()<br/>
  >Also, you could choose to encode you results (codebook, centroids, lookuptable, matrix R) of Optimized Product Quantization into such kind of files using [this script]()

# K Nearest Neighbors Search based on KNN-Graph

## How to run it
details are provided in `main.cpp`<br/>

## Performance
QPS-Precision on
[SIFT-1M and GIST-1M with 100NN-Graph] (https://github.com/guodongxiaren/ImageCache/raw/master/Logo/foryou.gif)
<br/>

QPS-Precision on[SIFT-1M with XNN-Graph]()<br/>
and [its memory usage]()<br/>

# Optimized Product Quantization
## How to run it
details are provided in `main.cpp`<br/>

## Performance
distortion with different code length on SIFT-1M
[]()

