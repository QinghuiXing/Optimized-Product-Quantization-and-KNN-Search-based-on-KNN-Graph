// to activate Eigen library
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

// common head file
#include "pch.h"
#include <iostream>
#include<fstream>
#include<vector>
#include<chrono>

// for Eigen library
#include<Eigen/Dense>
#include<Eigen/SVD>

// for Memory Mapped Files
#include<immintrin.h>

// For self-construct class
#include"utils.h"
#include"base.h"
#include"opq.h"



using namespace aknnspace; // self-construct namespace
using namespace Eigen;


int main()
{
	/*
		@read before you run this demo:
		Below lies three demo of 'Naive AKNN Search', 'AKNN Search based on Optimized Product Quantization' and
			'Optimized Product Quantization' respectively. For some variables have the same name, run one demo once a time
			and comment the rest.
	*/


	/*------   Naive AKNN Search    ------*/
		/*
			@brief:
					- this is a sample script to do AKNN Search in naive way.
					- with AVX-256 bydefault, but you could slightly modity it to run without AVX-256
					- file path need to be specified.
					- Self-made binary file with extension like .u8code and .fcode could be find
					  in dataset file. And they could be made by Matlab in a naive way,
					  read 'utils' class function 'load_my_binary_file()' for more information
					  about the self-made binary file.
			@run:
					remove the comment and run it
		*/

		
	const char * file_query = "I:/AKNN/sift/sift/sift_query.fvecs";
	const char * file_knngraph = "I:/AKNN/sift_100NN_100.graph";
	const char * file_groundtruth = "I:/AKNN/sift/sift/sift_groundtruth.ivecs";
	const char * file_basedata = "I:/AKNN/sift/sift/sift_base.fvecs";

	unsigned sift1m_num_of_samples = 0;
	unsigned sift1m_dimension = 0; // 960 for gist1m
	unsigned sift1m_query_num = 0; // 1000 for gist1m

	unsigned K = 100; // required number of nearest neighbors
	unsigned L = 400; // capacity of candidate pool
	unsigned L_threshold =400; // controlling accuracy-speed tradeoff
	unsigned knn = 100; // k of knngraph

	float * query = NULL;
	float * basedata = NULL;
	unsigned * knngraph = NULL;
	unsigned * groundtruth = NULL;
	
	utils Utils;
	 Utils.load_xvecs(file_query, query, sift1m_dimension, sift1m_query_num);
	 Utils.load_xvecs(file_knngraph, knngraph, knn);
	 Utils.load_xvecs(file_knngraph, knngraph);
	 Utils.load_xvecs(file_groundtruth, groundtruth);
	 Utils.load_xvecs(file_basedata, basedata, sift1m_dimension, sift1m_num_of_samples);

	unsigned * result = (unsigned*)malloc(sift1m_query_num * K * sizeof(unsigned));
	memset(result, 0, sift1m_query_num * K * sizeof(unsigned));

	base Base = base(sift1m_dimension, sift1m_num_of_samples, sift1m_query_num);
	Base.set_start_point();

	auto start_clock = std::chrono::steady_clock::now();

	for (unsigned i = 0; i < sift1m_query_num; i++) {
		Base.AKNN_Search(
			(query+i*sift1m_dimension),
			basedata,
			knngraph,
			K,
			L,
			(result + i * K),
			knn
		);
	}
		
	auto end_clock = std::chrono::steady_clock::now();
	std::chrono::duration<double> time = end_clock - start_clock;

	float precision = Base.test_precision(result, groundtruth, K);

	printf("time: %f seconds | precision: %.3f | L= %d | %dNNgraph | %d queries\n", time, precision, L, knn, sift1m_query_num);
		
		
	/*------    End     ------*/


	/*------   AKNN Search based on Optimized Product Quantization      ------*/
		/*
			@brief: 
					- this is a sample script to do AKNN Search using OPQ in ADC way.
					- you could slightly modify it to run in SDC way
					- file path need to be specified.
					- Self-made binary file with extension like .u8code and .fcode could be find
					  in dataset file. And they could be made by Matlab in a naive way, 
					  read 'utils' class function 'load_my_binary_file()' for more information
					  about the self-made binary file.
			@run:
					remove the comment and run it
		*/

		/* "remove me to run this script"

	const char * file_basebook = "I:/AKNN/sift/sift/new256/codebook_256codelength_100iter.u8code";
	//const char * file_lookuptable = "I:/AKNN/sift/sift/new256/lookuptable_256codelength_100iter.fcode";
	const char * file_centroids = "I:/AKNN/sift/sift/new256/centroids_256codelength_100iter.fcode";
	
	// sdc distance computation 
	const char * file_query = "I:/AKNN/sift/sift/new256/queryR_256codelength_100iter.fcode";
	//const char * file_querybook = "I:/AKNN/sift/sift/new256/querybook_256codelength_100iter.u8code";

	const char * file_knngraph = "I:/AKNN/sift_100NN_100.graph";
	const char * file_groundtruth = "I:/AKNN/sift/sift/sift_groundtruth.ivecs";

	unsigned sift1m_num_of_subspace = 32;
	unsigned sift1m_num_of_samples = 1000000;
	unsigned sift1m_dimension = 128;
	unsigned sift1m_query_num = 10000;

	unsigned K = 100; // required number of nearest neighbors
	unsigned L = 400; // capacity of candidate pool
	unsigned knn = 100; // XNN-Graph


	uint8_t * basebook = (uint8_t*)malloc(sift1m_num_of_samples * sift1m_num_of_subspace * sizeof(uint8_t));
	//float * lookuptable = (float*)malloc(256 * 256 * sift1m_num_of_subspace * sizeof(float));
	float * centroids = (float*)malloc(256 * sift1m_dimension * sizeof(float));
	float * queryR = (float*)malloc(sift1m_dimension * sift1m_query_num * sizeof(float));
	//uint8_t * querybook = (uint8_t*)malloc(sift1m_query_num * sift1m_num_of_subspace * sizeof(uint8_t));

	unsigned * knngraph = NULL;
	unsigned * result = NULL;
	unsigned * groundtruth = NULL;

	utils Utils;
	Utils.load_my_binary_file(file_basebook, basebook, sift1m_num_of_subspace, sift1m_num_of_samples);
	// Utils.load_my_binary_file(file_lookuptable, lookuptable, sift1m_num_of_subspace);
	// sdc distance computation
	
	Utils.load_my_binary_file(file_query, queryR, sift1m_query_num, sift1m_dimension);
	//Utils.load_my_binary_file(file_querybook, querybook, sift1m_num_of_subspace, sift1m_query_num);


	//Utils.load_fvecs(file_query, query, sift1m_dimension, sift1m_query_num);
	Utils.load_xvecs(file_knngraph, knngraph);
	Utils.load_xvecs(file_groundtruth, groundtruth);

	Utils.load_my_binary_file(file_centroids, centroids, sift1m_dimension);

	result = (unsigned*)malloc(sift1m_query_num * K * sizeof(unsigned));
	memset(result, 0, sift1m_query_num * K * sizeof(unsigned));

	base Base = base(sift1m_dimension, sift1m_num_of_samples, sift1m_query_num);
	Base.set_start_point();
	Base.set_num_subspace(sift1m_num_of_subspace);

	auto s = std::chrono::high_resolution_clock::now();

	for (unsigned i = 0; i < sift1m_query_num; i++) {

		// ADC
		Base.AKNN_Search_opq(
			(queryR + i * sift1m_dimension),
			basebook,
			NULL,
			centroids,
			knngraph,
			knn,
			K,
			L,
			(result + i * K)
		);
			
			

			// SDC
		//	Base.AKNN_Search_opq(
			//	NULL,
			//	basebook,
			//	lookuptable,
			//	NULL,
			//	knngraph,
			//	knn,
			//	K,
			//	L,
			//	(result+i*K),
			//	i,
			//	querybook,
			//	true
		//	);
			
	}
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = e - s;
	float precision = Base.test_precision(result, groundtruth, K);
	printf("time: %f seconds | precision: %f | L= %d | %dNNgraph | 10000 queries\n", time, precision, L, knn);
		
		*/ 

	/*------    End    ------*/

	

	/*------    Optimized Product Quantization     ------*/
		/*
			@brief:
					- this is a sample script to do non-parameter-OPQ initialized by parameter-OPQ
					- file path need to be specified.
			@run:
					remove the comment and run it
		*/


		/* "remove me to run this script"
	base Base;
	utils Utils;

	const char *file_basedata = "I:/AKNN/sift/sift/sift_base.fvecs";
	
	float *basedata = NULL;
	
	unsigned dimension, num; // dimensionality and number of samples
	unsigned num_iter = 4; // number of iterations for k-means in parameter-OPQ and overall procedure in non-parameters-OPQ
							// for matrix computation using Eigen is slower than Matlab，set num_iter=4 for a fast demo,
							// larger iterations would get smaller distortion
	unsigned k = 256; // (fixed) 8 bits
	unsigned num_subspace = 32; // for codelength=256  (256/8=32)

	Utils.load_xvecs(file_basedata, basedata, dimension, num);
	
	// mainly to initialize Xtrain, i.e. data in Eigen matrix form, with basedata(withdraw memory).
	opq Opq(num_iter, k, num_subspace, dimension, num, basedata);

	//	calculate covariance
	Eigen::MatrixXf covX = Opq.covariance(Opq.Xtrain);
	
	// eigen decomposition is time/memory consuming
	// decompose it for once to save time
	Eigen::EigenSolver<Eigen::MatrixXf> es(covX);
	Eigen::MatrixXf evecs = es.eigenvectors().real();
	Eigen::MatrixXf evals = es.eigenvalues().real();
	
	// test how distortion change with different code length
	std::vector<float> distortions_np;
	unsigned M = 2;
	while (M <= 128) {
		Opq.m_num_subspace = M;
		printf("# of subspaces: %d\ncode length: %d\n", M, 8 * M);
		
		// reallocate eigen vectors to make eigenvalues in different bins close to each other
		Eigen::MatrixXf R = Opq.eigen_allocation_NOdecomposition(evecs, evals, Opq.m_num_subspace, Opq.m_dimension);
		
		Opq.Xtrain = Opq.Xtrain * R;

		// parameter-OPQ as initialization
		float distortion = Opq.train_pq(Opq.Xtrain, Opq.m_num_subspace, Opq.m_num_iter / 2, Opq.centers_table, Opq.idx_table);

		// non-parameter-OPQ
		float distortion_opq_np = Opq.train_opq_np(Opq.Xtrain, Opq.m_num_subspace, Opq.idx_table, Opq.centers_table, R, Opq.m_num_iter / 2, 1);
		
		// distortion for this code length
		distortions_np.push_back(distortion_opq_np);
		
		// double subspace, double codelength
		M *= 2;

		// re-prepare data to run next round
		Opq.reconstruct(M);
	}
	
	printf("\ndistortions over code length 16, 32, 64, 128, 512, 1024 are:");
	for (unsigned i = 0; i < distortions_np.size(); i++)
		printf("%f ",distortions_np[i]);

		*/
	/*------    End    ------*/

	return 0;
}
