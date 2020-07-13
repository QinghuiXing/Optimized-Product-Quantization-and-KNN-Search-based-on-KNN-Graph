#include "pch.h"
#include <iostream>
#include<fstream>
#include<vector>
#include<chrono>
#include"utils.h"
#include"base.h"

using namespace aknnspace;

int main()
{
	base Base;
	utils Utils;

	const char * gist_graph= "G:/AKNN/sift/sift_100NN_100.graph";
	const char *gist_base = "G:/AKNN/sift/sift_base.fvecs";
	const char *gist_query = "G:/AKNN/sift/sift_query.fvecs";
	const char *gist_learn = "G:/AKNN/sift/sift_learn.fvecs";
	const char *gist_groundtruth= "G:/AKNN/sift/sift_groundtruth.ivecs";

	unsigned *graph = NULL;
	float *base_data = NULL;
	float *query_data = NULL;
	unsigned *groundtruth = NULL;

	unsigned K = 100;// number of neighbors to search
	unsigned L = 180;// size of candidates pool
	//unsigned k = 100;//k of input 'k'nn graph

	Utils.load_ivecs(gist_graph, graph);
	Utils.load_ivecs(gist_groundtruth, groundtruth);

	unsigned dimension, num;
	Utils.load_fvecs(gist_base, base_data, dimension, num);
	Base.set_dimension(dimension);
	Base.set_data_num(num);

	Utils.load_fvecs(gist_query, query_data, dimension, num);
	Base.set_query_num(num);
	Base.set_start_point_for_gist();

	unsigned *result = new unsigned[Base.query_num * K];



	auto s = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < Base.query_num; i++)
		Base.AKNN_Search((query_data+i*dimension), base_data, graph, K,  L, (result+i*K));
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = e - s;

	float precision = Base.test_precision(result, groundtruth, K, 100);

	printf("performance: %d queries , %f s, %d neighbors, L=%d ",
		Base.query_num, time.count(), K, L);
	printf("precision: %f\n", precision);

	delete base_data;
	delete query_data;
	delete groundtruth;
	delete graph;

	return 0;
}
