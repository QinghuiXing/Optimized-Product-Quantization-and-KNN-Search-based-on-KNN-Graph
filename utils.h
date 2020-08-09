#pragma once
#include<vector>
namespace aknnspace {
	class utils
	{
	public:
		utils();
		~utils();
		
		void load_xvecs(const char* filename, unsigned* &knngraph, unsigned k);
		/*
			@brief:load 'k'NN-Graph into one-dim array; k is specified.
			@params:
				filename: file path of the Graph
				data: pointer to head of kNN-Graph array, shape: num of sample * k (num of sample is encoded in the file)
				k: customed k for kNN-Graph.
		*/

		void load_xvecs(const char* filename, unsigned* &ivecs);
		/*
			@brief: load binary file encoded for 'int' into one-dim array
			@params:
				filename: file path of binary file
				data: pointer to head of data array, shape: number of sample * dimensionality (both encoded in the file)
		*/
		
		void load_xvecs(const char* filename, float* &fvecs, unsigned &dimension, unsigned &number);
		/*
			@brief: load binary file encoded for 'float' into one-dim array;
					initialized dimension and # of sample of data.
			@params:
				filename: file path of binary file
				data: pointer to head of data array, shape: number * dimension
				dimension: dimension of dataset, i.e. 128 for SIFT-1M, 960 for GIST-1M/1B
				number: number of sample of dataset, i.e. 1000000 for base-data in SIFT-1M,
						10000 for query-data in SIFT-1M
		*/

		void load_u8code(const char* filename, uint8_t* codebook, unsigned length);
		/*
		DEPRECATED
			@brief: load self-customed binary file encoded for 'uint8_t' into one-dim array;
			@params:
				filename: file path of binary file
				codebook: pointer to head of data array
				length: total size of array		
		*/

		void load_my_binary_file(const char* filename, float* queryR, unsigned query_num, unsigned dimension);
		/*
			@brief: load query points transformed by R(refer to non-parameter method in OPQ) into one-dim array
					self-customed binary file encoded in 'float'
			@params:
				filename: file path of binary file
				queryR: pointer to head of data array, shape: query_num * dimension
				query_num: # of query points. (not encoded in file, thus need to be specified)
							i.e. 10000 for SIFT-1M, 1000 for GIST-1M/1B
				dimension: dimensionality of query points. (also need to be specified)
							i.e. 128 for SIFT-1M, 960 for GIST-1M/1B
		*/

		void load_my_binary_file(const char* filename, uint8_t* codebook, unsigned num_of_subspace, unsigned num_of_sample);
		/*
			@brief: load base-data/query-data codebook into one-dim array from self-customed binary file
			@params:
				filename: file path of binary file
				codebook: pointer to head of data array, shape: num_of_sample * num_of_subspace
				num_of_subspace: # of subspace
				num_of_samples: # of sample
		*/

		void load_my_binary_file(const char* filename, float* lookuptable, unsigned num_of_subspace);
		/*
			@brief: load look-up-table into one-dim array from self-customed binary file
			@params:
				filename: file path of binary file
				lookuptable: pointer to head of data array, shape: 256 * 256 * num_of_space for fixed # of codewords (=256)
				num_pf_subspace: # of subspace
		*/

		void load_my_binary_file(const char* filename, float* centroids, unsigned dimension, bool iscentroid);
		/*
			@brief: load centroids into one-dim array from self-customed binary file
			@params:
				filename: file path of binary file
				centroids: pointer to head of data array, shape: 256 * dimension for fixed # of codewords (=256)
				dimension: dimensionality of centroids (same as base-data).
		*/
	};

}

