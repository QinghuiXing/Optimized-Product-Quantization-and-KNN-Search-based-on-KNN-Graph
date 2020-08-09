#include "utils.h"
#include<iostream>
#include<fstream>
#include<assert.h>
#include<vector>


namespace aknnspace {
	utils::utils()
	{
	}


	utils::~utils()
	{
	}

	void utils::load_xvecs(const char * filename, unsigned *& knngraph, unsigned k)
	{
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary|std::ios_base::in);
		
		// check if open successfully
		assert(infile.is_open());
		
		// every first 4 bytes of each vector represents its dimensionality
		unsigned dimensionality;
		infile.read((char*)&dimensionality, 4);
		
		// calculate number of sample
		// jump to end of file to calculate its length
		infile.seekg(0, std::ios_base::end);
		std::ios::pos_type size_tmp = infile.tellg();
		
		// convert into byte unit
		size_t size_in_byte = (size_t)size_tmp;

		// every vector contains 4 + d*4 bytes (dimensionality(4 bytes) + features(4 bytes per element))
		unsigned num = (unsigned)(size_in_byte / (dimensionality + 1) / 4);

		// allocate memory for data 
		//		'.ivecs' require unsigned or int type
		//		read first k element only
		unsigned diff = dimensionality - k;
		knngraph = new unsigned[num*k];
		
		// return to begin of the file
		infile.seekg(0, std::ios_base::beg);
		
		for (size_t i = 0; i < num; i++) {
			// step over first 4 bytes in every vector
			infile.seekg(4, std::ios_base::cur);

			// read k * 4 bytes every time
			infile.read((char*)(knngraph + i * k), k* sizeof(unsigned));

			// step over the rest data
			infile.seekg(diff * 4, std::ios_base::cur);
		}

		infile.close();
		printf("done!\n");
	}
	
	void utils::load_xvecs(const char *filename, unsigned* &ivecs) {
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary);

		// check if open successfully
		assert(infile.is_open());

		// every first 4 bytes of each vector represents its dimensionality
		unsigned dimensionality;
		infile.read((char*)&dimensionality, 4);

		// calculate number of sample
		// jump to end of file to calculate its length
		infile.seekg(0, std::ios_base::end);
		std::ios::pos_type size_tmp = infile.tellg();

		// convert into byte unit
		size_t size_in_byte = (size_t)size_tmp;

		// every vector contains 4 + d*4 bytes (dimensionality(4 bytes) + features(4 bytes per element))
		unsigned num = (unsigned)(size_in_byte / (dimensionality + 1) / 4);
		
		// allocate memory for data 
		//		'.ivecs' require unsigned or int type
		ivecs = new unsigned[num*dimensionality];
		
		// return to begin of the file
		infile.seekg(0, std::ios_base::beg);

		for (size_t i = 0; i < num; i++) {
			// step over first 4 bytes in every vector
			infile.seekg(4, std::ios_base::cur);

			// read d*4 bytes every time
			infile.read((char*)(ivecs + i * dimensionality), dimensionality * sizeof(unsigned));
		}

		infile.close();
		printf("done!\n");
	}
	
	void utils::load_xvecs(const char *filename, float* &fvecs,  unsigned &dimension, unsigned &number) {
		//	 most parts are similar to load_ivecs() except data type is float
		
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary);
		
		// check if open successfully
		assert(infile.is_open());
		
		// every first 4 bytes of each vector represents its dimensionality
		unsigned dimensionality;
		infile.read((char*)&dimensionality, 4);

		// calculate number of sample
		// jump to end of file to calculate its length
		infile.seekg(0, std::ios_base::end);
		std::ios::pos_type size_tmp = infile.tellg();
		
		// convert into byte unit
		size_t size_in_byte = (size_t)size_tmp;

		// every vector contains 4 + d*4 bytes (dimensionality(4 bytes) + features(4 bytes per element))
		unsigned num = (unsigned)(size_in_byte / (dimensionality + 1) / 4);
		
		//num = 10000;
		// allocate memory for data 
		//		(.fvecs require float type)
		fvecs = new float[num*dimensionality];

		// return to begin of the file
		infile.seekg(0, std::ios_base::beg);

		for (size_t i = 0; i < num; i++) {
			// step over first 4 bytes in every vector
			infile.seekg(4, std::ios_base::cur);

			// read d*4 bytes every time
			infile.read((char*)(fvecs + i * dimensionality), dimensionality * sizeof(float));
		}

		infile.close();

		dimension = dimensionality;
		number = num;
		printf("done!\n");
	}


	void utils::load_my_binary_file(const char * filename, float * queryR, unsigned query_num, unsigned dimension)
	{
		printf("loading data from %s\t", filename);
		std::fstream infile(filename, std::ios_base::binary | std::ios_base::in);
		
		// check if open successfully
		assert(infile.is_open());

		// read data into one-dim array,
		//		shape: query_num * dimension (float32)
		for (unsigned i = 0; i < query_num; i++)
			for (unsigned j = 0; j < dimension; j++) {
				infile.read((char*)(queryR + i * dimension + j), 4);
			}

		infile.close();
		printf("done!\n");
	}

	void utils::load_my_binary_file(const char *filename, uint8_t *codebook, unsigned num_of_subspace, unsigned num_of_sample)
	{
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary | std::ios_base::in);
		
		// check if open successfully
		assert(infile.is_open());

		// read data into one-dim array,
		//		shape: num_of_sample * num_of_subspace (uint8)
		for (unsigned tmp_i = 0; tmp_i < num_of_sample; tmp_i++) {
			for (unsigned tmp_j = 0; tmp_j < num_of_subspace; tmp_j++) {
				infile.read((char*)(codebook + tmp_i * num_of_subspace + tmp_j), 1);
			}
		}

		infile.close();
		printf("done!\n");
	}

	void utils::load_my_binary_file(const char * filename, float * lookuptable, unsigned num_of_subspace)
	{
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary | std::ios_base::in);
		
		// check if open successfully
		assert(infile.is_open());

		// read data into one-dim array,
		//		shape: 256 * 256 * num_of_subspace (float32)
		for (unsigned tmp_k = 0; tmp_k < num_of_subspace; tmp_k++)
			for (unsigned tmp_i = 0; tmp_i < 256; tmp_i++)
				for (unsigned tmp_j = 0; tmp_j < 256; tmp_j++) 
					infile.read((char*)(lookuptable + tmp_k * 65536 + tmp_i * 256 + tmp_j), 4);
	
		infile.close();
		printf("done!\n");
	}

	void utils::load_my_binary_file(const char * filename, float * centroids, unsigned dimension, bool iscentroid)
	{
		printf("loading data from %s\t", filename);
		std::ifstream infile(filename, std::ios_base::binary | std::ios_base::in);
		
		// check if open successfully
		assert(infile.is_open());

		// read data into one-dim array,
		//		shape: 256 * dimension (float32)
		for (unsigned tmp_i = 0; tmp_i < 256; tmp_i++)
			for (unsigned tmp_j = 0; tmp_j < dimension; tmp_j++)
				infile.read((char*)(centroids + tmp_i * dimension + tmp_j), 4);
				// infile >> centroids[tmp_i*dimension + tmp_j];

		infile.close();
		printf("done!\n");
	}



	void utils::load_u8code(const char * filename, uint8_t * codebook, unsigned length)
	{// DEPRECATED
		std::fstream infile(filename, std::ios_base::binary | std::ios_base::in);
		for (int i = 0; i < length; i++)
			infile.read((char*)(codebook + i), 1);
		return;
	}
}


