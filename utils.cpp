#include "utils.h"
#include<iostream>
#include<fstream>
#include<vector>


namespace aknnspace {
	utils::utils()
	{
	}


	utils::~utils()
	{
	}

	void utils::load_ivecs(const char *filename, unsigned* &data) {
		printf("loading data from %s ... ...\n", filename);
		std::ifstream fin(filename, std::ios::binary);
		// error exception
		if (!fin.is_open()) {
			std::cout << "fail to read " << filename;
			exit(-1);
		}
		// every first 4 bytes of each vector represents its dim
		unsigned dim;
		fin.read((char*)&dim, 4);
		/* calculate num of data */
		// jump to end of file to calculate its length
		fin.seekg(0, std::ios::end);
		std::ios::pos_type ps = fin.tellg();
		// convert into byte unit
		size_t fsize = (size_t)ps;
		unsigned num;

		// every vector contains 4 + d*4 bytes (dim(4 bytes) + features(4 bytes every element))
		num = (unsigned)(fsize / (dim + 1) / 4);
		// allocate memory for data ¡¾.ivecs require unsigned or int type¡¿
		data = new unsigned[num*dim];
		// return to begin of file
		fin.seekg(0, std::ios::beg);
		for (size_t i = 0; i < num; i++) {
			// step over first 4 bytes in every vector
			fin.seekg(4, std::ios::cur);
			// read d*4 bytes every time
			fin.read((char*)(data + i * dim), dim * sizeof(unsigned));
		}
		fin.close();
	}
	void utils::load_fvecs(const char *filename, float* &data,  unsigned &dimension, unsigned &number) {
		/*	 most parts are similar to load_ivecs() except data type if float */
		printf("loading data from %s ......\n", filename);

		std::ifstream fin(filename, std::ios::binary);
		// error exception
		if (!fin.is_open()) {
			std::cout << "fail to read " << filename;
			exit(-1);
		}
		// every first 4 bytes of each vector represents its dim
		unsigned dim;
		fin.read((char*)&dim, 4);
		/* calculate num of data */
		// jump to end of file to calculate its length
		fin.seekg(0, std::ios::end);
		std::ios::pos_type ps = fin.tellg();
		// convert into byte unit
		size_t fsize = (size_t)ps;
		unsigned num;

		// every vector contains 4 + d*4 bytes (dim(4 bytes) + features(4 bytes every element))
		num = (unsigned)(fsize / (dim + 1) / 4);
		// allocate memory for data ¡¾.fvecs require float type¡¿
		data = new float[num*dim];
		// return to begin of file
		fin.seekg(0, std::ios::beg);
		for (size_t i = 0; i < num; i++) {
			// step over first 4 bytes in every vector
			fin.seekg(4, std::ios::cur);
			// read d*4 bytes every time
			fin.read((char*)(data + i * dim), dim * sizeof(float));
		}
		fin.close();

		dimension = dim;
		number = num;
	}

}


