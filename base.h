#pragma once

namespace aknnspace {
	class base {
		
	public:
		base(unsigned m_dimension, unsigned m_data_num, unsigned m_query_num);
		base();
		~base();

		// DEPRECATED
		void set_dimension(unsigned dim);
		void set_data_num(unsigned num);
		void set_query_num(unsigned num);

		void set_num_subspace(unsigned num);
		void set_start_point();

		void show_basic_info();

		struct neighbor {
			/*
				@brief: structure of vertex
				@elems:
					id : the global index
					distance : the distance to Q
					flag : check if has been considered to insert into candidate pool
			*/
			unsigned id;
			float distance;
			bool flag;

			neighbor() : id(), distance(), flag() {}
			neighbor(unsigned int i, float dis, bool f) : id(i), distance(dis), flag(f){}
			~neighbor() {}
			
			// overload function comp, increasing order
			bool operator<(const neighbor nn) const {
				return this->distance < nn.distance;
			}
		};

		float test_precision(const unsigned* output, const unsigned* groundtruth, unsigned K);
		/*
			@brief: calculate precision over all queries
			@params:
				output: pointer to result array, shape: num of queries * K
				groundtruth: pointer to groundtruth array, same format as  output
				K: num of nearest neighbors obtained
		*/


		float test_part_precision(const unsigned* output, const unsigned* groundtruth, unsigned K);
		/*
			@brief: for DEBUG
		*/

		float adc_distance(unsigned id, const float* query, const uint8_t* basebook, const float* centroids);
		/*
			@brief: calculate distance two point in ADC way
			@params: 
				id: 
				query:
				basebook:
				centroids:
		*/

		float sdc_distance(unsigned base_id, unsigned query_id, const uint8_t* basebook, const uint8_t* querybook, const float* lookuptable);

		float distance(const float* basedata_start, const float* query, unsigned dim);
			/*
				@brief: compute euclidean distance between basedata and query point
				@params:
					basedata_start: start position of base data
					query: start position of query
					dim: dimention of data point (used to extract data)
			*/

		float avx_distance(const float* base, const float* query, unsigned dim);
		/*
			@brief: (AVX256 version) compute euclidean distance between basedata and query point
				@params:
					basedata_start: start position of base data
					query: start position of query
					dim: dimention of data point (used to extract data)
		*/

		// DEPRECATED
		void graph_1d_to_2d(const unsigned* d1graph, std::vector<std::vector<unsigned>> &d2graph, unsigned K);

		unsigned naive_insert(std::vector<neighbor> &pool, unsigned L, neighbor NN);
			/*
				@brief: insert NN into pool and rearrange pool
				@params:
					pool: candidates pool
					L: size of pool
					NN: neighbor to be inserted
			*/

		unsigned binary_insert(std::vector<neighbor> &pool, unsigned L, neighbor NN);
			/*
				@brief: most parts are similar to naive_insert()
						except that it employs binary search to find the proper position of NN
				@params:
					pool: candidates pool
					L: size of pool
					NN: neighbor to be inserted
			*/


		void AKNN_Search_opq(const float* query, const uint8_t* basebook, const float* lookuptable, const float* centroids,
			const unsigned* knngraph, unsigned num_nn_neighbor, unsigned K, unsigned L, unsigned* indices,
			unsigned query_id = 0, const uint8_t* querybook = NULL, bool is_sdc = false);
		/*
				@brief: (potimized vector quantization version) approximate nearest neighbor search algorithm based on KNN-Graph
				@params:
					query£º(Input) querydata£¬pointing to a one-dimentional (num_query*dim_query) vector
								with num representing total number of query vertex, dim representing
								the length of one query data point
					basedata£º(Input) basedata, same stucture (num_basedata*dim_basedata) as querydata.
									the dim must be the same as querydata !
					K£º(Input) the number of (approximate) nearest neighbors required to give.
					L£º(Input) the max number of vertexes in candidates pool, the larger, the more accurate the result is, and
									the slower.
					indices£º(Output) result container. pointing to the vector containing (num_query*K )indices of the K (approximate)
									nearest neighbors of  given queries.
			*/

		void AKNN_Search(const float* query, const float* basedata,const unsigned* knngraph, size_t K, size_t L, unsigned*indices, unsigned knn=100);
			/*	
				@brief: approximate nearest neighbor search algorithm based on KNN-Graph
				@params:
					query£º(Input) querydata£¬pointing to a one-dimentional (num_query*dim_query) vector
								with num representing total number of query vertex, dim representing
								the length of one query data point
					basedata£º(Input) basedata, same stucture (num_basedata*dim_basedata) as querydata.
									the dim must be the same as querydata !
					K£º(Input) the number of (approximate) nearest neighbors required to give.
					L£º(Input) the max number of vertexes in candidates pool, the larger, the more accurate the result is, and
									the slower.
					indices£º(Output) result container. pointing to the vector containing (num_query*K )indices of the K (approximate)
									nearest neighbors of  given queries.
			*/

		unsigned self_num_of_subspace;
		unsigned dimension;
		unsigned data_num;
		unsigned query_num;
		unsigned start_id;

	};
}