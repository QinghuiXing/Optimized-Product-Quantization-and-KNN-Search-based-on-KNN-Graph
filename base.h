#pragma once

namespace aknnspace {
	class base {
		
	public:

		base();
		~base();
	
		void set_dimension(unsigned dim);
		void set_data_num(unsigned num);
		void set_query_num(unsigned num);

		void set_start_point_for_gist();

		void show_basic_info();

		struct neighbor {
			/*
				structure of vertex
				id : the global index
				distance : the distance to Q
				flag :
			*/
			unsigned id;
			float distance;
			bool flag;

			neighbor() = default;
			neighbor(unsigned int i, float dis, bool f) : id{ i }, distance{ dis }, flag(f){}

			bool operator<(const neighbor nn) const {
				return this->distance < nn.distance;
			}
		};

		float test_precision(const unsigned* output, const unsigned* groundtruth, unsigned K, unsigned k);


		float distance(const float* basedata_start, const float* query, unsigned dim);
			/*
				function: compute euclidean distance between basedata and query point
				parameters:
						basedata_start: start position of base data
						query: start position of query
						dim: dimention of data point (used to extract data)
			*/

		void graph_1d_to_2d(const unsigned * d1graph, std::vector<std::vector<unsigned>> &d2graph, unsigned K);

		unsigned insert_into_pool(std::vector<neighbor> &pool, unsigned L, neighbor NN);
			/*
				function: insert NN into pool and rearrange pool
				parameters:
					pool: candidates pool
					L: size of pool
					NN: neighbor to be inserted
			*/


		//void find_start_point(float* basedata, unsigned *knngraph, const unsigned K);

		void AKNN_Search(const float *query, const float *basedata,const	unsigned *knngraph, size_t K, size_t L, unsigned*indices);
			/*		parameters:
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

	
		unsigned  dimension;
		unsigned data_num;
		unsigned query_num;
		unsigned start_id;

	};
}