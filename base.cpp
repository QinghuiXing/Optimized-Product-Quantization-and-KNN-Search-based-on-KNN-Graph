#include<vector>
#include<algorithm>
#include<unordered_map>
#include<queue>
#include<cassert>
#include<immintrin.h>
#include"base.h"

namespace aknnspace {
	base::base(unsigned m_dimension, unsigned m_data_num, unsigned m_query_num)
	{
		dimension = m_dimension;
		data_num = m_data_num;
		query_num = m_query_num;
	}
	base::base(){}
	base::~base(){}



	void base::set_dimension(unsigned dim)
	{
		this->dimension = dim;
	}

	void base::set_data_num(unsigned num)
	{
		this->data_num = num;
	}

	void base::set_query_num(unsigned num)
	{
		this->query_num = num;
	}

	void base::set_num_subspace(unsigned num)
	{
		this->self_num_of_subspace = num;
	}

	void base::set_start_point()
	{
		this->start_id = 0; // 73484
	}

	

	void base::show_basic_info()
	{
		printf("data_num: %d\n", this->data_num);
		printf("query_num: %d\n", this->query_num);
		printf("dimension: %d\n", this->dimension);
		printf("index of starting point in this dataset: %d", this->start_id);
	}

	float base::test_precision(const unsigned * output, const unsigned * groundtruth, unsigned K)
	{

		float average_precision = 0;

		for (unsigned i = 0; i < this->query_num; i++) {
			std::unordered_map<unsigned, bool> hash;

			for (unsigned j = 0; j < K; j++)
				hash.insert({{ groundtruth[i*K + j], true }});
			
			unsigned cnt = 0;
			
			for (unsigned j = 0; j < K; j++)
				if (hash.count(output[i*K + j]))cnt++;

			average_precision += (float)cnt / (float)K;
		}

		average_precision /= this->query_num;
		
		return average_precision;
	}

	float base::test_part_precision(const unsigned * output, const unsigned * groundtruth, unsigned K)
	{
		float average_precision = 0;
		std::unordered_map<unsigned, bool> hash;
		for (unsigned j = 0; j < K; j++) {
			hash.insert({ { groundtruth[j], true } });
		}
		unsigned cnt = 0;
		for (unsigned j = 0; j < K; j++) {
			if (hash.count(output[j]))cnt++;
		}
		average_precision = float(cnt) / 100;
		return average_precision;
	}

	float base::adc_distance(unsigned id, const float * query, const uint8_t *codebook, const float *centroids)
	{	//id starts from 0
		
		unsigned sub_dim = dimension / self_num_of_subspace;

		float dist = 0;
		for (unsigned subspace_index = 0; subspace_index < self_num_of_subspace; subspace_index++) {
			
			// index of centroid for this subspace (0-based)
			unsigned centroid_index = codebook[id*self_num_of_subspace + subspace_index];

			const float *sub_centroid = centroids + centroid_index*dimension + sub_dim * subspace_index;
			const float *sub_query = query+sub_dim*subspace_index;
			
			dist += avx_distance(sub_centroid, sub_query, sub_dim);
			// dist += distance(sub_centroid, sub_query, sub_dim);
		}

		return dist;
	}

	float base::sdc_distance(unsigned base_id, unsigned query_id, const uint8_t * basebook, const uint8_t * querybook, const float *lookuptable)
	{
		unsigned sub_dim = dimension / self_num_of_subspace;

		float dist = 0;
		for (unsigned subspace_index = 0; subspace_index < self_num_of_subspace; subspace_index++) {
			unsigned base_centroid_index = basebook[base_id * self_num_of_subspace + subspace_index];
			unsigned query_centroid_index = querybook[query_id * self_num_of_subspace + subspace_index];

			dist += *(lookuptable + subspace_index * 256 * 256 + base_centroid_index * 256 + query_centroid_index);			
		}
		return dist;
	}


	float base::distance(const float *basedata_start, const float *query, unsigned dim) {

		float res = .0, tmp = .0;

		while(dim--){
			tmp = (*basedata_start - *query);
			res += tmp * tmp;
			basedata_start++;
			query++;
		}

		return res;
	}

	float base::avx_distance(const float * base, const float * query, unsigned dim)
	{

		// AVX-256 cannot run on vector who's dim<8, thus calculate distance in exhausted way 
		if (dim < 8) {
			float res = .0, tmp = .0;
			while (dim--) {
				tmp = (*base - *query);
				res += tmp * tmp;
				base++;
				query++;
			}
			return res;
		}

		// AVX256, 8 float32 per register
		unsigned iters = dim / 8;
		unsigned rest = dim % 8;
		float res = 0;
		__m256 base_container, query_container;
		__m256 sum = _mm256_setzero_ps();

		for (unsigned i = 0; i < iters; i++) {
			base_container = _mm256_loadu_ps(base + i * 8);
			query_container = _mm256_loadu_ps(query + i * 8);

			// difference of base to query
			__m256 sub = _mm256_sub_ps(base_container, query_container);

			// square of difference
			sum = _mm256_fmadd_ps(sub, sub, sum);
		}

		// sum of values in register
		sum = _mm256_hadd_ps(sum, sum);
		sum = _mm256_hadd_ps(sum, sum);
		res = sum.m256_f32[0];
		res += sum.m256_f32[4];
		
		// the rest is calculated in a naive way
		for (int i = 0; i < rest; i++) {
			float b = (*(base + iters * 8 + i));
			float q = (*(query + iters * 8 + i));
			res += (b - q)*(b - q);
		}

		return res;
	}


	void base::graph_1d_to_2d(const unsigned * d1graph, std::vector<std::vector<unsigned>> &d2graph, unsigned K)
	{//DEPRECATED
		for (unsigned num_i=0; num_i < this->data_num; num_i++)
			for (unsigned K_i=0; K_i < K; K_i++) {
				d2graph[num_i][K_i] = d1graph[num_i*K + K_i];
			}
	}

	unsigned base::naive_insert(std::vector<neighbor> &pool, unsigned L, neighbor NN) 
	{
		/*
			 pool is sorted such that pool[left] is the nearest point to query,
			 if NN is nearer than it, NN is to be placed in the first one;
			 Note that memmove() will truncate the elements out of range
			 so pool's size is still L and pool will be rearranged automatically
		*/
		unsigned left = 0, right = L - 1;
		
		// most vectex are not in the range of candidate pool, thus check head and tail of candidate pool firstly to speed up
		if (NN.distance <= pool[left].distance) {
			memmove(&pool[left + 1], &pool[left], L * sizeof(neighbor));
			pool[left] = NN;
			return left;
		}
		else if (NN.distance >= pool[right].distance) return L + 1;

		// search for the first point in pool further than NN
		unsigned tmp_pos = left;
		for (; tmp_pos <= right; tmp_pos++) {
			if (NN.distance < pool[tmp_pos].distance)
				break;
		}
		
		// check if nn is identical to vertexs in pool
		if (NN.id == pool[tmp_pos - 1].id)return L + 1;
		
		// special case in using memmove()
		if (tmp_pos == right){
			pool[right] = NN;
			return right;
		}

		// put nn in its proper position
		memmove(&pool[tmp_pos + 1], &pool[tmp_pos], (L - tmp_pos) * sizeof(neighbor));
		pool[tmp_pos] = NN;
		return tmp_pos;
	}

	unsigned base::binary_insert(std::vector<neighbor>& pool, unsigned L, neighbor NN)
	{
		/*
			most parts are similar to naive_insert()
			except that it employs binary search to find the proper position of NN
		*/

		unsigned left = 0, right = L - 1, mid;

		// most vectex are not in the range of candidate pool, thus check head and tail of candidate pool firstly to speed up
		if (NN.distance <= pool[left].distance) {
			memmove(&pool[left + 1], &pool[left], L * sizeof(neighbor));
			pool[left] = NN;
			return left;
		}
		else if (NN.distance >= pool[right].distance) return L + 1;
		
		// find the number of vertexs 'pos' nearer to query than NN
		unsigned l = left, r = right + 1;
		while (l < r) {
			mid = l + (r - l) / 2;
			if (pool[mid].distance == NN.distance) {
				r = mid;
			}
			else if (pool[mid].distance < NN.distance) {
				l = mid + 1;
			}
			else if (pool[mid].distance > NN.distance) {
				r = mid;
			}
		}

		unsigned pos = mid;
		if (pos > right)
			return L + 1;

		while (pos <= right) {

			// check if nn is identical to vertexs in pool
			if (pool[pos].distance == NN.distance){
				if (pool[pos].id == NN.id)return L + 1;
				else {
					pos++;
					continue;
				}
			}
			if (pool[pos].distance > NN.distance) {
				if (pos == right) {

					// special case in using memmove()
					pool[pos] = NN;
					return right;
				}
				else {

					// put nn in its proper position
					memmove(&pool[pos + 1], &pool[pos], (L - pos) * sizeof(neighbor));
					pool[pos] = NN;
					return pos;
				}
			}
			pos++;
		}
		return L+1;
	}

	void base::AKNN_Search_opq(const float *query, const uint8_t *codebook, const float *lookuptable, const float *centroids,
		const unsigned *knngraph, unsigned num_nn_neighbor, unsigned K, unsigned L, unsigned *indices,
		unsigned query_id, const uint8_t * querybook, bool is_sdc)
	{
		// candidate pool
		std::vector<neighbor> candidates(L+1,neighbor(0,0,false));

		// check if this point has been considered to insert into candidate pool or not (true for already considered)
		std::vector<bool> flags(data_num,false);
		
		// index to indicate position in candidate pool during initialization
		unsigned candidate_pos = 0;
		
		// add starting point into candidate pool
		float dis = .0;
		if (is_sdc){
			
			// calculate distance in SDC way
			assert(query_id >= 0 && query_id < 10000);
			dis = sdc_distance(start_id, query_id, codebook, querybook, lookuptable);
		}
		else {
			
			// calculate distance in ADC way
			dis = adc_distance(start_id, query, codebook, centroids);
		}
		candidates[candidate_pos++] = neighbor(start_id, dis, true);
		flags[start_id] = true;
		
		// add neighbors of starting point into candidate pool, but within pool's capacity
		for (unsigned tmp_j = 0; tmp_j < L && tmp_j < num_nn_neighbor; tmp_j++) {
			unsigned tmp_id = knngraph[start_id * num_nn_neighbor + tmp_j];

			// calculate distance and insert into pool
			float dis = .0;
			if (is_sdc)
				dis = sdc_distance(tmp_id, query_id, codebook, querybook, lookuptable);
			else 
				dis = adc_distance(tmp_id, query, codebook, centroids);
			candidates[candidate_pos++] = neighbor(tmp_id, dis, true);
			flags[tmp_id] = true;
		}

		// if not fully filled by neighbors, randomly choose some points until candidate pool is full
		while (candidate_pos < L) {
			unsigned id = rand() % data_num;
			
			//if has been checked
			if (flags[id])continue;
			
			// calculate distance and insert into pool
			float dis = .0;
			if (is_sdc)
				dis = sdc_distance(id, query_id, codebook, querybook, lookuptable);
			else
				dis = adc_distance(id, query, codebook, centroids);
			candidates[candidate_pos++] = neighbor(id, dis, true);
			flags[id] = true;
		}

		// sort candidate pool in increasing order on distance to querypoint
		std::sort(candidates.begin(), candidates.begin() + L);

		/*****  line 3 to line 12 in Algorithm 1  *****/
		unsigned startpoint_pos = 0;
		
		// break when no points in candidate pool have neighbors nearer to query.(locally optimal)
		while (startpoint_pos < L) {
			
			//  position of this starting point's nearest neighbor in candidates pool, initialized as L
			unsigned nearest_pos = L;
			if (candidates[startpoint_pos].flag) { 

				// no point can be a starting point for twice
				candidates[startpoint_pos].flag = false;
				
				// global index of starting point
				unsigned id = candidates[startpoint_pos].id;
				for (unsigned i = 0; i < num_nn_neighbor; i++) {
					unsigned neighbor_id = knngraph[id * num_nn_neighbor + i];
					
					//if it has already been considered
					if (flags[neighbor_id])continue;
					flags[neighbor_id] = true;

					// calculate distance and insert into candidate pool in proper position
					float dis = .0;
					if (is_sdc)
						dis = sdc_distance(neighbor_id, query_id, codebook, querybook, lookuptable);
					else
						dis = adc_distance(neighbor_id, query, codebook, centroids);
					neighbor nn(neighbor_id, dis, true);
					unsigned tmp_pos = binary_insert(candidates, L, nn);
					//unsigned tmp_pos = insert_into_pool(candidates, L, nn);
					
					// update nearest neighbor
					if (tmp_pos < nearest_pos)
						nearest_pos = tmp_pos;
				}
			}

			// (line 4 in Algorithm 1) next starting point lies in (for now) the nearest point in candidate pool
			if (nearest_pos <= startpoint_pos)
				startpoint_pos = nearest_pos;
			else
				startpoint_pos++;
		}

		// select the nearest K candidates as K nearest neighbors to querypoint
		for (unsigned i = 0; i < K; i++) {
			indices[i] = candidates[i].id;
		}
		return;
	}

	void base::AKNN_Search(const float* query, const float* basedata, const unsigned* knngraph, size_t K, size_t L, unsigned* indices, unsigned knn) {

		// candidates pool
		std::vector<neighbor> candidates(L+1);

		// check if this point has been considered to insert into candidate pool or not (true for already considered)
		std::vector<bool> flags(data_num, false);

		// index to indicate position in candidate pool during initialization
		unsigned candi_pos = 0;

		// add starting point into candidates pool
		float dis = avx_distance(basedata + start_id * dimension, query, dimension);
		//float dis = distance(basedata + start_id * dimension, query, dimension);
		candidates[candi_pos++] = neighbor(start_id, dis, true);
		flags[start_id] = true;
		
		// add neighbors of starting point into candidate pool, but within pool's capacity
		for (unsigned tmp_j = 0; tmp_j < L && tmp_j < knn; tmp_j++) {
			unsigned tmp_id = knngraph[start_id*knn +tmp_j];

			// calculate distance and insert into pool
			float dis = avx_distance(basedata + tmp_id * dimension, query, dimension);
			//float dis = distance(basedata + tmp_id * dimension, query, dimension);
			candidates[candi_pos++] = neighbor(tmp_id, dis, true);
			flags[tmp_id] = true;
		} 

		// if not fully filled by neighbors, randomly choose some points until candidate pool is full
		while (candi_pos < L) {
			unsigned id = rand() % data_num;

			//if has been checked
			if (flags[id])continue;

			// calculate distance and insert into pool
			float dis = avx_distance(basedata + id * dimension, query, dimension);
			//float dis = distance(basedata + id * dimension, query, dimension);
			candidates[candi_pos++] = neighbor(id, dis, true);
			flags[id] = true;
		}


		// sort candidates pool in ascending order on the distance to query
		std::sort(candidates.begin(), candidates.begin() + L);

		// line 3 to line 12 in Algorithm 1
		unsigned sp_pos = 0;

		// break when no points in candidate pool have neighbors nearer to query.(locally optimal)
		while (sp_pos < L) {
			
			//  position of this starting point's nearest neighbor in candidates pool, initialized as L
			unsigned nearest_pos = L;
			
			// no point can be a starting point for twice
			if (candidates[sp_pos].flag) {

				// no point can be a starting point for twice
				candidates[sp_pos].flag = false;
				unsigned id = candidates[sp_pos].id; 
				for (unsigned i = 0; i < knn; i++) {
					unsigned neighbor_id = knngraph[id*knn +i];

					//if it has already been considered
					if (flags[neighbor_id])continue;
					flags[neighbor_id] = true;

					// calculate distance using optimized product quantization
					float dis = avx_distance(basedata + neighbor_id * dimension, query, dimension);
					//float dis = distance(basedata + neighbor_id * dimension, query, dimension);
					
					// insert neighbor into candidate pool in proper position
					neighbor nn(neighbor_id, dis, true);
					unsigned tmp_pos = binary_insert(candidates, L, nn);
					//unsigned tmp_pos = naive_insert(candidates, L, nn);

					// update nearest neighbor
					if (tmp_pos < nearest_pos)
						nearest_pos = tmp_pos;
				}
			}
			// (line 4 in Algorithm 1) next starting point lies in (for now) the nearest point in candidate 
			if (nearest_pos <= sp_pos)
				sp_pos = nearest_pos;
			else
				sp_pos++;

		}

		// select first K nearest neighbors from candidates pool
		for (unsigned i = 0; i < K; i++) {
			indices[i] = candidates[i].id;
		}
		
	}


}
