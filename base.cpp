#include<vector>
#include<algorithm>
#include<unordered_map>

#include"base.h"

namespace aknnspace {
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

	void base::set_start_point_for_gist()
	{
		this->start_id = 73484;
	}

	

	void base::show_basic_info()
	{
		printf("data_num: %d\n", this->data_num);
		printf("query_num: %d\n", this->query_num);
		printf("dimension: %d\n", this->dimension);
		printf("index of starting point in this dataset: %d", this->start_id);
	}

	float base::test_precision(const unsigned * output, const unsigned * groundtruth, unsigned K, unsigned k)
	{
		// output: K*query_num (K: number of neighbors required)
		// grountruth: k*query_num  (k: vector length of knn graph)
		
		float average_precision = 0;
		for (unsigned i = 0; i < this->query_num; i++) {
			std::unordered_map<unsigned, bool> hash;
			for (unsigned j = 0; j < K; j++) {
				hash.insert({{ groundtruth[i*k + j], true }});
			}
			unsigned cnt = 0;
			for (unsigned j = 0; j < K; j++) {
				if (hash.count(output[i*K + j]))cnt++;
			}
			average_precision += (float)cnt / (float)K;
		}
		average_precision /= this->query_num;
		return average_precision;
	}

	float base::distance(const float* basedata_start, const float* query, unsigned dim) {

		const float* last = basedata_start + dim;// end position
		float res = .0, tmp = .0;
		/*
			Euclidean Distance
		*/
		while (basedata_start < last) {
			tmp = (*basedata_start - *query);
			res += tmp * tmp;
			basedata_start++;
			query++;
		}
		return res;
	}

	void base::graph_1d_to_2d(const unsigned * d1graph, std::vector<std::vector<unsigned>> &d2graph, unsigned K)
	{
		for (unsigned num_i=0; num_i < this->data_num; num_i++)
			for (unsigned K_i=0; K_i < K; K_i++) {
				d2graph[num_i][K_i] = d1graph[num_i*K + K_i];
			}
	}

	unsigned base::insert_into_pool(std::vector<neighbor> &pool, unsigned L, neighbor NN) {
		unsigned left = 0, right = L - 1;
		/*
			 pool is sorted such that pool[left] is the nearest point to query,
			 if NN is nearer than it, NN is to be placed in the first one;
			 Note that memmove() will truncate the elements out of range
			 so pool's size is still L and pool will be rearranged automatically
		*/
		if (NN.distance <= pool[left].distance) {
			memmove(&pool[left + 1], &pool[left], L * sizeof(neighbor));
			pool[left] = NN;
			return left;
		}
		// NN is no shorter than the farest one, no need to insert.
		else if (NN.distance >= pool[right].distance) return L + 1;

		// find the first point further than NN
		// binary search can be faster, to be continued...
		unsigned tmp_pos = left;
		for (; tmp_pos <= right; tmp_pos++) {
			if (NN.distance < pool[tmp_pos].distance)
				break;
		}
		if (NN.id == pool[tmp_pos - 1].id)return L + 1;
		if (tmp_pos == right){
			pool[right] = NN;
			return right;
		}
		memmove(&pool[tmp_pos + 1], &pool[tmp_pos], (L - tmp_pos) * sizeof(neighbor));
		pool[tmp_pos] = NN;
		return tmp_pos;
		// coule linked list be more efficient than memmove() ? 

		/*
		// Or else,
		// find NN a proper position in pool
		// binary search is faster when L is large
		unsigned mid;
		while (left <= right) {
			mid = left + (right - left) / 2;
			if (NN.distance == pool[mid].distance)
				break;//scenario A : found, but NN & pool[mid] may be identical
			else if (NN.distance < pool[mid].distance)
				right = mid-1; //secnario B : not found and right<left
			else if(NN.distance > pool[mid].distance)
				left = mid+1; // secnario C : not found and right<left
		}

		// check identical or not
		if (NN.distance == pool[mid].distance&&NN.id == pool[mid].id)
			return -1;

		// in scenario B&C, need to find a proper position
		//	  scenario B :
		else if(NN.distance < pool[mid].distance) {
			while (mid > 0) {
				if (NN.distance > pool[mid].distance) {
					mid++;//the first point further than NN lies in mid
					break;
				}
				mid--;
			}
		}

		//  scenario C :
		else if (NN.distance > pool[mid].distance) {
			while (mid < L) {
				if (NN.distance < pool[mid].distance)
					break;//still, the first point further than NN lies in mid
			}
			mid++;
		}

		// In A,B or C, NN's place is one before mid
		memmove(&pool[mid + 1], &pool[mid], (L - mid) * sizeof(neighbor));
		pool[mid] = NN;
		return mid;
		*/
	}
    /*
	void base::find_start_point(float * basedata, unsigned * knngraph, unsigned K)
	{
		printf("finding starting point......\n");
		float* centroid = new float[this->dimension];

		for (unsigned dim_i = 0; dim_i < this->dimension; dim_i++)
			for (unsigned num_i = 0; num_i < this->data_num; num_i++)
				centroid[dim_i] += basedata[num_i*this->dimension + dim_i];

		
		printf("1d_2d transfer done!\n");
		this->start_id = rand() % this->data_num;
		printf("tmp start id: %d \n", this->start_id);

		//AKNN_Search(centroid, basedata, knngraph, 1, 100, &(this->start_id));
	}*/

	void base::AKNN_Search(const float*query, const float* basedata, const unsigned *knngraph, size_t K, size_t L, unsigned* indices) {

		//std::vector<std::vector<unsigned>> knngraph(this->data_num, std::vector<unsigned>(100));
		//graph_1d_to_2d(graph, knngraph, 100);

		//   candidates pool (starting point ans its K neignhbors, while minium of L is K, we set pool's size to L+1)
		std::vector<neighbor> candidates(L+1);
		//   find L points as initialization
		//std::vector<unsigned> init_ids(L);
		//    id point has been checked if flags[id] is true;
		std::vector<bool> flags(data_num, false);
		
		
		// add start point into candidates pool
		float dis = distance(basedata + start_id * dimension, query, dimension);
		unsigned candi_pos = 0;
		candidates[candi_pos++] = neighbor(start_id, dis, true);
		flags[start_id] = true;// mark start point as checked
		
		
		//unsigned candi_pos = 0;
		/* below is to full fill candidates pool, in order to make the updating process running safely */
		
		// fill candidates pool with the first k neighbors of start point ****knn graph****
		for (unsigned tmp_j = 0; tmp_j < L&&tmp_j < 100; tmp_j++) {
			unsigned tmp_id = knngraph[start_id*100+tmp_j];
			float dis = distance(basedata + tmp_id * dimension, query, dimension);
			candidates[candi_pos++] = neighbor(tmp_id, dis, true);
			flags[tmp_id] = true;
		} 

		// if not fully filled, fill with some  random points
		while (candi_pos < L) {
			unsigned id = rand() % data_num;
			if (flags[id])continue;//has been checked
			float dis = distance(basedata + id * dimension, query, dimension);
			candidates[candi_pos++] = neighbor(id, dis, true);
			//printf("%d ", L);
			flags[id] = true;
		}
		//printf("initilization done!\n");
		

		// sort candidates pool in ascending order of the distance to query
		//std::sort(candidates.begin(), candidates.begin() + L);
		/* main part of the algorithm  */
		/* searching from start point iteratively to update candidates pool  */
		// tmp_i £ºposition of start point in candidates pool 

		//unsigned cnt = 0;

		unsigned sp_pos = 0;
		while (sp_pos < L) {//no shorter neighbor of start point for now
			// position of nearest point in candidates pool, initialized as L
			unsigned nearest_pos = L;
			//if (cnt++ > 10) break;
			if (candidates[sp_pos].flag) {//if not checked (ticket to be a starting point)
				candidates[sp_pos].flag = false;//checked now
				unsigned id = candidates[sp_pos].id; // global index of starting point
																  //****knn graph****
				for (unsigned i = 0; i < 100; i++) {//processing k(=100) nearest neighbors of starting point
					unsigned neighbor_id = knngraph[id*100+i];
					if (flags[neighbor_id])continue;//if checked, continue
					flags[neighbor_id] = true;
					
					float dis = distance(basedata + neighbor_id * dimension, query, dimension);
					neighbor nn(neighbor_id, dis, true);
					//printf("ready to insert into pool\n");
					unsigned tmp_pos = insert_into_pool(candidates,L, nn);
					//printf("already inserted!\n");
					if (tmp_pos < nearest_pos)
						nearest_pos = tmp_pos;
				}
			}
			/*  1) "nearest_pos < tmp_i"  tells that the next start point's position in candidates pool is 'nearest_pos',
				 2)or else candidates[tmp_i] has no neighbor with a nearer distance to Q, thus the next start point goes to tmp_i++;
					which is the reason we fully fill candidates pool in advance.        */
			if (nearest_pos <= sp_pos)
				sp_pos = nearest_pos;
			else
				sp_pos++;
			//printf("update next start point!\n");
		}
		// select first K nearest neighbors from candidates pool
		for (unsigned i = 0; i < K; i++) {
			indices[i] = candidates[i].id;
			//printf("index %d : %d", i, indices[i]);
		}
		//printf("get %d beighbors !\n",K);
		
	}


}
