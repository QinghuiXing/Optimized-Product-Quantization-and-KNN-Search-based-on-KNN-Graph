#include "opq.h"
#include<Eigen/Dense>
#include<iostream>
#include<chrono>
#include<numeric>
#include<algorithm>



namespace aknnspace {
	opq::opq(unsigned num_iter, unsigned k, unsigned num_subspace, unsigned dimension, unsigned num_samples, float* base_data)
	{
		m_num_iter = num_iter;
		m_k = k;
		m_num_subspace = num_subspace;
		m_dimension = dimension;
		m_num_samples = num_samples;

		idx_table = std::vector<std::vector<unsigned>>(num_subspace);
		Xtrain.resize(num_samples, dimension);
		// maybe not very efficient
		for (unsigned tmp_i = 0; tmp_i < num_samples; tmp_i++)
			for (unsigned tmp_j = 0; tmp_j < dimension; tmp_j++) {
				Xtrain(tmp_i,tmp_j) = base_data[tmp_i*dimension + tmp_j];
			}
		delete base_data;

	}
	opq::opq()
	{
	}


	opq::~opq()
	{
	}

	void opq::reconstruct(unsigned new_num_subspace)
	{
		centers_table.clear();
		for (unsigned tmp_i = 0; tmp_i < idx_table.size(); tmp_i++)
			idx_table[tmp_i].clear();
		idx_table.clear();
		idx_table.resize(new_num_subspace);
	}

	Eigen::MatrixXf aknnspace::opq::covariance(Eigen::MatrixXf &X)
	{
		// mean of each dimension
		Eigen::MatrixXf meanVec = X.colwise().mean();
		Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanVec.data(), X.cols()));
		
		// sub mean
		Eigen::MatrixXf zeroMeanMat = X;
		zeroMeanMat.rowwise() -= meanVecRow;
		
		// calculate covariance matrix of X
		Eigen::MatrixXf covMat;
		if (X.rows() == 1)
			covMat = (zeroMeanMat.transpose()*zeroMeanMat) / float(X.rows());
		else
			covMat = (zeroMeanMat.transpose()*zeroMeanMat) / float(X.rows() - 1);

		return covMat;
	}

	Eigen::MatrixXf opq::rand_center(Eigen::MatrixXf & data, unsigned k)
	{
		int n = data.cols();
		
		// allocate space for centroids
		Eigen::MatrixXf centroids = Eigen::MatrixXf::Zero(k, n);
		
		// randomly generate centroids within the range of data 
		double min, range;
		for (int i = 0; i < n; i++)
		{
			min = data.col(i).minCoeff();
			range = data.col(i).maxCoeff() - min;
			centroids.col(i) = min * Eigen::MatrixXf::Ones(k, 1) + Eigen::MatrixXf::Random(k, 1).cwiseAbs() * range;
		}
		return centroids;
	}

	Eigen::MatrixXf opq::kmeans_customed(Eigen::MatrixXf &data, unsigned K, std::vector<unsigned> & labels, int iters=100)
	{
		/*
			@params:
				data:  shape (num_samples, dimensionality), one data per row 
				K: # cluster
				label: num_samples long vector, 0-based index for each sample
		*/

		unsigned num_samples = data.rows(), dim = data.cols();

		// allocate space for cluster centroids
		Eigen::MatrixXf centers = Eigen::MatrixXf::Zero(K, dim);
		
		// to record number of samples in each cluster
		unsigned *cnt = (unsigned*)malloc(K * sizeof(unsigned));
		memset(cnt, 0, K * sizeof(unsigned));

		// traverse data to calculate updated centroids
		for (unsigned i = 0; i < num_samples; i++) {
			unsigned index = labels[i];
			cnt[index]++;
			centers.row(index) += data.row(i);
		}
		for (unsigned i = 0; i < K; i++) {
			centers.row(i) /= cnt[i];
		}
		delete cnt;
	
		while (iters--) {
			
			// assign each sample to its nearest cluster 
			for (unsigned i = 0; i < num_samples; i++) {
				double min_dist = DBL_MAX;
				unsigned cluster_index = 0;
				for (unsigned j = 0; j < K; j++) {
					double dis = (data.row(i) - centers.row(j)).squaredNorm();
					if (dis < min_dist) {
						min_dist = dis;
						cluster_index = j;
					}
				}
				labels[i] = cluster_index;
			}

			/*****  update centroids based on new clusters  *****/
			Eigen::MatrixXf sum_all = Eigen::MatrixXf::Zero(K, dim);
			
			// to record number of samples in each cluster
			unsigned *sum_num_clusters = (unsigned*)malloc(K * sizeof(unsigned));
			memset(sum_num_clusters, 0, K * sizeof(unsigned));
			
			// traverse data to calculate updated centroids
			for (unsigned j = 0; j < num_samples; j++) {
				unsigned index = labels[j];
				sum_num_clusters[index]++;
				sum_all.row(index) += data.row(j);
			}
			for (unsigned tmp_k = 0; tmp_k < K; tmp_k++)
				centers.row(tmp_k) = sum_all.row(tmp_k) / sum_num_clusters[tmp_k];
			
			delete sum_num_clusters;

		}

		return centers;
	}

	Eigen::MatrixXf opq::kmeans_customed_NOinit(Eigen::MatrixXf & data, unsigned K, std::vector<unsigned>& labels, int iters)
	{
		unsigned num_samples = data.rows(), dim = data.cols();
		// randomly generate initial centroids form data
		Eigen::MatrixXf centers = rand_center(data, K);

		while (iters--) {
			auto s = std::chrono::high_resolution_clock::now();
			
			// assign each sample to its nearest cluster 
			for (unsigned i = 0; i < num_samples; i++) {
				double min_dist = DBL_MAX;
				unsigned cluster_index = 0;

				for (unsigned j = 0; j < K; j++) {
					double dis = (data.row(i) - centers.row(j)).squaredNorm();
					
					if (dis < min_dist) {
						min_dist = dis;
						cluster_index = j;
					}
				}
				if (labels.size() < num_samples) {
					labels.push_back(cluster_index);
				}else
					labels[i] = cluster_index;
			}

			auto e = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = e - s;
			printf("iters # %d. allocate index uses %f seconds\n", iters, time);


			
			/*****  update centroids based on new clusters  *****/
			auto s1 = std::chrono::high_resolution_clock::now();

			Eigen::MatrixXf sum_all = Eigen::MatrixXf::Zero(K, dim);
			
			// to record number of samples in each cluster
			unsigned *sum_num_clusters = (unsigned*)malloc(K * sizeof(unsigned));
			memset(sum_num_clusters, 0, K * sizeof(unsigned));

			// traverse data to calculate updated centroids
			for (unsigned j = 0; j < num_samples; j++) {
				unsigned index = labels[j];
				sum_num_clusters[index]++;
				sum_all.row(index) += data.row(j);
			}
			for (unsigned tmp_k = 0; tmp_k < K; tmp_k++)
				centers.row(tmp_k) = sum_all.row(tmp_k) / sum_num_clusters[tmp_k];
			
			delete sum_num_clusters;

			auto e1 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time1 = e1 - s1;
			printf("iters # %d. compute centroids uses %f seconds\n",iters, time1);
		}

		return centers;
	}

	std::vector<unsigned> aknnspace::opq::balanced_partition(Eigen::MatrixXf vals, unsigned M)
	{
		//Devide vals into M "balanced" subspaces

		// size of val, i.e. dimensionality of data
		unsigned dim = vals.size(); 
		// maximum number of eigenvectors of subspace
		unsigned dim_subspace = dim / M; 
		// to record index of eigen vector in each subspace
		std::vector<std::vector<unsigned>> dim_tables(M); 

		// balance the product of eigenvalues of the subspaces, i.e., the sum of log(eigenvalues)
		Eigen::MatrixXf fvals = (vals.array() - (1e-20)).log().matrix();
		
		// all positive
		fvals = (fvals.array() - fvals.minCoeff() + 1).matrix();

		// to record sum of eigen values in each subspace
		std::vector<unsigned> sum_list(M, 0);
		
		// starting subspace
		unsigned current_subspaceIdx = 1;
		for (unsigned d = 0; d < dim; d++) {
			dim_tables[current_subspaceIdx].push_back(d);
			sum_list[current_subspaceIdx] += fvals(d);
			if (dim_tables[current_subspaceIdx].size() == dim_subspace)
				sum_list[current_subspaceIdx] = 1e10;

			// search a smaller subspace to evenly fill each subspace with eigenvalues (for M is short, just enumerate to get min)
			for (unsigned tmp_i = 0; tmp_i < M; tmp_i++) {
				if (sum_list[tmp_i] < sum_list[current_subspaceIdx])
					current_subspaceIdx = tmp_i;
			}
		}

		// flatten the result
		std::vector<unsigned> res;
		for (unsigned tmp_i = 0; tmp_i < dim_tables.size(); tmp_i++) {
			for (unsigned tmp_j = 0; tmp_j < dim_tables[tmp_i].size(); tmp_j++) {
				res.push_back(dim_tables[tmp_i][tmp_j]);
			}
		}
			
		return res;
	}

	Eigen::MatrixXf opq::eigen_allocation_NOdecomposition(Eigen::MatrixXf evecs, Eigen::MatrixXf evals, unsigned M, unsigned dim)
	{
		// re-order the eigenvectors
		auto dim_ordered = balanced_partition(evals, M);		
		Eigen::MatrixXf R(dim, dim);
		for (unsigned i = 0; i < dim; i++) {
			unsigned vec_index = dim_ordered[i];
			R.col(i) = evecs.col(vec_index);
		}

		return R;
	}

	Eigen::MatrixXf aknnspace::opq::eigenvalue_allcation(Eigen::MatrixXf &X, unsigned M)
	{
		/*
					the order of eigen vectors is not the same as
					the result from Matlab code.
					thus, partition cannot guarantee the ultimate R
					is the same as the R from Matlab code.
					but the eigen value form this (C++) code is quite similar
					to those from Matlab code, the difference may come from
					details of the decomposition implementation or accuracy of digits...
		*/

		unsigned n = X.rows();
		unsigned dim = X.cols();

		// calculate covariance matrix
		Eigen::MatrixXf covX = covariance(X);

		// eigen decomposition
		Eigen::EigenSolver<Eigen::MatrixXf> es(covX);
		Eigen::MatrixXf evecs = es.eigenvectors().real();
		Eigen::MatrixXf evals = es.eigenvalues().real();

		// re-order the eigenvectors
		auto dim_ordered = balanced_partition(evals, M);		
		Eigen::MatrixXf R(dim, dim);
		for (unsigned i = 0; i < dim; i++) {

			unsigned vec_index = dim_ordered[i];
			R.col(i) = evecs.col(vec_index);
		}

		return R;
	}

	float aknnspace::opq::sqdist(Eigen::MatrixXf &data, Eigen::MatrixXf &centroids, std::vector<unsigned> idx)
	{
		float dist = 0;
		unsigned num_samples = data.rows();
		
		// compute distance from its nearest centroids
		for (unsigned tmp_i = 0; tmp_i < num_samples; tmp_i++) {
			unsigned centroid_id = idx[tmp_i];
			dist = (data.row(tmp_i) - centroids.row(centroid_id)).squaredNorm();
		}
		return dist;
	}

	float aknnspace::opq::train_pq(Eigen::MatrixXf &X, unsigned M, unsigned num_iter,
		std::vector<Eigen::MatrixXf> &centers_table, std::vector<std::vector<unsigned>> &idx_table)
	{
		// number of centers per subspaces (fixed 8 bits)
		unsigned k = 256; 
		// dimension of raw data samples
		unsigned dim = X.cols(); 
		//dimesion per subspace
		unsigned d = dim / M; 

		float distortion = 0;
		for (unsigned m = 0; m < M; m++) {

			// preparations for kmeans
			// data for subspace (i,j,p,q). one row per sample, shape (num-samples, sub-dimension)
			Eigen::MatrixXf Xsub = X.block(0, m*d, X.rows(), d); 

			// kmeans on part of data
			centers_table.push_back(kmeans_customed_NOinit(Xsub, k, idx_table[m], num_iter));

			// compute distortion
			float newdist = sqdist(Xsub, centers_table[m], idx_table[m]);
			distortion += newdist;

		}
		return distortion;
	}

	float aknnspace::opq::train_opq_np(Eigen::MatrixXf &X, unsigned M, std::vector<std::vector<unsigned>> &idx_table,
		std::vector<Eigen::MatrixXf> &centers_table, Eigen::MatrixXf R_init,
		unsigned num_iter_outer, unsigned num_iter_inner)
	{
		// number of centers per subspaces (fixed 8 bits)
		unsigned k = 256; 
		// dimension of raw data samples
		unsigned dim = X.cols(); 
		//dimesion per subspace
		unsigned d = dim / M; 

		float distortion = 0;
		for (unsigned iter_outer = 0; iter_outer < num_iter_outer; iter_outer++) {
			float distort = 0;

			// project data with rotation matrix
			Eigen::MatrixXf Xproj = X*R_init;

			Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(X.rows(), X.cols());

			// encode data for all subspace
			for (unsigned m = 0; m < M; m++) {

				// preparations for kmeans
				// data for subspace (i,j,p,q). one row per sample, shape (num-samples, sub-dimension)
				Eigen::MatrixXf Xsub = Xproj.block(0, m*d, X.rows(), d);
				
				// run kmeans on part of data
				Eigen::MatrixXf centers =  kmeans_customed(Xsub, k, idx_table[m], num_iter_inner);
				
				// compute distortion
				float dist = sqdist(Xsub, centers, idx_table[m]);
				distort += dist;

				for (unsigned tmp_i = 0; tmp_i < idx_table[m].size(); tmp_i++) {
					unsigned tmp_id = idx_table[m][tmp_i];
					Y.block(tmp_i, m*d, 1, d) = centers.row(tmp_id);
				}
			}
			
			// SVD decomposition using function from Eigen library
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(X.transpose()*Y, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::MatrixXf V = svd.matrixV(), U = svd.matrixU();
			R_init = U * V.transpose();
			printf("inner loop # %d \n, distortion: %f", iter_outer + 1, distort);

			distortion += distort;
		}
		
		return distortion;
	}

}


