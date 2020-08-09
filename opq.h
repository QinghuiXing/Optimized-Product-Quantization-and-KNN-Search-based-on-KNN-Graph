#pragma once
#include<vector>
#include<Eigen/Dense>


namespace aknnspace {
	class opq
	{
	public:
		opq(unsigned num_iter, unsigned k, unsigned num_subspace, unsigned dimension,unsigned num_samples, float* base_data);
		opq();
		~opq();

		unsigned m_num_iter; // iters for k-means 
		unsigned m_dimension; //dimension of original data space
		unsigned m_num_subspace; // # of subspace = code_length/log2(k) = (4, 8, 16)
		unsigned m_k; //  k =256
		unsigned m_num_samples; // # of samples total (10000 in sift for test)
		
		Eigen::MatrixXf Xtrain; // data container
		std::vector<Eigen::MatrixXf> centers_table; // each Matrix represents data of centroids in subapce
		std::vector<std::vector<unsigned>> idx_table; // each subvector represents index of cluster in subspace

		void reconstruct(unsigned new_num_subspace);
		/*
			for DEBUG
		*/


		Eigen::MatrixXf covariance(Eigen::MatrixXf &X);
		/*
			@brief: calculate covariance matrix of X
			@params:
				X: data container
				RETURN: convariance of X
		*/

		Eigen::MatrixXf rand_center(Eigen::MatrixXf &data, unsigned k);
		/*
			@brief: randomly generate k centroids within data's range, same dimension
			@params:
				data: data to cluster
				k: number of centroids to generate
				RETURN: K centers of data
		*/


		Eigen::MatrixXf kmeans_customed(Eigen::MatrixXf & data, unsigned K, std::vector<unsigned> &labels, int iters);
		/*
			@brief: running kmeans on data; self-implemented to run on Eigen::MatrixXf,
					without optimization so the speed is not very fast compared to Matlab or Opencv. 
			@params:
				data: data to cluster
				K: number of centroids
				labels: (Input/Output)container for cluster index of each data, need to contain initial cluster label of
						the data. see kmeans_customed_NOinit() for kmeans without initialization,  
				iters: number of iteration for kmeans
				RETURN: K centers of data
		*/


		Eigen::MatrixXf kmeans_customed_NOinit(Eigen::MatrixXf & data, unsigned K, std::vector<unsigned> &labels, int iters);
		/*
			@brief: running kmeans on data; self-implemented to run on Eigen::MatrixXf,
					without optimization so the speed is not very fast compared to Matlab or Opencv. 
			@params:
				data: data to cluster
				K: number of centroids
				labels: (Input/Output)container for cluster index of each data, no need to contain initial cluster label of
						the data. see kmeans_customed() for kmeans with initialization
				iters: number of iteration for kmeans
				RETURN: K centers of data
		*/


		std::vector<unsigned> balanced_partition(Eigen::MatrixXf vals, unsigned M);
		/*
			@brief: reorder the eigen vector based on the sum of log() results of eigenvalue
					to make value in each bin close to each other
			@params:
				vals: eigenvalues
				M: number of subspace
				RETURN: new order of eigen-vectors
		*/


		Eigen::MatrixXf eigen_allocation_NOdecomposition(Eigen::MatrixXf evecs, Eigen::MatrixXf evals, unsigned M, unsigned dim);
		/*
			@brief: reorder the eigen vector based on the sum of log() results of eigenvalue
					to make value in each bin close to each other
			@params:
				evecs: eigen-vectors
				evals: eigen-values
				M: number of subspace
				dim: dimensionality of data
				RETURN: reordered eigen-vectors, i.e. matrix R in non-parameters-OPQ
		*/


		Eigen::MatrixXf eigenvalue_allcation(Eigen::MatrixXf &X, unsigned M);

		/*
			@brief: reorder the eigen vector based on the sum of log() results of eigenvalue
					to make value in each bin close to each other. No need to pre-calculate 
					corvariance of data matrix X
			@params:
				X: data matrix
				M: num of subspace
				RETURN: reordered eigen-vectors, i.e. matrix R in non-parameters-OPQ
		*/


		float sqdist(Eigen::MatrixXf &data, Eigen::MatrixXf &centroids, std::vector<unsigned> idx);
		/*
			@brief: compute distortion
			@params:
				data: data matrix
				centroids: data of centers in all subspace
				idx: index of data in all subspace
				RETURN: overall distortion
		*/


		float train_pq(Eigen::MatrixXf &X, unsigned M, unsigned num_iter, std::vector<Eigen::MatrixXf> &centers_table, std::vector<std::vector<unsigned>> &idx_table);
		/*
			@brief: parameter-OPQ, see "Optimized Product Quantization" for more details
			@params:
				X: data matrix
				M: num of subspace
				num_iter: num of iterations for kmeans
				centers_table: container of data of centers in all subspace
				idx_table: container of index of data in all subspace
				RETURN: distortion of parameter-OPQ method
		*/


		float train_opq_np(Eigen::MatrixXf &X, unsigned M, std::vector<std::vector<unsigned>> &idx_table,
			std::vector<Eigen::MatrixXf> &centers_table, Eigen::MatrixXf R_init,
			unsigned num_iter_outer, unsigned num_iter_inner);
		/*
			@brief: non-parameters-OPQ, see "Optimized Product Quantization" for more details
			@params:
				X: data matrix
				M: num of subspace
				idx_table: container of index of data in all subspace
				centers_table: container of data of centers in all subspace
				R_init: matrix R in non-parameters-OPQ
				num_iter_outer: overall iterations
				num_iter_inner: iterations of kmeans, usually set it to 1
				RETURN: distortion of non-parameters-OPQ
		*/

	};
}
