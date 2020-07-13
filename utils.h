#pragma once
#include<vector>
namespace aknnspace {
	class utils
	{
	public:
		utils();
		~utils();

		void load_ivecs(const char *filename, unsigned*& data);
		void load_fvecs(const char *filename, float* &data, unsigned &dimension, unsigned &number);

	};

}

