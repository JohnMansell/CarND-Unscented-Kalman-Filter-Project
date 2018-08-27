#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}


//=================================
//    RMSE		-- Lesson 5 :: 23
//=================================
	VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
								  const vector<VectorXd> &ground_truth) {

		// Initialize
			VectorXd rmse = VectorXd(4);
			rmse << 0, 0, 0, 0;

		// Check validity of inputs:
			if ( ( estimations.size() != ground_truth.size() ) || ( estimations.empty()) )
			{
				std::cout << "Invalid estimation or ground_truth date " << std::endl;
				return rmse;
			}

		// Accumulate squared residuals
			for (unsigned int i = 0; i < estimations.size(); i++)
			{
				VectorXd residual = estimations[i] - ground_truth[i];

				// Coefficient-wise multiplication
					residual = residual.array() * residual.array();
					rmse += residual;
			}

		// Mean
			rmse = rmse / estimations.size();

		// Square root
			rmse = rmse.array().sqrt();

		// Return
			return rmse;
	}