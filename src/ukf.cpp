#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.3;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	// initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// time when the state is true, in us
	time_us_ = 0.0;

	// state dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = 7;

	// Sigma point spreading parameter
	lambda_ = 3 - n_aug_;

	// predicted sigma points matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	//create vector for weights
	weights_ = VectorXd(2 * n_aug_ + 1);

	// the current NIS for radar
	NIS_radar_ = 0.0;

	// the current NIS for laser
	NIS_laser_ = 0.0;

	// Set Weights
		double weight_0 = lambda_ / (lambda_ + n_aug_);
		weights_(0) = weight_0;

		for (int i=1; i < 2 * n_aug_ + 1; i++)
		{
			double weight = 0.5 / (n_aug_ + lambda_);
			weights_(i) = weight;
		}

}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
//=================================
//    Initialize Measurement
//=================================
bool UKF::InitializeMeasurement(MeasurementPackage meas_package){

	// First Measurement		-- [px, py, v, psi, psi_dot]
		x_ << 0, 0, 0, 0, 0;

	// Covariance matrix
		P_ << 0.15,    0, 0, 0, 0,
				0,  0.15, 0, 0, 0,
				0,     0, 1, 0, 0,
				0,     0, 0, 1, 0,
				0,     0, 0, 0, 1;

	// Init timestamp
		time_us_ = meas_package.timestamp_;

	// LASER
		if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
		}

	// RADAR
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			// Convert radar from polar to cartesian
			float ro     = meas_package.raw_measurements_(0);
			float phi    = meas_package.raw_measurements_(1);
			//float ro_dot = meas_package.raw_measurements_(2);

			// Initialize
			x_(0) = ro * cos(phi);
			x_(1) = ro * sin(phi);
		}

	return true;
}



//=================================
//    Process Measurements
//=================================
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	//-----------------------
	//		Initialize
	//-----------------------
		if (!is_initialized_)
		{
			is_initialized_ = InitializeMeasurement(meas_package);
			return;
		}

	//-----------------------
	//		Predict
	//-----------------------
		// Time
			double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
			time_us_ = meas_package.timestamp_;

		// Prediction
			Prediction(dt);

	//-----------------------
	//		Update
	//-----------------------

		// LASER
			if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
				UpdateLidar(meas_package);
			}

		// RADAR
			if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
				UpdateRadar(meas_package);
			}

}

//=================================
//    Prediction
//=================================
	void UKF::Prediction(double delta_t) {

	//----------------------------------------------------
	//  Augment Sigma Points		-- Lesson 7 :: 18
	//----------------------------------------------------

		// Sigma Point Matrix
			VectorXd x_aug = VectorXd(n_aug_);

		// Augmented State covarience
			MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

		// Sigma Point Matrix
			MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

		// Set lambda for augmented sigma points
			lambda_ = 3 - n_aug_;

		// Create augmented mean state
			x_aug.head(5) = x_;
			x_aug(5) = 0;
			x_aug(6) = 0;

		// Create augmented covarience matrix
			P_aug.fill(0.0);
			P_aug.topLeftCorner(5, 5) = P_;
			P_aug(5, 5) = std_a_ * std_a_;
			P_aug(6, 6) = std_yawdd_ * std_yawdd_;

		// Create a square root matrix
			MatrixXd L = P_aug.llt().matrixL();

		// Create augmented Sigma Points
			Xsig_aug.col(0) = x_aug;
			for (int i=0; i < n_aug_; i++)
			{
				Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
				Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
			}

	//---------------------------------------------------
	//  Predict Sigma Points		-- Lesson 7 :: 21
	//---------------------------------------------------

		for (int i=0; i < 2 * n_aug_ + 1; i++)
		{
			// Extract values
				const double p_x 		= Xsig_aug(0, i);
				const double p_y		= Xsig_aug(1, i);
				const double v			= Xsig_aug(2, i);
				const double yaw		= Xsig_aug(3, i);
				const double yawd		= Xsig_aug(4, i);
				const double nu_a		= Xsig_aug(5, i);
				const double nu_yawdd	= Xsig_aug(6, i);

			// Predict state values
				double px_p, py_p;

			// Check for division by Zero
				if (fabs(yawd) > 0.001)
				{
					px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
					py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
				}

				else
				{
					px_p = p_x + v * delta_t * cos(yaw);
					py_p = p_y + v * delta_t * sin(yaw);
				}

				double v_p 		= v;
				double yaw_p 	= yaw + yawd * delta_t;
				double yawd_p 	= yawd;

			// Add Noise
				px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
				py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
				v_p  = v_p + nu_a * delta_t;

				yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
				yawd_p = yawd_p + nu_yawdd * delta_t;

			// Write Predicted sigma points into right column
				Xsig_pred_(0, i) = px_p;
				Xsig_pred_(1, i) = py_p;
				Xsig_pred_(2, i) = v_p;
				Xsig_pred_(3, i) = yaw_p;
				Xsig_pred_(4, i) = yawd_p;

		}

	//------------------------------------------------------
	//  Predict Mean and Covariance		-- Lesson 7 :: 24
	//------------------------------------------------------

		// Predicted State Mean
			x_.fill(0.0);
			for (int i = 0; i < 2 * n_aug_ + 1; i++)
			{
				x_ = x_ + weights_(i) * Xsig_pred_.col(i);
			}

		// Predicted state Covariance Matrix
			P_.fill(0.0);
			for (int i=0; i < 2 * n_aug_ + 1; i++)
			{
				// State difference
					VectorXd x_diff = Xsig_pred_.col(i) - x_;

				// Normalize angle
					NormalzeAngle(x_diff(3));

					P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
			}
}


//================================
//    Update Lidar
//=================================
void UKF::UpdateLidar(MeasurementPackage meas_package) {

	// --- Lidar Update is Linear
	// --- Taken directly from Extended KF project

	// Get Measurements
		VectorXd z = meas_package.raw_measurements_;


	// Create a matrix for sigma points
		MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points into map space
		for (int i=0; i < 2 * n_aug_ + 1; i++)
		{
			// Extract Values
				double p_x = Xsig_pred_(0, i);
				double p_y = Xsig_pred_(1, i);

			// Map space
				Zsig(0, i) = p_x;
				Zsig(1, i) = p_y;
		}

	//mean predicted measurement
		VectorXd z_pred = VectorXd(n_z);
		z_pred.fill(0.0);
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			z_pred = z_pred + weights_(i) * Zsig.col(i);
		}

	//measurement covariance matrix S
		MatrixXd S = MatrixXd(n_z, n_z);
		S.fill(0.0);
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

			//residual
				VectorXd z_diff = Zsig.col(i) - z_pred;

				S = S + weights_(i) * z_diff * z_diff.transpose();
		}

	//add measurement noise covariance matrix
		MatrixXd R = MatrixXd(n_z, n_z);
		R << std_laspx_*std_laspx_, 0,
				0, std_laspy_*std_laspy_;
		S = S + R;
		S = S + R;

	//--------------------------------
	//		UKF Update for Laser
	//--------------------------------

	// Cross Corelation matrix
		MatrixXd Tc = MatrixXd(n_x_, n_z);
		Tc.fill(0.0);
		for (int i=0; i < 2 * n_aug_ + 1; i++)
		{
			// Residual
				VectorXd z_diff = Zsig.col(i) - z_pred;

			// State difference
				VectorXd x_diff = Xsig_pred_.col(i) - x_;

				Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}

	// Kalman Gain K
		MatrixXd K = Tc * S.inverse();

	// Residual
		VectorXd z_diff = z - z_pred;

	// NIS
		NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

	// Update
		x_ = x_ + K * z_diff;
		P_ = P_ - K * S * K.transpose();

}


//=================================
//    Update Radar
//=================================
void UKF::UpdateRadar(MeasurementPackage meas_package) {

	// --- Measurements are NOT linear
	// --- Use Unscented Kalman filter
	// --- Predict Sigma Points

	//----------------------
	//	Lesson 7 :: 27
	//----------------------

	// Get Measurements
		VectorXd z = meas_package.raw_measurements_;
		int n_z = 3; 	// rho, phi, rho_dot

	// Sigma Points Matrix
		MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points
		for (int i=0; i < 2 * n_aug_ + 1; i ++)
		{
			// Get values
				const double p_x = Xsig_pred_(0, i);
				const double p_y = Xsig_pred_(1, i);
				const double v   = Xsig_pred_(2, i);
				const double yaw = Xsig_pred_(3, i);

			// Velocity (x, y)
				const double v1 = cos(yaw) * v;
				const double v2 = sin(yaw) * v;

			// Measurement Model
				Zsig(0, i) = sqrt( (p_x*p_x) + (p_y*p_y) );								// rho
				Zsig(1, i) = atan2(p_y, p_x);											// phi
				Zsig(2, i) = ( (p_x * v1) + (p_y * v2) ) / sqrt(p_x*p_x + p_y*p_y);		// rho dot
		}

	// Mean predicted Measurement
		VectorXd z_pred = VectorXd(n_z);
		z_pred.fill(0.0);
		for (int i=0; i < 2 * n_aug_ + 1; i ++)
		{
			z_pred = z_pred + weights_(i) * Zsig.col(i);
		}

	// Measurement Covariance Matrix S
		MatrixXd S = MatrixXd(n_z, n_z);
		S.fill(0.0);
		for (int i=0; i < 2 * n_aug_ + 1; i++)
		{
			// Residual
				VectorXd z_diff = Zsig.col(i) - z_pred;

			// Angle Normalization
				NormalzeAngle(z_diff(1));

				S = S + weights_(i) * z_diff * z_diff.transpose();
		}

	// Measurement Noise Covariant matrix
		MatrixXd R = MatrixXd(n_z, n_z);
		R << std_radr_ * std_radr_ , 0, 0,
				0, (std_radphi_ * std_radphi_),0,
				0, 0, (std_radrd_ * std_radrd_);
		S = S + R;
	//-------------------------------------------------------
	//		UKF Update for Radar		-- Lesson 7 :: 30
	//-------------------------------------------------------

	// Cross Corelation Matrix -- Tc
		MatrixXd Tc = MatrixXd(n_x_, n_z);
		Tc.fill(0.0);

		for (int i=0; i < 2 * n_aug_ + 1; i++)
		{
			// Residual
				VectorXd z_diff = Zsig.col(i) - z_pred;

			// Angle normalization
				NormalzeAngle(z_diff(1));

			// State difference
				VectorXd x_diff = Xsig_pred_.col(i) - x_;

			// Angle Normalization
				NormalzeAngle(x_diff(3));

				Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}

	// Kalman Gain K
		MatrixXd K = Tc * S.inverse();

	// Residual
		VectorXd z_diff = z - z_pred;

	// Angle Normalization
		NormalzeAngle(z_diff(1));

	// NIS
		NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

	// Update
		x_ = x_ + K * z_diff;
		P_ = P_ - K*S*K.transpose();

}

//=================================
//    Normalize Angle
//=================================
void UKF::NormalzeAngle(double &phi) {

	while (phi >  M_PI) phi -= 2.0 * M_PI;
	while (phi < -M_PI) phi += 2.0 * M_PI;

}




































