#define _USE_MATH_DEFINES
#include <librealsense2/rs.hpp> 
#include <librealsense2/hpp/rs_export.hpp>
#include "example.hpp"         
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>  
#include <conio.h>
#include <mutex>
#include "example-imgui.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/ply_io.h>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric>
using namespace std;
using namespace cv;
using namespace rs2;
using namespace Eigen;

struct state 
{ 
	double yaw, pitch, last_x, last_y;
	bool ml;
	float offset_x, offset_y;
	texture tex;
};

struct calibration_state
{
	
	float pitch;
	float yaw;
	float roll;
	MatrixXd Transform;
};
struct dimensions
{
	float height;
	float width;
	float depth;
	float volume;
};

void register_glfw_callbacks(window& app, glfw_state& app_state);


/*Function to transform a set of points given a 3d transformation matrix.
* Input:
*   -T: 3d transformation matrix.
*   -points: points to transform.
* Output:
*   -transformedPoints: the transformed points.
*/
MatrixXd transformPoints(const Matrix4d& T, const MatrixXd& points)
{
	// Assemble the points into a matrix of 4D homogeneous points
	MatrixXd homogeneousPoints(points.rows(), 4);
	homogeneousPoints.leftCols<3>() = points;
	homogeneousPoints.col(3).setOnes();

	// Multiply the points by the transformation matrix
	MatrixXd transformedPoints = homogeneousPoints * T.transpose();

	// Return the transformed points as a matrix of 3D points
	return transformedPoints.leftCols<3>();
}

/*Function to find the 3d transformation matrix between two sets of 3d points.
* Input:
*   -points1: the first set of points.
*   -points2: the second set of points.
* Output:
*   -T: the transformation matrix.
*/
MatrixXd findTransformation(const MatrixXd& points1, const MatrixXd& points2)
{
	// Check if the input matrices have the same number of rows and columns
	if (points1.rows() != points2.rows() || points1.cols() != points2.cols())
	{
		throw std::invalid_argument("Input matrices have different sizes");
	}

	// Compute the centroids of the two sets of points
	Vector3d centroid1 = points1.colwise().mean();
	Vector3d centroid2 = points2.colwise().mean();

	// Subtract the centroids from the points
	MatrixXd points1_centered = points1.rowwise() - centroid1.transpose();
	MatrixXd points2_centered = points2.rowwise() - centroid2.transpose();

	// Compute the covariance matrix
	Matrix3d cov = points2_centered.transpose() * points1_centered;

	// Compute the SVD of the covariance matrix
	JacobiSVD<Matrix3d> svd(cov, ComputeFullU | ComputeFullV);

	// Compute the rotation matrix
	Matrix3d rotation = (svd.matrixV() * svd.matrixU().transpose()).inverse();

	// Compute the translation vector
	Vector3d translation = centroid2 - rotation * centroid1;

	// Assemble the transformation matrix
	Matrix4d T;
	T.block<3, 3>(0, 0) = rotation;
	T.block<3, 1>(0, 3) = translation;
	T.row(3) << 0, 0, 0, 1;

	return T;
}


MatrixXd generate_board(int rows, int cols, double dist, bool display, bool reverse, double height)
{
	double cur_x;
	double cur_y;
	MatrixXd objective_pts(cols * rows, 3);
	int cur = 0;
	if (reverse)
	{
		cur_x = 0;
		cur_y = (rows - 1) * dist;
	}
	else
	{
		cur_x = (cols - 1) * dist;
		cur_y = 0.0;
	}

	for (int i = 0; i < cols; i++)
	{
		if (reverse)
		{
			cur_y = (rows - 1) * dist;
		}
		else
		{
			cur_y = 0.0;
		}

		for (int j = 0; j < rows; j++)
		{
			objective_pts(cur, 0) = cur_x;
			objective_pts(cur, 1) = cur_y;
			objective_pts(cur, 2) = height;
			if (display)
			{
				cout << "[" << cur_x << "," << cur_y << ", 1.0],";
			}
			if (reverse)
			{
				cur_y -= dist;
			}
			else
			{
				cur_y += dist;
			}

			cur += 1;
		}
		if (reverse)
		{
			cur_x += dist;
		}
		else
		{
			cur_x -= dist;
		}

	}

	if (display)
	{
		cout << endl;
	}
	return(objective_pts);
}

calibration_state calibrateCamera(Mat img, frameset frames, int verbose, bool save)
{
	if (verbose == 5)
	{
		cout << "-------------------Calibrating-Camera-------------------" << endl;
	}
	int numCornersHor = 6;
	int numCornersVer = 8;
	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	vector<Point2d> corners;
	rs2::align align_to_color(RS2_STREAM_COLOR);
	colorizer c;
	bool found = false;
	while (found != true)
	{
		found = findChessboardCorners(img, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
	}
	if (verbose == 5)
	{
		cout << "Found chessboard corners" << endl;
	}
	float upoint[3];
	float upixel[2];
	vector<int> remove_rows;
	vector<double> ptsx;
	vector<double> ptsy;
	vector<double> ptsz;
	MatrixXd objective_pts_temp(6 * 8, 3);
	vector <double> corner_points;

	//Aligning depth and colour frame 
	frames = align_to_color.process(frames);
	auto color = frames.get_color_frame();
	auto depth = frames.get_depth_frame();
	auto colorized_depth = c.colorize(depth);
	const int w = colorized_depth.as<rs2::video_frame>().get_width();
	const int h = colorized_depth.as<rs2::video_frame>().get_height();

	//Converting depth and colour frames to opencv images
	Mat image(Size(w, h), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
	Mat image2(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

	//getting camera intrinisic              
	rs2_intrinsics intr = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

	//visualizing points found and getting 3d coordinates of points found
	for (int i = 0; i < corners.size(); i++)
	{
		circle(image, corners.at(i), 5, (0, 0, i * 10), 5);
		circle(image2, corners.at(i), 5, (0, 0, i * 10), 5);
		upixel[0] = corners.at(i).x;
		upixel[1] = corners.at(i).y;
		if (depth.get_distance(corners.at(i).x, corners.at(i).y) != 0.0)
		{
			rs2_deproject_pixel_to_point(upoint, &intr, upixel, depth.get_distance(corners.at(i).x, corners.at(i).y));
			ptsx.push_back(upoint[0]);
			ptsy.push_back(upoint[1]);
			ptsz.push_back(upoint[2]);
			remove_rows.push_back(i);
		}

	}

	if (verbose == 5)
	{
		cout << "Found 3D coordinates of chessboard corners from camera's perspective." << endl;
	}

	//dealing with the ordering of points found on chessboard
	if (corners.at(0).x > corners.at(20).x)
	{

		objective_pts_temp = generate_board(6, 8, 0.025, false, false, 0.53);
	}
	else
	{

		objective_pts_temp = generate_board(6, 8, 0.025, false, true, 0.53);
	}

	//dealing with points with missing depth data
	MatrixXd objective_pts(ptsx.size(), 3);
	MatrixXd found_pts(ptsx.size(), 3);
	for (int i = 0; i < ptsx.size(); i++)
	{
		objective_pts(i, 0) = objective_pts_temp(remove_rows.at(i), 0);
		objective_pts(i, 1) = objective_pts_temp(remove_rows.at(i), 1);
		objective_pts(i, 2) = objective_pts_temp(remove_rows.at(i), 2);
		found_pts(i, 0) = ptsx.at(i);
		found_pts(i, 1) = ptsy.at(i);
		found_pts(i, 2) = ptsz.at(i);
	}

	//Finding transformation matrix
	MatrixXd R = findTransformation(found_pts, objective_pts);
	if (verbose == 5)
	{
		cout << "Calculated transformation matrix." << endl;
	}

	//Checking transformation
	MatrixXd validation_pts = transformPoints(R, found_pts);

	//Converting transformation matrix to angle of pitch, yaw and roll
	double pitch = atan2(-R(2, 0), sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0))) * (180.0 / M_PI);
	double yaw = atan2(R(1, 0), R(0, 0)) * (180 / M_PI);
	double roll = atan2(R(2, 1), R(2, 2)) * (180 / M_PI);
	//Finding mean-squared error or transformation
	double MSE = 0;
	for (int i = 0; i < ptsx.size(); i++) {
		Eigen::Vector3d diff = objective_pts.row(i) - validation_pts.row(i);
		MSE += diff.squaredNorm();
	}
	MSE /= found_pts.rows();
	if (verbose > 0)
	{
		cout << "Transformation matrix: " << endl;
		cout << R << endl;
		cout << "yaw:" << yaw << endl;
		cout << "pitch:" << pitch << endl;
		cout << "roll:" << roll << endl;
		std::cout << "MSE: " << MSE << std::endl;
	}
	if (verbose == 5)
	{
		cout << "--------------------------------------------------------" << endl << endl;
		cout << "Found points and validation points in python syntax:" << endl;
	}

	//printing out the found points and the transformed points
	if (verbose >= 3)
	{
		cout << "Found points:" << endl;
		for (int x = 0; x < ptsx.size(); x++)
		{
			if (x != ptsx.size() - 1)
			{
				cout << "[" << found_pts(x, 0) << "," << found_pts(x, 1) << "," << found_pts(x, 2) << "],";
			}
			else
			{
				cout << "[" << found_pts(x, 0) << "," << found_pts(x, 1) << "," << found_pts(x, 2) << "]";
			}
			
		}
		cout << endl << endl << "Validation points:" << endl;
		for (int x = 0; x < ptsx.size(); x++)
		{
			if (x != ptsx.size() - 1)
			{
				cout << "[" << validation_pts(x, 0) << "," << validation_pts(x, 1) << "," << validation_pts(x, 2) << "],";
			}
			else
			{
				cout << "[" << validation_pts(x, 0) << "," << validation_pts(x, 1) << "," << validation_pts(x, 2) << "]";
			}
			
		}

		cout << endl << "--------------------------------------------------------" << endl;
	}
	if (save)
	{
		std::ofstream file("calibration.txt");
		if (file.is_open())
		{
			file << R;
		}
	}

	//displaying depth and color images with chessboard corners found
	if (verbose >= 2)
	{
		imshow("depth", image);
		imshow("color", image2);
	}
	calibration_state output;
	output.Transform = R;
	output.pitch = pitch;
	output.yaw = yaw;
	output.roll = roll;
	return output;
}

calibration_state loadCalibrationFromFile(string filename)
{
	std::ifstream in(filename);
	Eigen::MatrixXd R;
	in >> R;
	cout << "e" << endl;
	in.close();
	double pitch = atan2(-R(2, 0), sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0))) * (180.0 / M_PI);
	double yaw = atan2(R(1, 0), R(0, 0)) * (180 / M_PI);
	double roll = atan2(R(2, 1), R(2, 2)) * (180 / M_PI);
	calibration_state output;
	output.Transform = R;
	output.pitch = pitch;
	output.yaw = yaw;
	output.roll = roll;
}

float distanceSensor(MatrixXd points, int num_pts, float dist_to_camera)
{
	vector<float> all_pointsy;
	for (int i = 0; i < num_pts; i++)
	{
		if (points(i, 2) > 0.1)
		{
			all_pointsy.push_back(points(i, 1));
		}
	}
	float closest_point = dist_to_camera - *max_element(all_pointsy.begin(), all_pointsy.end();
	return(closest_point)
}

dimensions calculate_dims(MatrixXd points, float lg_height, float lg_y, float lg_width, float pallet_max_width, float pallet_max_length, int num_pts)
{
	vector<float> load_guard_pointsx;
	vector<float> load_guard_pointsy;
	vector<float> load_guard_pointsz;
	vector<float> pallet_pointsx;
	vector<float> pallet_pointsy;
	vector<float> pallet_pointsz;
	vector<float> all_pointsx;
	vector<float> all_pointsy;
	vector<float> all_pointsz;	
	for (int i = 0; i < num_pts; i++)
	{
		all_pointsx.push_back(points(i, 0));
		all_pointsy.push_back(points(i, 1));
		all_pointsz.push_back(points(i, 2));
		//seperating from floor (correct)
		if (points(i, 2) > 0.2)
		{
			//seperating points in distance
			if (points(i, 0) < pallet_max_width / 2 && points(i, 0) > -1 * (pallet_max_width / 2) && points(i, 1) < pallet_max_length / 2 && points(i, 1) > -1 * (pallet_max_length / 2) && points(i, 1) < lg_y+lg_width)
			{
				//adding points to load guard
				if (points(i, 1) > lg_y)
				{
					load_guard_pointsx.push_back(points(i, 0));
					load_guard_pointsy.push_back(points(i, 1));
					load_guard_pointsz.push_back(points(i, 2));
				}

				//adding points to pallet
				else
				{
					pallet_pointsx.push_back(points(i, 0));
					pallet_pointsy.push_back(points(i, 1));
					pallet_pointsz.push_back(points(i, 2));
					
				}
			}
		}		
	}
	dimensions dims;
	dims.width = float(*max_element(pallet_pointsx.begin(), pallet_pointsx.end()) - *min_element(pallet_pointsx.begin(), pallet_pointsx.end()));
	dims.depth = float(*max_element(pallet_pointsy.begin(), pallet_pointsy.end()) - *min_element(pallet_pointsy.begin(), pallet_pointsy.end()));
	dims.height = lg_height - (*max_element(load_guard_pointsz.begin(), load_guard_pointsz.end()) - (*max_element(pallet_pointsz.begin(), pallet_pointsz.end())));
	dims.volume = dims.width * dims.depth * dims.height;
	cout << "Width: " << dims.width << endl;
	cout << "Depth: " << dims.depth << endl;
	cout << "Height: " << dims.height << endl;
	cout << "Volume: " << dims.volume << endl;
	return(dims);
}

int main(int argc, char* argv[]) try
{

	window app(1280, 720, "RealSense Pointcloud Example");
	glfw_state app_state;
	pointcloud pc;
	points points;
	pipeline pipe;
	ImGui_ImplGlfw_Init(app, false);
	colorizer c;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_DEPTH);
	cfg.enable_stream(RS2_STREAM_COLOR);
	rs2::align align_to_color(RS2_STREAM_COLOR);
	Mat gray_image;
	int numCornersHor = 6;
	int numCornersVer = 8;
	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	vector<vector<Point3d>> object_points;
	vector<vector<Point2d>> image_points;
	vector<Point2d> corners;
	calibration_state cal_state;
	auto profile = pipe.start(cfg);

	while (app)
	{
		register_glfw_callbacks(app, app_state);
		auto frames = pipe.wait_for_frames();
		auto color = frames.get_color_frame();
		if (!color)
			color = frames.get_infrared_frame();

		auto depth = frames.get_depth_frame();

		const int w = color.as<rs2::video_frame>().get_width();
		const int h = color.as<rs2::video_frame>().get_height();
		pc.map_to(color);
		points = pc.calculate(depth);
		app_state.tex.upload(color);
		draw_pointcloud(app.width(), app.height(), app_state, points);
		Mat image(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
		cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);
		imshow("Image", gray_image);

		if (_kbhit()) {
			char key = _getch();
			if (key == 'c')
			{
				cal_state = calibrateCamera(gray_image, frames, 1, true);
			}
			else if (key == 'd')
			{
				//Aligning depth and colour frame 
				frames = align_to_color.process(frames);
				auto color = frames.get_color_frame();
				auto depth = frames.get_depth_frame();
				auto colorized_depth = c.colorize(depth);
				const int w = colorized_depth.as<rs2::video_frame>().get_width();
				const int h = colorized_depth.as<rs2::video_frame>().get_height();

				//Converting depth and colour frames to opencv images
				Mat image(Size(w, h), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
				Mat image2(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

				//getting camera intrinisic              
				rs2_intrinsics intr = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
				float upoint[3];
				float upixel[2];
				vector<int> remove_rows;
				vector<double> ptsx;
				vector<double> ptsy;
				vector<double> ptsz;
				int total = 0;

				for (int i = 0; i < depth.get_width(); i++)
				{
					for (int j = 0; j < depth.get_height(); j++)
					{
						upixel[0] = i;
						upixel[1] = j;
						if (depth.get_distance(i, j) != 0.0)
						{
							rs2_deproject_pixel_to_point(upoint, &intr, upixel, depth.get_distance(i, j));
							ptsx.push_back(upoint[0]);
							ptsy.push_back(upoint[1]);
							ptsz.push_back(upoint[2]);
							total += 1;
						}
					}
				}
				//converting point cloud points to a matrix
				MatrixXd point_cloud_mat(total, 3);
				MatrixXd transformed_point_cloud(total, 3);
				int cur = 0;
				for (int i = 0; i < total; i++)
				{
					point_cloud_mat(i, 0) = ptsx[i];
					point_cloud_mat(i, 1) = ptsy[i];
					point_cloud_mat(i, 2) = ptsz[i];
				}
				//transforming entire pointcloud
				transformed_point_cloud = transformPoints(cal_state.Transform, point_cloud_mat);
				dimensions dims = calculate_dims(transformed_point_cloud, 0.46, 0.52, 0.05, 1.5, 1.5, total);
			}											
		}
	}
	return 1;
}

catch (const rs2::error& e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
