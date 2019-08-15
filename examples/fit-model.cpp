/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "eos/core/Image.hpp"
#include "eos/core/Image_opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/cpp17/optional.hpp"

#include "eos/cz/io.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;


/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 * In addition to fit-model-simple, this example uses blendshapes, contour-
 * fitting, and can iterate the fitting.
 *
 * 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper.
 */
int main(int argc, char* argv[])
{
    string modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile,
        blendshapesfile, outputbasename, init_pca_shape_coeffs_fp, init_expression_coeffs_fp, save_mode, name_3dmm;
	bool save_obj, save_wireframe, save_texture, fix_pca_shape_coeffs, fix_expression_coeffs, define_path_manual;
	int num_fit_iter, texture_resolution;
    float landmark_scale;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<string>(&modelfile)->required()->default_value("./bfm2017/bfm2017-1_bfm_nomouth.bin"),//"../share/sfm_shape_3448.bin"),
                "a Morphable Model stored as cereal BinaryArchive")
            ("image,i", po::value<string>(&imagefile)->required()->default_value("./data/image_0010.png"),
                "an input image")
            ("landmarks,l", po::value<string>(&landmarksfile)->required()->default_value("./data/image_0010.pts"),
                "2D landmarks for the image, in ibug .pts format")
            ("mapping,p", po::value<string>(&mappingsfile)->required()->default_value("./bfm2017/ibug_to_bfm2017-1_bfm_nomouth.txt"),//"../share/ibug_to_sfm.txt"),
                "landmark identifier to model vertex number mapping")
            ("model-contour,c", po::value<string>(&contourfile)->required()->default_value("./bfm2017/bfm2017-1_bfm_nomouth_model_contours.json"),//"../share/sfm_model_contours.json"),
                "file with model contour indices")
            ("edge-topology,e", po::value<string>(&edgetopologyfile)->required()->default_value("./bfm2017/bfm2017-1_bfm_nomouth_edge_topology.json"),//"../share/sfm_3448_edge_topology.json"),
                "file with model's precomputed edge topology")
            ("blendshapes,b", po::value<string>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
                "file with blendshapes")
			("num-fit-iter,n", po::value<int>(&num_fit_iter)->required()->default_value(5),
				"number of fitting iterations")
            ("output,o", po::value<string>(&outputbasename)->required()->default_value("./data/image_0010.out"),
                "basename for the output rendering and obj files")
			("save-mode,s", po::value<std::string>(&save_mode)->required()->default_value("minimum"),
				"save mode, available options are \"none\", \"minimum\", \"core\", \"all\"")
			("save-obj", po::value<bool>(&save_obj)->required()->default_value(true),
				"whether save obj file")
			("save-wireframe", po::value<bool>(&save_wireframe)->required()->default_value(true),
				"whether save wireframe image")
			("save-texture", po::value<bool>(&save_texture)->required()->default_value(true),
				"whether to save texture image")
			("init-pca-shape-coeffs-fp", po::value<string>(&init_pca_shape_coeffs_fp)->required()->default_value(""),
				"file path to initial pca shape coefficients")
			("init-expression-coeffs-fp", po::value<string>(&init_expression_coeffs_fp)->required()->default_value(""),
				"file path to initial expression coefficients")
			("fix-pca-shape-coeffs", po::value<bool>(&fix_pca_shape_coeffs)->required()->default_value(false),
				"whether to fix pca shape coefficients")
			("fix-expression-coeffs", po::value<bool>(&fix_expression_coeffs)->required()->default_value(false),
				"whether to fix expression coefficients")
			("resolution,r", po::value<int>(&texture_resolution)->required()->default_value(2048),
				"texture resolution")
			("landmark-scale", po::value<float>(&landmark_scale)->required()->default_value(1),
				"the rescale factor of landmark coordinates")
			("name-3dmm", po::value<string>(&name_3dmm)->required()->default_value("bfm2017"),
				"name of 3dmm")
			("define-path-manual", po::value<bool>(&define_path_manual)->required()->default_value(true),
				"whether to save texture image");
		// clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: fit-model [options]" << endl;
            cout << desc;
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    } catch (const po::error& e)
    {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        return EXIT_FAILURE;
    }

	// determine paths based on 3dmm for use
	if (name_3dmm == "bfm2017") {
		if (!define_path_manual) {
			modelfile = "../../bfm2017/bfm2017-1_bfm_nomouth.bin";
			mappingsfile = "../../bfm2017/ibug_to_bfm2017-1_bfm_nomouth.txt";
			contourfile = "../../bfm2017/bfm2017-1_bfm_nomouth_model_contours.json";
			edgetopologyfile = "../../bfm2017/bfm2017-1_bfm_nomouth_edge_topology.json";
		}
	}
	else if (name_3dmm == "sfm") {
		if (!define_path_manual) {
			modelfile = "../../sfm/model/eos/sfm_shape_29587.bin";
			mappingsfile = "../../sfm/ibug_to_sfm.txt";
			contourfile = "../../sfm/sfm_model_contours.json";
			edgetopologyfile = "../../sfm/regions/sfm_29587_edge_topology.json";
			blendshapesfile = "../../sfm/expression/expression_blendshapes_29587.bin";
		}
	}
	else {
		cout << "Unrecognized 3dmm name" << endl;
		return EXIT_FAILURE;
	}

    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
    Mat image = cv::imread(imagefile);
    LandmarkCollection<Eigen::Vector2f> landmarks;
    try
    {
        landmarks = core::read_pts_landmarks(landmarksfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error reading the landmarks: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    // The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    core::LandmarkMapper landmark_mapper;
    try
    {
        landmark_mapper = core::LandmarkMapper(mappingsfile);
    } catch (const std::exception& e)
    {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Get morphable model
	vector<morphablemodel::Blendshape> blendshapes;
	morphablemodel::MorphableModel morphable_model_with_expressions;
	if (name_3dmm == "bfm2017") {
		morphable_model_with_expressions = morphablemodel::MorphableModel(morphable_model.get_shape_model(), morphable_model.get_expression_model().value(), morphable_model.get_color_model(), morphable_model.get_texture_coordinates());
	}
	else if (name_3dmm == "sfm") {
		blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
		morphable_model_with_expressions = morphablemodel::MorphableModel(morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), morphable_model.get_texture_coordinates());
	}
    //const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

    //morphablemodel::MorphableModel morphable_model_with_expressions(morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), morphable_model.get_texture_coordinates());
	//morphablemodel::MorphableModel morphable_model_with_expressions(morphable_model.get_shape_model(), morphable_model.get_expression_model().value(), morphable_model.get_color_model(), morphable_model.get_texture_coordinates());

    // These two are used to fit the front-facing contour to the ibug contour landmarks:
    const fitting::ModelContour model_contour =
        contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);

    // The edge topology is used to speed up computation of the occluding face contour fitting:
    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);

    // CZ: load initial coeffs
    std::vector<float> init_pca_shape_coeffs;
    if (init_pca_shape_coeffs_fp != "")
        init_pca_shape_coeffs = cz::io::read_coeffs_from_file(init_pca_shape_coeffs_fp);
    std::vector<float> init_expression_coeffs;
	if (init_expression_coeffs_fp != "")
		init_expression_coeffs = cz::io::read_coeffs_from_file(init_expression_coeffs_fp);

	// // CZ: rescale landmarks
	// for (auto&& lm : landmarks)
	// {
	// 	lm.coordinates[0] *= landmark_scale;
	// 	lm.coordinates[1] *= landmark_scale;
	// }

    // Draw the loaded landmarks:
    Mat outimg = image.clone();
    for (auto&& lm : landmarks)
    {
        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
                      cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
    }

    // Fit the model, get back a mesh and the pose:
    //core::Mesh mesh;
    //fitting::RenderingParameters rendering_params;
    auto [mesh, rendering_params, pca_shape_coeffs, expression_coeffs] = fitting::fit_shape_and_pose(
        morphable_model_with_expressions, landmarks, landmark_mapper, image.cols, image.rows, edge_topology,
        ibug_contour, model_contour, init_pca_shape_coeffs, init_expression_coeffs, fix_pca_shape_coeffs, fix_expression_coeffs, num_fit_iter, cpp17::nullopt, 30.0f);

	fs::path outputfile;

	// Save the mesh as textured obj:
	if (save_obj) {
		outputfile = outputbasename + ".obj";
		core::write_textured_obj(mesh, outputfile.string());
	}

	// save 3dmm coefficients
	outputfile = outputbasename + ".coeffs_pca_shape.txt";
	cz::io::write_coeffs_to_file(outputfile.string(), pca_shape_coeffs);
	outputfile = outputbasename + ".coeffs_expression.txt";
	cz::io::write_coeffs_to_file(outputfile.string(), expression_coeffs);

	// Draw the fitted mesh as wireframe, and save the image:
	if (save_wireframe) {
		render::draw_wireframe(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
			fitting::get_opencv_viewport(image.cols, image.rows));

		outputfile = outputbasename + ".png";
		cv::imwrite(outputfile.string(), outimg);
	}

    //// The 3D head pose can be recovered as follows:
    //float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
    //// and similarly for pitch and roll.

	Eigen::Matrix<float, 3, 4> affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
	const auto model_view_matrix = eos::fitting::to_eigen(rendering_params.get_modelview());

	// CZ: rescale affine_from_ortho
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < 4; ++x) {
			affine_from_ortho(y, x) *= landmark_scale;
		}
	}

	// Extract the texture from the image using given mesh and camera parameters:
	core::Image4u isomap;
	if (save_texture || (save_mode != "none" && save_mode != "minimum")) {
		isomap = render::extract_texture(
			mesh, affine_from_ortho, model_view_matrix, core::from_mat(image), true,
			eos::render::TextureInterpolation::NearestNeighbour,
			boost::lexical_cast<int>(texture_resolution), outputbasename, save_mode);
	}

	if (save_texture) {
		// convert rgba texture to rgb texture
		core::Image3u isomap_rgb(texture_resolution, texture_resolution);
		for (int y = 0; y < texture_resolution; ++y) {
			for (int x = 0; x < texture_resolution; ++x) {
				isomap_rgb(y, x)[0] = isomap(y, x)[0];
				isomap_rgb(y, x)[1] = isomap(y, x)[1];
				isomap_rgb(y, x)[2] = isomap(y, x)[2];
			}
		}

		// And save the isomap:
		outputfile = outputbasename + ".isomap.png";
		cv::imwrite(outputfile.string(), core::to_mat(isomap_rgb));
	}

	// save modelview matrix
	outputfile = outputbasename + ".modelview.txt";
	cz::io::write_glm4x4_to_file(outputfile.string(), rendering_params.get_modelview());

	// save full projection matrix
	outputfile = outputbasename + ".affine_from_ortho.txt";
	std::ofstream outFile;
	outFile.open(outputfile.string());
	outFile << affine_from_ortho << std::endl;
	outFile.close();
    

	return EXIT_SUCCESS;
}
