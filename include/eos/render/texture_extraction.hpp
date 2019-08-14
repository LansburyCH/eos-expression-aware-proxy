/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/texture_extraction.hpp
 *
 * Copyright 2014-2017 Patrik Huber
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
#pragma once

#ifndef TEXTURE_EXTRACTION_HPP_
#define TEXTURE_EXTRACTION_HPP_

#include "eos/core/Image.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/render/detail/texture_extraction_detail.hpp"
#include "eos/render/render_affine.hpp"
//#include "eos/render/utils.hpp" // for clip_to_screen_space() in v2::
//#include "eos/render/Rasterizer.hpp"
//#include "eos/render/FragmentShader.hpp"
#include "eos/fitting/closest_edge_fitting.hpp" // for ray_triangle_intersect(). Move to eos/render/raycasting.hpp?

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "Eigen/Core"
#include "Eigen/QR"

#include <tuple>
#include <cassert>
#include <future>
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>

// CZ
#include <fstream>
#include <typeinfo>
#include "eos/cz/io.hpp"
#include "eos/cz/util.hpp"

namespace eos {
namespace render {

/* This function is copied from OpenCV,	originally under BSD licence.
 * imgwarp.cpp from OpenCV-3.2.0.
 *
 * Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */
// Note: The original functions used doubles.
inline Eigen::Matrix<float, 2, 3> get_affine_transform(const std::array<Eigen::Vector2f, 3>& src,
                                                       const std::array<Eigen::Vector2f, 3>& dst)
{
    using Eigen::Matrix;
    assert(src.size() == dst.size() && src.size() == 3);

    Matrix<float, 6, 6> A;
    Matrix<float, 6, 1> b;

    for (int i = 0; i < 3; i++)
    {
        A.block<1, 2>(2 * i, 0) = src[i];       // the odd rows
        A.block<1, 2>((2 * i) + 1, 3) = src[i]; // even rows
        A(2 * i, 2) = 1.0f;
        A((2 * i) + 1, 5) = 1.0f;
        A.block<1, 3>(2 * i, 3).setZero();
        A.block<1, 3>((2 * i) + 1, 0).setZero();
        b.segment<2>(2 * i) = dst[i];
    }

    Matrix<float, 6, 1> X = A.colPivHouseholderQr().solve(b);

    Matrix<float, 2, 3> transform_matrix;
    transform_matrix.block<1, 3>(0, 0) = X.segment<3>(0);
    transform_matrix.block<1, 3>(1, 0) = X.segment<3>(3);

    return transform_matrix;
};

/**
 * The interpolation types that can be used to map the
 * texture from the original image to the isomap.
 */
enum class TextureInterpolation { NearestNeighbour, Bilinear, Area };

// CZ: change of function interface
// Forward declarations:
core::Image4u extract_texture(const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix,
                              Eigen::Matrix<float, 4, 4> model_view_matrix, const core::Image3u& image,
                              const core::Image1d& depthbuffer, bool compute_view_angle,
                              TextureInterpolation mapping_type, int isomap_resolution,
                              std::string outputbasename, core::Image3d* coordsbuffer, std::string save_mode,
                              core::Image2d* texcoordsbuffer);

namespace detail {
core::Image4u interpolate_black_line(core::Image4u& isomap);
}

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 *
 * Note/Todo: Only use TextureInterpolation::NearestNeighbour
 * for the moment, the other methods don't have correct handling of
 * the alpha channel (and will most likely throw an exception).
 *
 * Todo: These should be renamed to extract_texture_affine? Can we combine both cases somehow?
 * Or an overload with RenderingParameters?
 *
 * For TextureInterpolation::NearestNeighbour, returns a 4-channel isomap
 * with the visibility in the 4th channel (0=invis, 255=visible).
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from. Should be 8UC3, other types not supported yet.
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned.
 * If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90? 127
 * meaning facing a 45?angle and 255 meaning front-facing, and all values in between). If set to false, the
 * alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */

// CZ: change of function interface
inline core::Image4u extract_texture(
    const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix,
    Eigen::Matrix<float, 4, 4> model_view_matrix, const core::Image3u& image, bool compute_view_angle = false,
    TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 2048,
    std::string outputbasename = "", std::string save_mode = "all")
{
    // Render the model to get a depth buffer:
    core::Image1d depthbuffer;

    // CZ
    core::Image3d coordsbuffer(image.rows, image.cols);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                coordsbuffer(y, x)[channel] = std::numeric_limits<double>::max();
            }
        }
    }

	// CZ
	core::Image2d texcoordsbuffer(image.rows, image.cols);
	for (int y = 0; y < image.rows; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			for (int channel = 0; channel < 2; ++channel)
			{
				texcoordsbuffer(y, x)[channel] = std::numeric_limits<double>::max();
			}
		}
	}

    // CZ: change of function interface
    std::tie(std::ignore, depthbuffer) = render::render_affine(
        mesh, affine_camera_matrix, image.cols, image.rows, true, &model_view_matrix, &coordsbuffer, &texcoordsbuffer);
    // Note: There's potential for optimisation here - we don't need to do everything that is done in
    // render_affine to just get the depthbuffer.

    // CZ: change of function interface
    // Now forward the call to the actual texture extraction function:
    return extract_texture(mesh, affine_camera_matrix, model_view_matrix, image, depthbuffer,
                           compute_view_angle, mapping_type, isomap_resolution, outputbasename, &coordsbuffer,
                           save_mode, &texcoordsbuffer);
};

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 * This function can be used if a depth buffer has already been computed.
 * To just run the texture extraction, see the overload
 * extract_texture(Mesh, cv::Mat, cv::Mat, TextureInterpolation, int). // Todo: I think this signature needs
 * updating.
 *
 * It might be wise to remove this overload as it can get quite confusing
 * with the zbuffer. Obviously the depthbuffer given should have been created
 * with the same (affine or ortho) projection matrix than the texture extraction is called with.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from.
 * @param[in] depthbuffer A pre-calculated depthbuffer image.
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned.
 * If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90? 127
 * meaning facing a 45?angle and 255 meaning front-facing, and all values in between). If set to false, the
 * alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */

// CZ: change of function interface
inline core::Image4u
extract_texture(const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix,
                Eigen::Matrix<float, 4, 4> model_view_matrix, const core::Image3u& image,
                const core::Image1d& depthbuffer, bool compute_view_angle = false,
                TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour,
                int isomap_resolution = 2048, std::string outputbasename = "",
                core::Image3d* coordsbuffer = 0, std::string save_mode = "all", core::Image2d* texcoordsbuffer = 0)
{
    assert(mesh.vertices.size() == mesh.texcoords.size());

    using Eigen::Vector2f;
    using Eigen::Vector3f;
    using Eigen::Vector4f;
    using std::ceil;
    using std::floor;
    using std::max;
    using std::min;
    using std::pow;
    using std::round;
    using std::sqrt;

    // CZ
    int num_triangle = mesh.tvi.size();
    int num_pixel_src = image.rows * image.cols;
    int num_pixel_dst = isomap_resolution * isomap_resolution;

	double* coordsbuffer_mat = 0;
	double* texcoordsbuffer_mat = 0;
    float* face_normal_mat = 0;
    float* src_normal_mat = 0;
    float* dst_normal_mat = 0;
    float* affine_camera_matrix_with_z_mat = 0;
    float* model_view_matrix_mat = 0;
    float* face_vertices_coor_src_mat = 0;
    float* face_vertices_coor_dst_mat = 0;
    float* src_pixel_face_mat = 0;
    float* dst_pixel_face_mat = 0;
    float* face_vertices_coor_3d_mat = 0;
    float* face_vertices_coor_3d_transform_mat = 0;
    float* face_vertices_indices_mat = 0;
    double* depthbuffer_mat = 0;
    double* remap_src_to_dst_mat = 0;
    double* remap_dst_to_src_mat = 0;

	if (save_mode == "none") 
	{
	} else if (save_mode == "minimum")
	{
		//affine_camera_matrix_with_z_mat = new float[4 * 4];
		//memset(affine_camera_matrix_with_z_mat, SCHAR_MAX, 4 * 4 * sizeof(float));

		//depthbuffer_mat = new double[num_pixel_src];
		//memset(depthbuffer_mat, SCHAR_MAX, num_pixel_src * sizeof(double));

		//remap_src_to_dst_mat = new double[num_pixel_src * 2];
		//memset(remap_src_to_dst_mat, SCHAR_MAX, num_pixel_src * 2 * sizeof(double));
	} else if (save_mode == "core")
    {
        coordsbuffer_mat = new double[num_pixel_src * 3];
        memset(coordsbuffer_mat, SCHAR_MAX, num_pixel_src * 3 * sizeof(double));

        face_normal_mat = new float[num_triangle * 3];
        memset(face_normal_mat, SCHAR_MAX, num_triangle * 3 * sizeof(float));

        src_normal_mat = new float[num_pixel_src * 3];
        memset(src_normal_mat, SCHAR_MAX, num_pixel_src * 3 * sizeof(float));
    } else if (save_mode == "all")
    {
		model_view_matrix_mat = new float[4 * 4];
		memset(model_view_matrix_mat, SCHAR_MAX, 4 * 4 * sizeof(float));

        coordsbuffer_mat = new double[num_pixel_src * 3];
        memset(coordsbuffer_mat, SCHAR_MAX, num_pixel_src * 3 * sizeof(double));

        face_normal_mat = new float[num_triangle * 3];
        memset(face_normal_mat, SCHAR_MAX, num_triangle * 3 * sizeof(float));

        src_normal_mat = new float[num_pixel_src * 3];
        memset(src_normal_mat, SCHAR_MAX, num_pixel_src * 3 * sizeof(float));

        dst_normal_mat = new float[num_pixel_dst * 3];
        memset(dst_normal_mat, SCHAR_MAX, num_pixel_dst * 3 * sizeof(float));

        affine_camera_matrix_with_z_mat = new float[4 * 4];
        memset(affine_camera_matrix_with_z_mat, SCHAR_MAX, 4 * 4 * sizeof(float));

        face_vertices_coor_src_mat = new float[num_triangle * 3 * 2];
        memset(face_vertices_coor_src_mat, SCHAR_MAX, num_triangle * 3 * 2 * sizeof(float));

        face_vertices_coor_dst_mat = new float[num_triangle * 3 * 2];
        memset(face_vertices_coor_dst_mat, SCHAR_MAX, num_triangle * 3 * 2 * sizeof(float));

        src_pixel_face_mat = new float[num_pixel_src];
        memset(src_pixel_face_mat, SCHAR_MAX, num_pixel_src * sizeof(float));

        dst_pixel_face_mat = new float[num_pixel_dst];
        memset(dst_pixel_face_mat, SCHAR_MAX, num_pixel_dst * sizeof(float));

        face_vertices_coor_3d_mat = new float[num_triangle * 3 * 3];
        memset(face_vertices_coor_3d_mat, SCHAR_MAX, num_triangle * 3 * 3 * sizeof(float));

        face_vertices_coor_3d_transform_mat = new float[num_triangle * 3 * 3];
        memset(face_vertices_coor_3d_transform_mat, SCHAR_MAX, num_triangle * 3 * 3 * sizeof(float));

        face_vertices_indices_mat = new float[num_triangle * 3];
        memset(face_vertices_indices_mat, SCHAR_MAX, num_triangle * 3 * sizeof(float));

        depthbuffer_mat = new double[num_pixel_src];
        memset(depthbuffer_mat, SCHAR_MAX, num_pixel_src * sizeof(double));

        remap_src_to_dst_mat = new double[num_pixel_src * 2];
        memset(remap_src_to_dst_mat, SCHAR_MAX, num_pixel_src * 2 * sizeof(double));

        remap_dst_to_src_mat = new double[num_pixel_dst * 2];
        memset(remap_dst_to_src_mat, SCHAR_MAX, num_pixel_dst * 2 * sizeof(double));

		texcoordsbuffer_mat = new double[num_pixel_src * 2];
		memset(texcoordsbuffer_mat, SCHAR_MAX, num_pixel_src * 2 * sizeof(double));
    } else
    {
        std::cerr << "extract_texture: Unrecognized option for save_mode" << std::endl;
    }

    // CZ: save depthbuffer to .mat
    if (depthbuffer_mat != 0)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            for (int y = 0; y < image.rows; ++y)
            {
                depthbuffer_mat[y + x * image.rows] = depthbuffer(y, x);
            }
        }
    }

    // CZ: save coordsbuffer to .mat
    if (coordsbuffer_mat != 0 && coordsbuffer != 0)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            for (int y = 0; y < image.rows; ++y)
            {
                coordsbuffer_mat[y + x * image.rows + 0 * num_pixel_src] = (*coordsbuffer)(y, x)[0];
                coordsbuffer_mat[y + x * image.rows + 1 * num_pixel_src] = (*coordsbuffer)(y, x)[1];
                coordsbuffer_mat[y + x * image.rows + 2 * num_pixel_src] = (*coordsbuffer)(y, x)[2];
            }
        }
    } else if (coordsbuffer == 0)
    {
        std::cout << "Warning: coordsbuffer is empty" << std::endl;
    }

	// CZ: save texcoordsbuffer to .mat
	if (texcoordsbuffer_mat != 0 && texcoordsbuffer != 0)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			for (int y = 0; y < image.rows; ++y)
			{
				texcoordsbuffer_mat[y + x * image.rows + 0 * num_pixel_src] = (*texcoordsbuffer)(y, x)[0];
				texcoordsbuffer_mat[y + x * image.rows + 1 * num_pixel_src] = (*texcoordsbuffer)(y, x)[1];
			}
		}
	}

    Eigen::Matrix<float, 4, 4> affine_camera_matrix_with_z =
        detail::calculate_affine_z_direction(affine_camera_matrix);

    // CZ: save affine_camera_matrix_with_z to .mat
    if (affine_camera_matrix_with_z_mat != 0)
    {
        for (int x = 0; x < 4; ++x)
        {
            for (int y = 0; y < 4; ++y)
            {
                affine_camera_matrix_with_z_mat[y + x * 4] = affine_camera_matrix_with_z(y, x);
            }
        }
    }

    // CZ: save model_view_matrix to .mat
    if (model_view_matrix_mat != 0)
    {
        for (int x = 0; x < 4; ++x)
        {
            for (int y = 0; y < 4; ++y)
            {
                model_view_matrix_mat[y + x * 4] = model_view_matrix(y, x);
            }
        }
    }

    // Todo: We should handle gray images, but output a 4-channel isomap nevertheless I think.
    core::Image4u isomap(isomap_resolution, isomap_resolution); // We should initialise with zeros.
                                                                // Incidentially, the current Image4u c'tor
                                                                // does that.

    std::vector<std::future<void>> results;

    // CZ
    // for (const auto& triangle_indices : mesh.tvi)
    for (int triangle_id = 0; triangle_id < num_triangle; ++triangle_id)
    {

        // Note: If there's a performance problem, there's no need to capture the whole mesh - we could
        // capture only the three required vertices with their texcoords.
        auto extract_triangle = [&mesh, &affine_camera_matrix_with_z, &depthbuffer, &isomap, &mapping_type,
                                 &image, &compute_view_angle, &model_view_matrix, &num_triangle,
                                 &num_pixel_src, &num_pixel_dst, triangle_id, &face_normal_mat,
                                 &face_vertices_coor_src_mat, &face_vertices_coor_dst_mat,
                                 &src_pixel_face_mat, &dst_pixel_face_mat, &isomap_resolution,
                                 &src_normal_mat, &dst_normal_mat, &face_vertices_coor_3d_mat,
                                 &face_vertices_coor_3d_transform_mat, &face_vertices_indices_mat,
                                 &remap_src_to_dst_mat, &remap_dst_to_src_mat]() {
            // CZ
            const auto& triangle_indices = mesh.tvi[triangle_id];

            // Find out if the current triangle is visible:
            // We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
            // check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to
            // extract the texture.
            // Possible improvement: - If only part of the triangle is visible, split it

            // This could be optimized in 2 ways though:
            // - Use render(), or as in render(...), transfer the vertices once, not in a loop over all
            //   triangles (vertices are getting transformed multiple times)
            // - We transform them later (below) a second time. Only do it once.

            const Vector4f v0_as_Vector4f(mesh.vertices[triangle_indices[0]][0],
                                          mesh.vertices[triangle_indices[0]][1],
                                          mesh.vertices[triangle_indices[0]][2], 1.0f);
            const Vector4f v1_as_Vector4f(mesh.vertices[triangle_indices[1]][0],
                                          mesh.vertices[triangle_indices[1]][1],
                                          mesh.vertices[triangle_indices[1]][2], 1.0f);
            const Vector4f v2_as_Vector4f(mesh.vertices[triangle_indices[2]][0],
                                          mesh.vertices[triangle_indices[2]][1],
                                          mesh.vertices[triangle_indices[2]][2], 1.0f);

            // Project the triangle vertices to screen coordinates, and use the depthbuffer to check whether
            // the triangle is visible:
            const Vector4f v0 = affine_camera_matrix_with_z * v0_as_Vector4f;
            const Vector4f v1 = affine_camera_matrix_with_z * v1_as_Vector4f;
            const Vector4f v2 = affine_camera_matrix_with_z * v2_as_Vector4f;

            // CZ: store face vertices indices
            if (face_vertices_indices_mat != 0)
            {
                for (int face_vertex_id = 0; face_vertex_id < 3; ++face_vertex_id)
                {
                    face_vertices_indices_mat[triangle_id + face_vertex_id * num_triangle] =
                        triangle_indices[face_vertex_id] + 1;
                }
            }

            // CZ: store face vertices coordinates (original and transformed) in 3d space
            if (face_vertices_coor_3d_mat != 0)
            {
                for (int coor_id = 0; coor_id < 3; ++coor_id)
                {
                    face_vertices_coor_3d_mat[triangle_id + 0 * num_triangle + coor_id * 3 * num_triangle] =
                        v0_as_Vector4f[coor_id];
                    face_vertices_coor_3d_mat[triangle_id + 1 * num_triangle + coor_id * 3 * num_triangle] =
                        v1_as_Vector4f[coor_id];
                    face_vertices_coor_3d_mat[triangle_id + 2 * num_triangle + coor_id * 3 * num_triangle] =
                        v2_as_Vector4f[coor_id];
                }
            }
            if (face_vertices_coor_3d_transform_mat != 0)
            {
                for (int coor_id = 0; coor_id < 3; ++coor_id)
                {
                    face_vertices_coor_3d_transform_mat[triangle_id + 0 * num_triangle +
                                                        coor_id * 3 * num_triangle] = v0[coor_id];
                    face_vertices_coor_3d_transform_mat[triangle_id + 1 * num_triangle +
                                                        coor_id * 3 * num_triangle] = v1[coor_id];
                    face_vertices_coor_3d_transform_mat[triangle_id + 2 * num_triangle +
                                                        coor_id * 3 * num_triangle] = v2[coor_id];
                }
            }

            if (!detail::is_triangle_visible(glm::tvec4<float>(v0[0], v0[1], v0[2], v0[3]),
                                             glm::tvec4<float>(v1[0], v1[1], v1[2], v1[3]),
                                             glm::tvec4<float>(v2[0], v2[1], v2[2], v2[3]), depthbuffer))
            {
                // continue;
                return;
            }

            float alpha_value;
            if (compute_view_angle)
            {
                // Calculate how well visible the current triangle is:
                // (in essence, the dot product of the viewing direction (0, 0, 1) and the face normal)
                const Vector3f face_normal =
                    compute_face_normal(v0_as_Vector4f, v1_as_Vector4f, v2_as_Vector4f);

                // CZ
                // Transform the normal to "screen" (kind of "eye") space using the upper 3x3 part of the
                // Vector3f face_normal_transformed = affine_camera_matrix_with_z.block<3, 3>(0, 0) *
                // face_normal; // Incorrect I think, because we want normal of real geometry instead of
                // projected geometry
                Vector3f face_normal_transformed = model_view_matrix.block<3, 3>(0, 0) * face_normal;
                face_normal_transformed.normalize(); // normalise to unit length

                // CZ: store normal of this face
                if (face_normal_mat != 0)
                {
                    face_normal_mat[triangle_id + 0 * num_triangle] = face_normal_transformed[0];
                    face_normal_mat[triangle_id + 1 * num_triangle] = face_normal_transformed[1];
                    face_normal_mat[triangle_id + 2 * num_triangle] = face_normal_transformed[2];
                }

                // Implementation notes regarding the affine camera matrix and the sign:
                // If the matrix given were the model_view matrix, the sign would be correct.
                // However, affine_camera_matrix includes glm::ortho, which includes a z-flip.
                // So we need to flip one of the two signs.
                // * viewing_direction(0.0f, 0.0f, 1.0f) is correct if affine_camera_matrix were only a
                // model_view matrix
                // * affine_camera_matrix includes glm::ortho, which flips z, so we flip the sign of
                // viewing_direction. We don't need the dot product since viewing_direction.xy are 0 and .z is
                // 1:
                const float angle = -face_normal_transformed[2]; // flip sign, see above
                assert(angle >= -1.f && angle <= 1.f);
                // angle is [-1, 1].
                //  * +1 means   0?(same direction)
                //  *  0 means  90?
                //  * -1 means 180?(facing opposite directions)
                // It's a linear relation, so +0.5 is 45?etc.
                // An angle larger than 90?means the vertex won't be rendered anyway (because it's
                // back-facing) so we encode 0?to 90?
                if (angle < 0.0f)
                {
                    alpha_value = 0.0f;
                } else
                {
                    alpha_value = angle * 255.0f;
                }
            } else
            {
                // no visibility angle computation - if the triangle/pixel is visible, set the alpha chan to
                // 255 (fully visible pixel).
                alpha_value = 255.0f;
            }

            // Todo: Documentation
            std::array<Vector2f, 3> src_tri;
            std::array<Vector2f, 3> dst_tri;

            Vector4f vec(mesh.vertices[triangle_indices[0]][0], mesh.vertices[triangle_indices[0]][1],
                         mesh.vertices[triangle_indices[0]][2], 1.0f);
            Vector4f res = affine_camera_matrix_with_z * vec;
            src_tri[0] = Vector2f(res[0], res[1]);

            vec = Vector4f(mesh.vertices[triangle_indices[1]][0], mesh.vertices[triangle_indices[1]][1],
                           mesh.vertices[triangle_indices[1]][2], 1.0f);
            res = affine_camera_matrix_with_z * vec;
            src_tri[1] = Vector2f(res[0], res[1]);

            vec = Vector4f(mesh.vertices[triangle_indices[2]][0], mesh.vertices[triangle_indices[2]][1],
                           mesh.vertices[triangle_indices[2]][2], 1.0f);
            res = affine_camera_matrix_with_z * vec;
            src_tri[2] = Vector2f(res[0], res[1]);

            dst_tri[0] = Vector2f((isomap.cols - 0.5) * mesh.texcoords[triangle_indices[0]][0],
                                  (isomap.rows - 0.5) * mesh.texcoords[triangle_indices[0]][1]);
            dst_tri[1] = Vector2f((isomap.cols - 0.5) * mesh.texcoords[triangle_indices[1]][0],
                                  (isomap.rows - 0.5) * mesh.texcoords[triangle_indices[1]][1]);
            dst_tri[2] = Vector2f((isomap.cols - 0.5) * mesh.texcoords[triangle_indices[2]][0],
                                  (isomap.rows - 0.5) * mesh.texcoords[triangle_indices[2]][1]);

            // CZ: store face vertices coordinates on image (src) and isomap (dst)
            if (face_vertices_coor_src_mat != 0)
            {
                for (int face_vertex_id = 0; face_vertex_id < 3; ++face_vertex_id)
                {
                    for (int coor_id = 0; coor_id < 2; ++coor_id)
                    {
                        face_vertices_coor_src_mat[triangle_id + face_vertex_id * num_triangle +
                                                   coor_id * 3 * num_triangle] =
                            src_tri[face_vertex_id][coor_id];
                    }
                }
            }
            if (face_vertices_coor_dst_mat != 0)
            {
                for (int face_vertex_id = 0; face_vertex_id < 3; ++face_vertex_id)
                {
                    for (int coor_id = 0; coor_id < 2; ++coor_id)
                    {
                        face_vertices_coor_dst_mat[triangle_id + face_vertex_id * num_triangle +
                                                   coor_id * 3 * num_triangle] =
                            dst_tri[face_vertex_id][coor_id];
                    }
                }
            }

            // We now have the source triangles in the image and the source triangle in the isomap
            // We use the inverse/ backward mapping approach, so we want to find the corresponding texel
            // (texture-pixel) for each pixel in the isomap

            // Get the inverse Affine Transform from original image: from dst (pixel in isomap) to src (in
            // image)
            Eigen::Matrix<float, 2, 3> warp_mat_org_inv = get_affine_transform(dst_tri, src_tri);

            // We now loop over all pixels in the triangle and select, depending on the mapping type, the
            // corresponding texel(s) in the source image
            for (int x = min(dst_tri[0][0], min(dst_tri[1][0], dst_tri[2][0]));
                 x < max(dst_tri[0][0], max(dst_tri[1][0], dst_tri[2][0])); ++x)
            {
                for (int y = min(dst_tri[0][1], min(dst_tri[1][1], dst_tri[2][1]));
                     y < max(dst_tri[0][1], max(dst_tri[1][1], dst_tri[2][1])); ++y)
                {
                    if (detail::is_point_in_triangle(Vector2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2]))
                    {
                        // CZ: store the information that this dst pixel belongs to this face
                        if (dst_pixel_face_mat != 0)
                        {
                            dst_pixel_face_mat[y + x * isomap_resolution] = triangle_id;
                        }
                        // CZ: store dst normal
                        if (dst_normal_mat != 0)
                        {
                            dst_normal_mat[y + x * isomap_resolution + 0 * num_pixel_dst] =
                                face_normal_mat[triangle_id + 0 * num_triangle];
                            dst_normal_mat[y + x * isomap_resolution + 1 * num_pixel_dst] =
                                face_normal_mat[triangle_id + 1 * num_triangle];
                            dst_normal_mat[y + x * isomap_resolution + 2 * num_pixel_dst] =
                                face_normal_mat[triangle_id + 2 * num_triangle];
                        }

                        // As the coordinates of the transformed pixel in the image will most likely not lie
                        // on a texel, we have to choose how to calculate the pixel colors depending on the
                        // next texels

                        // There are three different texture interpolation methods: area, bilinear and nearest
                        // neighbour

                        // Area mapping: calculate mean color of texels in transformed pixel area
                        if (mapping_type == TextureInterpolation::Area)
                        {

                            // calculate positions of 4 corners of pixel in image (src)
                            const Vector3f homogenous_dst_upper_left(x - 0.5f, y - 0.5f, 1.0f);
                            const Vector3f homogenous_dst_upper_right(x + 0.5f, y - 0.5f, 1.0f);
                            const Vector3f homogenous_dst_lower_left(x - 0.5f, y + 0.5f, 1.0f);
                            const Vector3f homogenous_dst_lower_right(x + 0.5f, y + 0.5f, 1.0f);

                            const Vector2f src_texel_upper_left =
                                warp_mat_org_inv * homogenous_dst_upper_left;
                            const Vector2f src_texel_upper_right =
                                warp_mat_org_inv * homogenous_dst_upper_right;
                            const Vector2f src_texel_lower_left =
                                warp_mat_org_inv * homogenous_dst_lower_left;
                            const Vector2f src_texel_lower_right =
                                warp_mat_org_inv * homogenous_dst_lower_right;

                            const float min_a = min(min(src_texel_upper_left[0], src_texel_upper_right[0]),
                                                    min(src_texel_lower_left[0], src_texel_lower_right[0]));
                            const float max_a = max(max(src_texel_upper_left[0], src_texel_upper_right[0]),
                                                    max(src_texel_lower_left[0], src_texel_lower_right[0]));
                            const float min_b = min(min(src_texel_upper_left[1], src_texel_upper_right[1]),
                                                    min(src_texel_lower_left[1], src_texel_lower_right[1]));
                            const float max_b = max(max(src_texel_upper_left[1], src_texel_upper_right[1]),
                                                    max(src_texel_lower_left[1], src_texel_lower_right[1]));

                            Eigen::Vector3i color; // std::uint8_t actually.
                            int num_texels = 0;

                            // loop over square in which quadrangle out of the four corners of pixel is
                            for (int a = ceil(min_a); a <= floor(max_a); ++a)
                            {
                                for (int b = ceil(min_b); b <= floor(max_b); ++b)
                                {
                                    // check if texel is in quadrangle
                                    if (detail::is_point_in_triangle(Vector2f(a, b), src_texel_upper_left,
                                                                     src_texel_lower_left,
                                                                     src_texel_upper_right) ||
                                        detail::is_point_in_triangle(Vector2f(a, b), src_texel_lower_left,
                                                                     src_texel_upper_right,
                                                                     src_texel_lower_right))
                                    {
                                        if (a < image.cols && b < image.rows)
                                        { // check if texel is in image
                                            num_texels++;
                                            color += Eigen::Vector3i(image(b, a)[0], image(b, a)[1],
                                                                     image(b, a)[2]);
                                        }
                                    }
                                }
                            }
                            if (num_texels > 0)
                                color = color / num_texels;
                            else
                            { // if no corresponding texel found, nearest neighbour interpolation
                                // calculate corresponding position of dst_coord pixel center in image (src)
                                Vector3f homogenous_dst_coord(x, y, 1.0f);
                                Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

                                if ((round(src_texel[1]) < image.rows) && round(src_texel[0]) < image.cols)
                                {
                                    const int y = round(src_texel[1]);
                                    const int x = round(src_texel[0]);
                                    color = Eigen::Vector3i(image(y, x)[0], image(y, x)[1], image(y, x)[2]);
                                }
                            }
                            isomap(y, x) = {
                                static_cast<std::uint8_t>(color[0]), static_cast<std::uint8_t>(color[1]),
                                static_cast<std::uint8_t>(color[2]), static_cast<std::uint8_t>(alpha_value)};
                        }
                        // Bilinear mapping: calculate pixel color depending on the four neighbouring texels
                        else if (mapping_type == TextureInterpolation::Bilinear)
                        {

                            // calculate corresponding position of dst_coord pixel center in image (src)
                            const Vector3f homogenous_dst_coord(x, y, 1.0f);
                            const Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

                            // calculate euclidean distances to next 4 texels
                            float distance_upper_left = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) +
                                                             pow(src_texel[1] - floor(src_texel[1]), 2));
                            float distance_upper_right = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) +
                                                              pow(src_texel[1] - ceil(src_texel[1]), 2));
                            float distance_lower_left = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) +
                                                             pow(src_texel[1] - floor(src_texel[1]), 2));
                            float distance_lower_right = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) +
                                                              pow(src_texel[1] - ceil(src_texel[1]), 2));

                            // normalise distances that the sum of all distances is 1
                            const float sum_distances = distance_lower_left + distance_lower_right +
                                                        distance_upper_left + distance_upper_right;
                            distance_lower_left /= sum_distances;
                            distance_lower_right /= sum_distances;
                            distance_upper_left /= sum_distances;
                            distance_upper_right /= sum_distances;

                            // set color depending on distance from next 4 texels
                            // (we map the data from std::array<uint8_t, 3> to an Eigen::Map, then cast that
                            // to float to multiply with the float-scalar distance.)
                            // (this is untested!)
                            const Vector3f color_upper_left =
                                Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(
                                    image(floor(src_texel[1]), floor(src_texel[0])).data(), 3)
                                    .cast<float>() *
                                distance_upper_left;
                            const Vector3f color_upper_right =
                                Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(
                                    image(floor(src_texel[1]), ceil(src_texel[0])).data(), 3)
                                    .cast<float>() *
                                distance_upper_right;
                            const Vector3f color_lower_left =
                                Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(
                                    image(ceil(src_texel[1]), floor(src_texel[0])).data(), 3)
                                    .cast<float>() *
                                distance_lower_left;
                            const Vector3f color_lower_right =
                                Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(
                                    image(ceil(src_texel[1]), ceil(src_texel[0])).data(), 3)
                                    .cast<float>() *
                                distance_lower_right;

                            // isomap(y, x)[color] = color_upper_left + color_upper_right + color_lower_left +
                            // color_lower_right;
                            isomap(y, x)[0] = static_cast<std::uint8_t>(
                                glm::clamp(color_upper_left[0] + color_upper_right[0] + color_lower_left[0] +
                                               color_lower_right[0],
                                           0.f, 255.0f));
                            isomap(y, x)[1] = static_cast<std::uint8_t>(
                                glm::clamp(color_upper_left[1] + color_upper_right[1] + color_lower_left[1] +
                                               color_lower_right[1],
                                           0.f, 255.0f));
                            isomap(y, x)[2] = static_cast<std::uint8_t>(
                                glm::clamp(color_upper_left[2] + color_upper_right[2] + color_lower_left[2] +
                                               color_lower_right[2],
                                           0.f, 255.0f));
                            isomap(y, x)[3] = static_cast<std::uint8_t>(alpha_value); // pixel is visible
                        }
                        // NearestNeighbour mapping: set color of pixel to color of nearest texel
                        else if (mapping_type == TextureInterpolation::NearestNeighbour)
                        {

                            // calculate corresponding position of dst_coord pixel center in image (src)
                            const Vector3f homogenous_dst_coord(x, y, 1.0f);
                            Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

                            if ((round(src_texel[1]) < image.rows) && (round(src_texel[0]) < image.cols) &&
                                round(src_texel[0]) > 0 && round(src_texel[1]) > 0)
                            {
                                // CZ
                                //src_texel[0] *= tex_scale;
                                //src_texel[1] *= tex_scale;
								if (src_texel[0] > image.cols - 1 || (src_texel[1] > image.rows - 1)) {
									continue;
								}

                                isomap(y, x)[0] = image(round(src_texel[1]), round(src_texel[0]))[0];
                                isomap(y, x)[1] = image(round(src_texel[1]), round(src_texel[0]))[1];
                                isomap(y, x)[2] = image(round(src_texel[1]), round(src_texel[0]))[2];
                                isomap(y, x)[3] = static_cast<std::uint8_t>(alpha_value); // pixel is visible

                                // CZ
                                int src_x = (int)round(src_texel[0]);
                                int src_y = (int)round(src_texel[1]);

                                // CZ: store the information that this src pixel belongs to this face
                                if (src_pixel_face_mat != 0)
                                {
                                    src_pixel_face_mat[src_y + src_x * image.rows] = triangle_id;
                                }
                                // CZ: store src normal
                                if (src_normal_mat != 0)
                                {
                                    src_normal_mat[src_y + src_x * image.rows + 0 * num_pixel_src] =
                                        face_normal_mat[triangle_id + 0 * num_triangle];
                                    src_normal_mat[src_y + src_x * image.rows + 1 * num_pixel_src] =
                                        face_normal_mat[triangle_id + 1 * num_triangle];
                                    src_normal_mat[src_y + src_x * image.rows + 2 * num_pixel_src] =
                                        face_normal_mat[triangle_id + 2 * num_triangle];
                                }
                                // CZ: store the remapping from src to dst
                                if (remap_src_to_dst_mat != 0)
                                {
                                    remap_src_to_dst_mat[src_y + src_x * image.rows + 0 * num_pixel_src] = x;
                                    remap_src_to_dst_mat[src_y + src_x * image.rows + 1 * num_pixel_src] = y;
                                }
                                // CZ: store the remapping from dst to src
                                if (remap_dst_to_src_mat != 0)
                                {
                                    // remap_dst_to_src_mat[y + x * isomap_resolution + 0 * num_pixel_dst] =
                                    // src_x; remap_dst_to_src_mat[y + x * isomap_resolution + 1 *
                                    // num_pixel_dst] = src_y;
                                    remap_dst_to_src_mat[y + x * isomap_resolution + 0 * num_pixel_dst] =
                                        src_texel[0];
                                    remap_dst_to_src_mat[y + x * isomap_resolution + 1 * num_pixel_dst] =
                                        src_texel[1];
                                }
                            }
                        }
                    }
                }
            }
        }; // end lambda auto extract_triangle();
        results.emplace_back(std::async(extract_triangle));
    } // end for all mesh.tvi
    // Collect all the launched tasks:
    for (auto&& r : results)
    {
        r.get();
    }

    // Workaround for the black line in the isomap (see GitHub issue #4):
    /*if (mesh.texcoords.size() <= 3448)
      {
          isomap = detail::interpolate_black_line(isomap);
      } */

    // CZ
    std::cout << "Finish texture_extraction" << std::endl;

    /********** Write to binary file **********/
    // CZ: save coordsbuffer
    if (coordsbuffer_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(uint64_t)image.rows, (uint64_t)image.cols, 3};
        cz::io::BinaryFile_Write(outputbasename + ".coordsbuffer.dat", coordsbuffer_mat, num_dim, dims);
        delete[] dims;
        delete[] coordsbuffer_mat;
        std::cout << "coordsbuffer saved" << std::endl;
    }

	// CZ: save texcoordsbuffer
	if (texcoordsbuffer_mat != 0)
	{
		uint64_t num_dim = 3;
		uint64_t* dims = new uint64_t[num_dim]{ (uint64_t)image.rows, (uint64_t)image.cols, 2 };
		cz::io::BinaryFile_Write(outputbasename + ".texcoordsbuffer.dat", texcoordsbuffer_mat, num_dim, dims);
		delete[] dims;
		delete[] texcoordsbuffer_mat;
		std::cout << "texcoordsbuffer saved" << std::endl;
	}

    // CZ: save face_normal
    if (face_normal_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{(uint64_t)num_triangle, 3};
        cz::io::BinaryFile_Write(outputbasename + ".face_normal.dat", face_normal_mat, num_dim, dims);
        delete[] dims;
        delete[] face_normal_mat;
        std::cout << "face_normal saved" << std::endl;
    }

    // CZ: save src_normal
    if (src_normal_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(uint64_t)image.rows, (uint64_t)image.cols, 3};
        cz::io::BinaryFile_Write(outputbasename + ".src_normal.dat", src_normal_mat, num_dim, dims);
        delete[] dims;
        delete[] src_normal_mat;
        std::cout << "src_normal saved" << std::endl;
    }

    // CZ: save dst_normal
    if (dst_normal_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)isomap_resolution, (size_t)isomap_resolution, 3};
        cz::io::BinaryFile_Write(outputbasename + ".dst_normal.dat", dst_normal_mat, num_dim, dims);
        delete[] dims;
        delete[] dst_normal_mat;
        std::cout << "dst_normal saved" << std::endl;
    }

    // CZ: save affine_camera_matrix_with_z
    if (affine_camera_matrix_with_z_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{4, 4};
        cz::io::BinaryFile_Write(outputbasename + ".affine_camera_matrix_with_z.dat",
                                 affine_camera_matrix_with_z_mat, num_dim, dims);
        delete[] dims;
        delete[] affine_camera_matrix_with_z_mat;
        std::cout << "affine_camera_matrix_with_z saved" << std::endl;
    }

    // CZ: save model_view_matrix
    if (model_view_matrix_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{4, 4};
        cz::io::BinaryFile_Write(outputbasename + ".model_view_matrix.dat", model_view_matrix_mat, num_dim,
                                 dims);
        delete[] dims;
        delete[] model_view_matrix_mat;
        std::cout << "model_view_matrix saved" << std::endl;
    }

    // CZ: save face_vertices_coor_src
    if (face_vertices_coor_src_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)num_triangle, 3, 2};
        cz::io::BinaryFile_Write(outputbasename + ".face_vertices_coor_src.dat", face_vertices_coor_src_mat,
                                 num_dim, dims);
        delete[] dims;
        delete[] face_vertices_coor_src_mat;
        std::cout << "face_vertices_coor_src saved" << std::endl;
    }

    // CZ: save face_vertices_coor_dst
    if (face_vertices_coor_dst_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)num_triangle, 3, 2};
        cz::io::BinaryFile_Write(outputbasename + ".face_vertices_coor_dst.dat", face_vertices_coor_dst_mat,
                                 num_dim, dims);
        delete[] dims;
        delete[] face_vertices_coor_dst_mat;
        std::cout << "face_vertices_coor_dst saved" << std::endl;
    }

    // CZ: save src_pixel_face
    if (src_pixel_face_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{image.rows, image.cols};
        cz::io::BinaryFile_Write(outputbasename + ".src_pixel_face.dat", src_pixel_face_mat, num_dim, dims);
        delete[] dims;
        delete[] src_pixel_face_mat;
        std::cout << "src_pixel_face saved" << std::endl;
    }

    // CZ: save dst_pixel_face
    if (dst_pixel_face_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)isomap_resolution, (size_t)isomap_resolution};
        cz::io::BinaryFile_Write(outputbasename + ".dst_pixel_face.dat", dst_pixel_face_mat, num_dim, dims);
        delete[] dims;
        delete[] dst_pixel_face_mat;
        std::cout << "dst_pixel_face saved" << std::endl;
    }

    // CZ: save face_vertices_coor_3d
    if (face_vertices_coor_3d_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)num_triangle, 3, 3};
        cz::io::BinaryFile_Write(outputbasename + ".face_vertices_coor_3d.dat", face_vertices_coor_3d_mat,
                                 num_dim, dims);
        delete[] dims;
        delete[] face_vertices_coor_3d_mat;
        std::cout << "face_vertices_coor_3d saved" << std::endl;
    }

    // CZ: save face_vertices_coor_3d_transform
    if (face_vertices_coor_3d_transform_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)num_triangle, 3, 3};
        cz::io::BinaryFile_Write(outputbasename + ".face_vertices_coor_3d_transform.dat",
                                 face_vertices_coor_3d_transform_mat, num_dim, dims);
        delete[] dims;
        delete[] face_vertices_coor_3d_transform_mat;
        std::cout << "face_vertices_coor_3d_transform saved" << std::endl;
    }

    // CZ: save face_vertices_indices
    if (face_vertices_indices_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)num_triangle, 3};
        cz::io::BinaryFile_Write(outputbasename + ".face_vertices_indices.dat", face_vertices_indices_mat,
                                 num_dim, dims);
        delete[] dims;
        delete[] face_vertices_indices_mat;
        std::cout << "face_vertices_indices saved" << std::endl;
    }

    // CZ: save depthbuffer
    if (depthbuffer_mat != 0)
    {
        uint64_t num_dim = 2;
        uint64_t* dims = new uint64_t[num_dim]{(uint64_t)image.rows, (uint64_t)image.cols};
        cz::io::BinaryFile_Write(outputbasename + ".depthbuffer.dat", depthbuffer_mat, num_dim, dims);
        delete[] dims;
        delete[] depthbuffer_mat;
        std::cout << "depthbuffer saved" << std::endl;
    }

    // CZ: save remap_src_to_dst
    if (remap_src_to_dst_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{image.rows, image.cols, 2};
        cz::io::BinaryFile_Write(outputbasename + ".remap_src_to_dst.dat", remap_src_to_dst_mat, num_dim,
                                 dims);
        delete[] dims;
        delete[] remap_src_to_dst_mat;
        std::cout << "remap_src_to_dst saved" << std::endl;
    }

    // CZ: save remap_dst_to_src
    if (remap_dst_to_src_mat != 0)
    {
        uint64_t num_dim = 3;
        uint64_t* dims = new uint64_t[num_dim]{(size_t)isomap_resolution, (size_t)isomap_resolution, 2};
        cz::io::BinaryFile_Write(outputbasename + ".remap_dst_to_src.dat", remap_dst_to_src_mat, num_dim,
                                 dims);
        delete[] dims;
        delete[] remap_dst_to_src_mat;
        std::cout << "remap_dst_to_src saved" << std::endl;
    }

    return isomap;
};

/* New texture extraction, will replace above one at some point: */
namespace v2 {

/**
 * @brief Extracts the texture of the face from the given image and stores it as isomap (a rectangular texture
 * map).
 *
 * New texture extraction, will replace above one at some point.
 * Copy the documentation from above extract_texture function, once we replace it.
 *
 * Note/Todo: Add an overload that takes a vector of bool / visible vertices, for the case when we already
 * computed the visibility? (e.g. for edge-fitting)
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] view_model_matrix Todo.
 * @param[in] projection_matrix Todo.
 * @param[in] viewport Not needed at the moment. Might be, if we change clip_to_screen_space() to take a
 * viewport.
 * @param[in] image The image to extract the texture from. Todo: Does it have to be 8UC3 or something, or does
 * it not matter?
 * @param[in] compute_view_angle Unused at the moment.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
// cv::Mat extract_texture(const core::Mesh& mesh, glm::mat4x4 view_model_matrix, glm::mat4x4
// projection_matrix,
//                        glm::vec4 /*viewport, not needed at the moment */, cv::Mat image,
//                        bool /* compute_view_angle, unused atm */, int isomap_resolution = 512)
/*
{
    using detail::divide_by_w;
    using glm::vec2;
    using glm::vec3;
    using glm::vec4;
    using std::vector;
    // actually we only need a rasteriser for this!
    Rasterizer<ExtractionFragmentShader> extraction_rasterizer(isomap_resolution, isomap_resolution);
    Texture image_to_extract_from_as_tex = create_mipmapped_texture(image, 1);
    extraction_rasterizer.enable_depth_test = false;
    extraction_rasterizer.extracting_tex = true;

    vector<bool> visibility_ray;
    vector<vec4> rotated_vertices;
    // In perspective case... does the perspective projection matrix not change visibility? Do we not need to
    // apply it?
    // (If so, then we can change the two input matrices to this function to one (mvp_matrix)).
    std::for_each(std::begin(mesh.vertices), std::end(mesh.vertices),
                  [&rotated_vertices, &view_model_matrix](auto&& v) {
                      rotated_vertices.push_back(view_model_matrix * v);
                  });
    // This code is duplicated from the edge-fitting. I think I can put this into a function in the library.
    for (const auto& vertex : rotated_vertices)
    {
        bool visible = true;
        // For every tri of the rotated mesh:
        for (auto&& tri : mesh.tvi)
        {
            auto& v0 = rotated_vertices[tri[0]]; // const?
            auto& v1 = rotated_vertices[tri[1]];
            auto& v2 = rotated_vertices[tri[2]];

            vec3 ray_origin(vertex);
            vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
            auto intersect = fitting::ray_triangle_intersect(ray_origin, ray_direction, vec3(v0), vec3(v1),
                                                             vec3(v2), false);
            // first is bool intersect, second is the distance t
            if (intersect.first == true)
            {
                // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the
                // intersection:
                // Check if in front or behind?
                if (intersect.second.get() <= 1e-4)
                {
                    continue; // the intersection is behind the vertex, we don't care about it
                }
                // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
                visible = false;
                break;
            }
        }
        visibility_ray.push_back(visible);
    }

    vector<vec4> wnd_coords; // will contain [x_wnd, y_wnd, z_ndc, 1/w_clip]
    for (auto&& vtx : mesh.vertices)
    {
        auto clip_coords = projection_matrix * view_model_matrix * vtx;
        clip_coords = divide_by_w(clip_coords);
        const vec2 screen_coords = clip_to_screen_space(clip_coords.x, clip_coords.y, image.cols, image.rows);
        clip_coords.x = screen_coords.x;
        clip_coords.y = screen_coords.y;
        wnd_coords.push_back(clip_coords);
    }

    // Go on with extracting: This only needs the rasteriser/FS, not the whole Renderer.
    const int tex_width = isomap_resolution;
    const int tex_height =
        isomap_resolution; // keeping this in case we need non-square texture maps at some point
    for (const auto& tvi : mesh.tvi)
    {
        if (visibility_ray[tvi[0]] && visibility_ray[tvi[1]] &&
            visibility_ray[tvi[2]]) // can also try using ||, but...
        {
            // Test with a rendered & re-extracted texture shows that we're off by a pixel or more,
            // definitely need to correct this. Probably here.
            // It looks like it is 1-2 pixels off. Definitely a bit more than 1.
            detail::Vertex<double> pa{
                vec4(mesh.texcoords[tvi[0]][0] * tex_width,
                                         mesh.texcoords[tvi[0]][1] * tex_height,
                     wnd_coords[tvi[0]].z, // z_ndc
                                         wnd_coords[tvi[0]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[0]].x / image.cols,
                    wnd_coords[tvi[0]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the
projected/rendered model triangle (in the input img). Normalised to 0,1.
                                        )};
            detail::Vertex<double> pb{
                vec4(mesh.texcoords[tvi[1]][0] * tex_width,
                                mesh.texcoords[tvi[1]][1] * tex_height,
                wnd_coords[tvi[1]].z, // z_ndc
                                wnd_coords[tvi[1]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[1]].x / image.cols,
                    wnd_coords[tvi[1]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the
projected/rendered model triangle (in the input img). Normalised to 0,1.
                                        )};
            detail::Vertex<double> pc{
                vec4(mesh.texcoords[tvi[2]][0] * tex_width,
                                mesh.texcoords[tvi[2]][1] * tex_height,
                wnd_coords[tvi[2]].z, // z_ndc
                                wnd_coords[tvi[2]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[2]].x / image.cols,
                    wnd_coords[tvi[2]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the
projected/rendered model triangle (in the input img). Normalised to 0,1.
                                        )};
            extraction_rasterizer.raster_triangle(pa, pb, pc, image_to_extract_from_as_tex);
        }
    }

    return extraction_rasterizer.colorbuffer;
};
*/

} /* namespace v2 */

namespace detail {

// Workaround for the pixels that don't get filled in extract_texture().
// There's a vertical line of missing values in the middle of the isomap,
// as well as a few pixels on a horizontal line around the mouth. They
// manifest themselves as black lines in the final isomap. This function
// just fills these missing values by interpolating between two neighbouring
// pixels. See GitHub issue #4.
inline core::Image4u interpolate_black_line(core::Image4u& isomap)
{
    // Replace the vertical black line ("missing data"):
    using RGBAType = Eigen::Matrix<std::uint8_t, 1, 4>;
    using Eigen::Map;
    const int col = isomap.cols / 2;
    for (int row = 0; row < isomap.rows; ++row)
    {
        if (isomap(row, col) == std::array<std::uint8_t, 4>{0, 0, 0, 0})
        {
            Eigen::Vector4f pixel_val =
                Map<const RGBAType>(isomap(row, col - 1).data(), 4).cast<float>() * 0.5f +
                Map<const RGBAType>(isomap(row, col + 1).data(), 4).cast<float>() * 0.5f;
            isomap(row,
                   col) = {static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]),
                           static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3])};
        }
    }

    // Replace the horizontal line around the mouth that occurs in the
    // isomaps of resolution 512x512 and higher:
    if (isomap.rows == 512) // num cols is 512 as well
    {
        const int r = 362;
        for (int c = 206; c <= 306; ++c)
        {
            if (isomap(r, c) == std::array<std::uint8_t, 4>{0, 0, 0, 0})
            {
                Eigen::Vector4f pixel_val =
                    Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f +
                    Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
                isomap(r, c) = {
                    static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]),
                    static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3])};
            }
        }
    }
    if (isomap.rows == 1024) // num cols is 1024 as well
    {
        int r = 724;
        for (int c = 437; c <= 587; ++c)
        {
            if (isomap(r, c) == std::array<std::uint8_t, 4>{0, 0, 0, 0})
            {
                Eigen::Vector4f pixel_val =
                    Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f +
                    Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
                isomap(r, c) = {
                    static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]),
                    static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3])};
            }
        }
        r = 725;
        for (int c = 411; c <= 613; ++c)
        {
            if (isomap(r, c) == std::array<std::uint8_t, 4>{0, 0, 0, 0})
            {
                Eigen::Vector4f pixel_val =
                    Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f +
                    Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
                isomap(r, c) = {
                    static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]),
                    static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3])};
            }
        }
    }
    // Higher resolutions are probably affected as well but not used so far in practice.

    return isomap;
};
} /* namespace detail */

} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_EXTRACTION_HPP_ */
