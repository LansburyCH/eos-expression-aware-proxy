import numpy as np
import eos
import h5py
import scipy.io

# This script converts the Basel Face Model 2017 (BFM2017, [1]) to the eos model format,
# specifically the files model2017-1_face12_nomouth.h5 and model2017-1_bfm_nomouth.h5 from the BFM2017 download.
#
# The BFM2017 does not come with texture (uv-) coordinates. If you have texture coordinates for the BFM, they can be
# added to the eos.morphablemodel.MorphableModel(...) constructor in the third argument. Note that eos only supports one
# uv-coordinate per vertex.
#
# [1]: Morphable Face Models - An Open Framework,
#      arXiv preprint, 2017.
#      http://faces.cs.unibas.ch/bfm/bfm2017.html

# Set this to the path of the model2017-1_bfm_nomouth.h5 or model2017-1_face12_nomouth.h5 file from the BFM2017 download:
bfm2017_file = r"../../bfm2017/model2017-1_bfm_nomouth.h5"

with h5py.File(bfm2017_file, 'r') as hf:
    # The PCA shape model:
    shape_mean = np.array(hf['shape/model/mean'])
    shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
    # Their basis is unit norm: np.linalg.norm(shape_pca_basis[:,0]) == ~1.0
    # And the basis vectors are orthogonal: np.dot(shape_pca_basis[:,0], shape_pca_basis[:,0]) == 1.0
    #                                       np.dot(shape_pca_basis[:,0], shape_pca_basis[:,1]) == 1e-10
    shape_pca_variance = np.array(hf['shape/model/pcaVariance']) # the PCA variances are the eigenvectors

    triangle_list = np.array(hf['shape/representer/cells'])

    shape_model = eos.morphablemodel.PcaModel(shape_mean, shape_orthogonal_pca_basis, shape_pca_variance, triangle_list.transpose().tolist())

    # PCA colour model:
    color_mean = np.array(hf['color/model/mean'])
    color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
    color_pca_variance = np.array(hf['color/model/pcaVariance'])

    color_model = eos.morphablemodel.PcaModel(color_mean, color_orthogonal_pca_basis, color_pca_variance, triangle_list.transpose().tolist())

    # PCA expression model:
    expression_mean = np.array(hf['expression/model/mean'])
    expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
    expression_pca_variance = np.array(hf['expression/model/pcaVariance'])

    expression_model = eos.morphablemodel.PcaModel(expression_mean, expression_pca_basis, expression_pca_variance, triangle_list.transpose().tolist())

    # # texture uv
    texture_uv = scipy.io.loadmat('../../bfm2017/texture_uv.mat')['texture_uv']
    texture_uv[:, 1] = 1 - texture_uv[:, 1]
    texture_uv = texture_uv.tolist()

    # print(help(eos.morphablemodel.MorphableModel))
    # print(type(texture_uv))
    # print(type(texture_uv[0]))
    # print(type(texture_uv[0][0]))
    # print(texture_uv[0])

    # Construct and save an eos model from the BFM data:
    model = eos.morphablemodel.MorphableModel(shape_model, expression_model, color_model, texture_coordinates = texture_uv) # uv-coordinates can be added here
    eos.morphablemodel.save_model(model, "../../bfm2017/bfm2017-1_bfm_nomouth.bin")
    print("Converted and saved model as bfm2017-1_bfm_nomouth.bin.")

    # also save data to .mat
    # scipy.io.savemat('../../bfm2017/bfm2017-1_bfm_nomouth.mat', {'shape_mean':shape_mean, 'shape_orthogonal_pca_basis':shape_orthogonal_pca_basis, 'shape_pca_variance':shape_pca_variance, 'triangle_list':triangle_list, 'color_mean':color_mean, 'color_orthogonal_pca_basis':color_orthogonal_pca_basis, 'color_pca_variance':color_pca_variance, 'expression_mean':expression_mean, 'expression_pca_basis':expression_pca_basis, 'expression_pca_variance':expression_pca_variance, 'texture_uv':texture_uv})
    # print("Converted and saved model as bfm2017-1_bfm_nomouth.mat.")


