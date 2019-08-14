#pragma once

#ifndef CZ_UTIL_HPP
#define CZ_UTIL_HPP

#include "../morphablemodel/Blendshape.hpp"
#include "../morphablemodel/PcaModel.hpp"
#include "../morphablemodel/ExpressionModel.hpp"
#include "../cpp17/variant.hpp"

#include <string>
#include <vector>

namespace cz {
namespace util {

// get blendshapes from expression model
std::vector<eos::morphablemodel::Blendshape>
expression_pca_to_blendshapes(const eos::morphablemodel::ExpressionModel expression_model,
                              int num_pca_keep = -1)
{
    std::vector<eos::morphablemodel::Blendshape> blendshapes;

    if (eos::cpp17::holds_alternative<eos::morphablemodel::PcaModel>(expression_model))
    {
        const auto& pca_expression_model = eos::cpp17::get<eos::morphablemodel::PcaModel>(expression_model);

        if (num_pca_keep < 0)
        {
            num_pca_keep = pca_expression_model.get_num_principal_components();
        }
        blendshapes.resize(num_pca_keep);

        for (int i = 0; i < blendshapes.size(); ++i)
        {
            blendshapes[i].name = "pca_" + std::to_string(i);
            blendshapes[i].deformation = pca_expression_model.get_orthonormal_pca_basis().col(i);
        }
    } 
	else if (eos::cpp17::holds_alternative<eos::morphablemodel::Blendshapes>(expression_model))
    {
        blendshapes = eos::cpp17::get<eos::morphablemodel::Blendshapes>(expression_model);
    } 
	else
    {
        throw std::runtime_error("The given ExpressionModel doesn't contain an expression model in the form "
                                 "of a PcaModel or Blendshapes.");
    }
    return blendshapes;
}

} /* namespace util */
} /* namespace cz */

#endif