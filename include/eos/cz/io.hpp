#pragma once

#ifndef CZ_IO_HPP
#define CZ_IO_HPP


#include <string>
#include <vector>


namespace cz {
	namespace io {


		// read coefficients from file
		std::vector<float> read_coeffs_from_file(std::string fp) {
			std::vector<float> coeffs;

			std::ifstream file;
			file.open(fp);
			if (!file) {
				std::string err_msg = "read_coeffs_from_file: No such file as " + fp;
				perror(err_msg.c_str());
			}

			float datum;
			while (file >> datum)
				coeffs.push_back(datum);

			return coeffs;
		}


		// write coeffs to file
		void write_coeffs_to_file(std::string fp, std::vector<float> coeffs) {
			std::ofstream outFile;
			outFile.open(fp);
			for (const auto &val : coeffs) outFile << val << "\n";
			outFile.close();

			return;
		}


		// write glm matrix to file
		void write_glm4x4_to_file(std::string fp, glm::mat4x4 glm_matrix) {
			std::ofstream outFile;
			outFile.open(fp);
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					outFile << glm_matrix[j][i] << " ";
				}
				outFile << "\n";
			}
			outFile.close();

			return;
		}


		// write array to binary file
		template <typename T>
		void BinaryFile_Write(std::string fn, T* data, uint64_t num_dim, uint64_t *dims) {
			std::ofstream fout(fn, std::ios::binary);
			std::string data_type = typeid(data[0]).name();
			uint64_t data_type_byte_num = data_type.size();
			uint64_t num_element = 1;
			for (int i = 0; i < num_dim; ++i) {
				num_element *= dims[i];
			}
			fout.write((char*)&num_dim, sizeof(num_dim));
			fout.write((char*)dims, sizeof(dims[0]) * num_dim);
			fout.write((char*)&data_type_byte_num, sizeof(data_type_byte_num));
			fout.write(data_type.c_str(), sizeof(char) * data_type_byte_num);
			fout.write((char*)data, sizeof(data[0]) * num_element);
			fout.close();

			return;
		}

	} /* namespace io */
} /* namespace cz */

#endif