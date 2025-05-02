#include "nanoml/csv.hpp"
#include <sstream>
#include <cassert>

namespace ml {
    CSVReader::CSVReader(const std::string& filename, bool header) {
        load(filename, header);
    }

    bool CSVReader::load(const std::string& filename, bool header) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            loaded_ = false;
            return false;
        }

        headers_.clear();
        data_.clear();
        
        std::string line;
        has_header_ = header;

        if (has_header_ && std::getline(file, line)) {
            headers_ = split(line);
        }

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::vector<std::string> row = split(line);
            data_.push_back(row);

            if (headers_.empty()) {
                headers_.resize(row.size());
                for (std::size_t i = 0; i < row.size(); ++i) {
                    headers_[i] = "Column" + std::to_string(i);
                }
            }
        }
        
        loaded_ = true;
        return true;
    }

    std::vector<std::string> CSVReader::split(const std::string& line, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream token_stream(line);
        
        while (std::getline(token_stream, token, delimiter)) {
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            
            if (start != std::string::npos && end != std::string::npos) {
                token = token.substr(start, end - start + 1);
            } else if (start == std::string::npos) {
                token = "";
            }
            
            tokens.push_back(token);
        }
        
        return tokens;
    }

    std::size_t CSVReader::find_column_index(const std::string& col_name) const {
        for (std::size_t i = 0; i < headers_.size(); ++i) {
            if (headers_[i] == col_name) {
                return i;
            }
        }
        return headers_.size();
    }

    Vector CSVReader::get_column(std::size_t col_idx) const {
        assert(loaded_ && "CSV file not loaded");
        assert(col_idx < cols() && "Column index out of bounds");
        
        Vector column(rows());
        
        for (std::size_t i = 0; i < data_.size(); ++i) {
            if (col_idx < data_[i].size()) {
                try {
                    column[i] = std::stod(data_[i][col_idx]);
                } catch (const std::exception& e) {
                    column[i] = 0.0;
                }
            }
        }
        
        return column;
    }

    Vector CSVReader::get_column(const std::string& col_name) const {
        assert(loaded_ && "CSV file not loaded");
        std::size_t col_idx = find_column_index(col_name);
        assert(col_idx < cols() && "Column name not found");
        
        return get_column(col_idx);
    }

    Matrix CSVReader::get_columns(const std::vector<std::size_t>& col_indices) const {
        assert(loaded_ && "CSV file not loaded");
        
        Matrix result(rows(), col_indices.size());
        
        for (std::size_t j = 0; j < col_indices.size(); ++j) {
            std::size_t col_idx = col_indices[j];
            assert(col_idx < cols() && "Column index out of bounds");
            
            for (std::size_t i = 0; i < data_.size(); ++i) {
                if (col_idx < data_[i].size()) {
                    try {
                        result(i, j) = std::stod(data_[i][col_idx]);
                    } catch (const std::exception& e) {
                        result(i, j) = 0.0;
                    }
                }
            }
        }
        
        return result;
    }

    Matrix CSVReader::get_columns(const std::vector<std::string>& col_names) const {
        assert(loaded_ && "CSV file not loaded");
        
        std::vector<std::size_t> col_indices;
        col_indices.reserve(col_names.size());
        
        for (const auto& name : col_names) {
            std::size_t idx = find_column_index(name);
            assert(idx < cols() && "Column name not found");
            col_indices.push_back(idx);
        }
        
        return get_columns(col_indices);
    }

    Matrix CSVReader::to_matrix() const {
        assert(loaded_ && "CSV file not loaded");
        
        std::size_t num_rows = data_.size();
        std::size_t num_cols = 0;
        
        for (const auto& row : data_) {
            num_cols = std::max(num_cols, row.size());
        }
        
        Matrix mat(num_rows, num_cols);
        
        for (std::size_t i = 0; i < num_rows; ++i) {
            for (std::size_t j = 0; j < data_[i].size() && j < num_cols; ++j) {
                try {
                    mat(i, j) = std::stod(data_[i][j]);
                } catch (const std::exception& e) {
                    mat(i, j) = 0.0;
                }
            }
        }
        
        return mat;
    }
}