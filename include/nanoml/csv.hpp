#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"

namespace ml {
    class CSVReader {
        public:
            CSVReader() = default;

            explicit CSVReader(const std::string& filename, bool header = true);

            bool load(const std::string& filename, bool header = true);

            [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

            [[nodiscard]] const std::vector<std::string>& headers() const noexcept { return headers_; }

            [[nodiscard]] std::size_t rows() const noexcept { return data_.size(); }

            [[nodiscard]] std::size_t cols() const noexcept { return headers_.size(); }

            [[nodiscard]] const std::vector<std::vector<std::string>>& data() const noexcept { return data_; }
            
            [[nodiscard]] Vector get_column(std::size_t col_idx) const;
            [[nodiscard]] Vector get_column(const std::string& col_name) const;
            
            [[nodiscard]] Matrix get_columns(const std::vector<std::size_t>& col_indices) const;
            [[nodiscard]] Matrix get_columns(const std::vector<std::string>& col_names) const;
            
            [[nodiscard]] Matrix to_matrix() const;
            
        private:
            bool loaded_ = false;
            bool has_header_ = false;
            std::vector<std::string> headers_;
            std::vector<std::vector<std::string>> data_;
            
            std::vector<std::string> split(const std::string& line, char delimiter = ',');
            std::size_t find_column_index(const std::string& col_name) const;
        };
}