#include "nanoml/tensor_utils.hpp"

namespace ml {
    namespace tensor_utils {

        Matrix extract_image(
            const Matrix& batch_data, std::size_t batch_idx,
            std::size_t channels, std::size_t height, std::size_t width) {
            
            std::size_t img_size = channels * height * width;
            Matrix result(channels, height * width, 0.0);

            for (std::size_t c = 0; c < channels; ++c) {
                for (std::size_t hw = 0; hw < height * width; ++hw) {
                    std::size_t flat_idx = c * height * width + hw;
                    result(c, hw) = batch_data(batch_idx, flat_idx);
                }
            }
            
            return result;
        }

        Matrix im2col(
            const Matrix& input, std::size_t batch_idx,
            std::size_t channels, std::size_t height, std::size_t width,
            std::size_t kernel_size, std::size_t stride, std::size_t padding) {

            std::size_t output_h = get_conv_output_dim(height, kernel_size, stride, padding);
            std::size_t output_w = get_conv_output_dim(width, kernel_size, stride, padding);
            
            // Each column of the result will contain the values in one receptive field
            // Number of rows = kernel_size * kernel_size * channels
            // Number of columns = output_h * output_w
            Matrix result(kernel_size * kernel_size * channels, output_h * output_w, 0.0);
            
            std::size_t col_idx = 0;
            for (std::size_t out_y = 0; out_y < output_h; ++out_y) {
                for (std::size_t out_x = 0; out_x < output_w; ++out_x) {
                    std::size_t in_y_start = out_y * stride;
                    std::size_t in_x_start = out_x * stride;
                    
                    std::size_t row_idx = 0;
                    for (std::size_t c = 0; c < channels; ++c) {
                        for (std::size_t ky = 0; ky < kernel_size; ++ky) {
                            for (std::size_t kx = 0; kx < kernel_size; ++kx) {
                                int in_y = static_cast<int>(in_y_start + ky) - static_cast<int>(padding);
                                int in_x = static_cast<int>(in_x_start + kx) - static_cast<int>(padding);

                                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                    std::size_t flat_idx = c * height * width + in_y * width + in_x;
                                    result(row_idx, col_idx) = input(batch_idx, flat_idx);
                                }
                                row_idx++;
                            }
                        }
                    }
                    col_idx++;
                }
            }
            
            return result;
        }

        Matrix col2im(
            const Matrix& cols, std::size_t batch_size,
            std::size_t channels, std::size_t height, std::size_t width,
            std::size_t kernel_size, std::size_t stride, std::size_t padding) {

            Matrix result(batch_size, channels * height * width, 0.0);

            std::size_t output_h = get_conv_output_dim(height, kernel_size, stride, padding);
            std::size_t output_w = get_conv_output_dim(width, kernel_size, stride, padding);

            for (std::size_t batch = 0; batch < batch_size; ++batch) {
                std::vector<double> img_grad(channels * height * width, 0.0);
                
                std::size_t col_idx = 0;
                for (std::size_t out_y = 0; out_y < output_h; ++out_y) {
                    for (std::size_t out_x = 0; out_x < output_w; ++out_x) {
                        std::size_t in_y_start = out_y * stride;
                        std::size_t in_x_start = out_x * stride;
                        
                        std::size_t row_idx = 0;
                        for (std::size_t c = 0; c < channels; ++c) {
                            for (std::size_t ky = 0; ky < kernel_size; ++ky) {
                                for (std::size_t kx = 0; kx < kernel_size; ++kx) {
                                    int in_y = static_cast<int>(in_y_start + ky) - static_cast<int>(padding);
                                    int in_x = static_cast<int>(in_x_start + kx) - static_cast<int>(padding);

                                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                        std::size_t flat_idx = c * height * width + in_y * width + in_x;
                                        img_grad[flat_idx] += cols(row_idx, col_idx);
                                    }
                                    row_idx++;
                                }
                            }
                        }
                        col_idx++;
                    }
                }

                for (std::size_t i = 0; i < channels * height * width; ++i) {
                    result(batch, i) = img_grad[i];
                }
            }
            
            return result;
        }

        std::vector<std::size_t> infer_image_shape(std::size_t total_size) {
            
            std::vector<std::size_t> common_channel_counts = {1, 3, 4};
            
            for (const auto& channels : common_channel_counts) {
                std::size_t pixels = total_size / channels;
                std::size_t side_length = static_cast<std::size_t>(std::sqrt(pixels));
                
                if (side_length * side_length == pixels) {
                    return {channels, side_length, side_length};
                }
            }
            
            std::size_t side_length = static_cast<std::size_t>(std::sqrt(total_size));
            return {1, side_length, side_length};
        }

        Matrix reshape(const Matrix& input, std::size_t new_rows, std::size_t new_cols) {
            assert(input.rows() * input.cols() == new_rows * new_cols && "Cannot reshape to different total size");
            
            Matrix result(new_rows, new_cols);
            
            for (std::size_t i = 0; i < input.rows() * input.cols(); ++i) {
                std::size_t original_row = i / input.cols();
                std::size_t original_col = i % input.cols();
                
                std::size_t new_row = i / new_cols;
                std::size_t new_col = i % new_cols;
                
                result(new_row, new_col) = input(original_row, original_col);
            }
            
            return result;
        }
    }
}