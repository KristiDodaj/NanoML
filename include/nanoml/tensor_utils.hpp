#pragma once
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace ml {
    /**
    * Tensor utilities for CNN operations.
    * Since the library uses matrices throughout, these utilities help with
    * reshaping and manipulating data for CNN operations.
    */
    namespace tensor_utils {

        /**
        * @brief Convert a batch of images in matrix form to 4D tensor coordinates
        * @param batch Sample index in batch
        * @param channel Channel index
        * @param height Row index in image
        * @param width Column index in image
        * @param channels Number of channels
        * @param height_dim Height of the image
        * @param width_dim Width of the image
        * @return Linear index in the matrix representation
        */
        inline std::size_t idx4d_to_flat(
            std::size_t batch, std::size_t channel, std::size_t height, std::size_t width,
            std::size_t channels, std::size_t height_dim, std::size_t width_dim) {
            
            return width +
                width_dim * height +
                width_dim * height_dim * channel +
                width_dim * height_dim * channels * batch;
        }

        /**
        * @brief Calculate output dimensions after a convolution operation
        * @param input_dim Input dimension (height or width)
        * @param kernel_size Filter size
        * @param stride Stride value
        * @param padding Padding value
        * @return Output dimension
        */
        inline std::size_t get_conv_output_dim(
            std::size_t input_dim, std::size_t kernel_size,
            std::size_t stride, std::size_t padding) {
            
            return (input_dim + 2 * padding - kernel_size) / stride + 1;
        }
        
        /**
        * @brief Calculate output dimensions after a pooling operation
        * @param input_dim Input dimension (height or width) 
        * @param pool_size Pool size
        * @param stride Stride value
        * @return Output dimension
        */
        inline std::size_t get_pool_output_dim(
            std::size_t input_dim, std::size_t pool_size, std::size_t stride) {
            
            return (input_dim - pool_size) / stride + 1;
        }

        /**
        * @brief Extract a single image from a batch as a Matrix
        * @param batch_data Matrix containing batch of images [batch_size, channels*height*width]
        * @param batch_idx Index of the sample in batch
        * @param channels Number of channels
        * @param height Image height
        * @param width Image width
        * @return Matrix of shape [channels, height*width]
        */
        Matrix extract_image(
            const Matrix& batch_data, std::size_t batch_idx,
            std::size_t channels, std::size_t height, std::size_t width);
        
        /**
        * @brief Perform im2col operation for efficient convolution
        * @param input Input matrix [batch_size, channels*height*width]
        * @param batch_idx Batch index
        * @param channels Number of channels
        * @param height Image height
        * @param width Image width
        * @param kernel_size Filter size
        * @param stride Stride value
        * @param padding Padding value
        * @return Matrix with columns arranged for convolution
        */
        Matrix im2col(
            const Matrix& input, std::size_t batch_idx,
            std::size_t channels, std::size_t height, std::size_t width,
            std::size_t kernel_size, std::size_t stride, std::size_t padding);

        /**
        * @brief Perform col2im operation (reverse of im2col)
        * @param cols Column matrix from im2col
        * @param batch_size Batch size
        * @param channels Number of channels
        * @param height Image height
        * @param width Image width
        * @param kernel_size Filter size 
        * @param stride Stride value
        * @param padding Padding value
        * @return Original image matrix
        */
        Matrix col2im(
            const Matrix& cols, std::size_t batch_size,
            std::size_t channels, std::size_t height, std::size_t width,
            std::size_t kernel_size, std::size_t stride, std::size_t padding);
        
        /**
        * @brief Get the shapes of an image in the format [channels, height, width]
        * @param total_size Total size of the image (channels*height*width)
        * @return Vector with three elements [channels, height, width]
        */
        std::vector<std::size_t> infer_image_shape(std::size_t total_size);

        /**
        * @brief Reshape a matrix to a specified shape
        * @param input Input matrix
        * @param new_rows New number of rows
        * @param new_cols New number of columns
        * @return Reshaped matrix
        */
        Matrix reshape(const Matrix& input, std::size_t new_rows, std::size_t new_cols);

    }
}