// Copyright 2023 Nesterov Alexander
// здесь писать саму задачу
#include "mpi/zolotareva_a_smoothing_image/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <seq/zolotareva_a_smoothing_image/include/ops_seq.hpp>
#include <sstream>
#include <vector>
using namespace std;
std::vector<float> zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::create_gaussian_kernel(int radius,
                                                                                                   float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float norm = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
    norm += kernel[i + radius];
  }
  for (float& val : kernel) {
    val /= norm;
  }
  return kernel;
}

void zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(const std::vector<uint8_t>& input,
                                                                            int height, int width,
                                                                            const std::vector<float>& kernel,
                                                                            std::vector<float>& temp) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_x = std::clamp(x + k, 0, width - 1);
        sum += input[y * width + pixel_x] * kernel[k + kernel_radius];
      }
      temp[y * width + x] = int(sum);
    }
  }
}

void zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(const std::vector<float>& temp,
                                                                               int height, int width,
                                                                               const std::vector<float>& kernel,
                                                                               std::vector<uint8_t>& output) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_y = std::clamp(y + k, 0, height - 1);
        sum += temp[pixel_y * width + x] * kernel[k + kernel_radius];
      }
      output[y * width + x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(int(sum))), 0, 255));
      ;
    }
  }
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  height_ = taskData->inputs_count[0];
  width_ = taskData->inputs_count[1];
  input_.resize(height_ * width_);
  const uint8_t* raw_data = reinterpret_cast<uint8_t*>(taskData->inputs[0]);

  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      input_[i * width_ + j] = raw_data[i * width_ + j];
    }
  }
  result_.resize(height_ * width_);
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  const std::vector<float>& vertical_kernel = horizontal_kernel;
  std::vector<float> temp(height_ * width_, 0.0f);
  convolve_rows(input_, height_, width_, horizontal_kernel, temp);
  convolve_columns(temp, height_, width_, vertical_kernel, result_);

  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      output_raw[i * width_ + j] = result_[i * width_ + j];
    }
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int world_size = world.size();
  if (world.rank() == 0) {
    height_ = taskData->inputs_count[0];
    width_ = taskData->inputs_count[1];
    input_.clear();
    input_.resize(height_ * width_);
    result_.resize(height_ * width_);
    const uint8_t* raw_data = taskData->inputs[0];
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        input_[i * width_ + j] = raw_data[i * width_ + j];
      }
    }
    if (world_size == 1) {
      return true;
    }
  }
  boost::mpi::broadcast(world, height_, 0);
  boost::mpi::broadcast(world, width_, 0);
  world.barrier();
  if (world.rank() == 0) {
    int base_height = height_ / world_size;
    int remainder = height_ % world_size;
    int send_start = (base_height + remainder - 1) * width_;
    for (int proc = 1; proc < world_size - 1; proc++) {
      world.send(proc, 0, input_.data() + send_start, (base_height + 2) * width_);
      send_start += base_height * width_;
    }
    world.send(world_size - 1, 0, input_.data() + send_start, base_height * width_ + width_);

    local_height_ = base_height + remainder + 1;
    local_input_.resize(local_height_);
    for (int i = 0; i < local_height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        local_input_[i * width_ + j] = input_[i * width_ + j];
      }
    }
  } else {
    if (world.rank() == world_size - 1)
      local_height_ = height_ / world_size + 1;
    else
      local_height_ = height_ / world_size + 2;
    local_input_.resize(local_height_ * width_);
    world.recv(0, 0, local_input_.data(), local_height_ * width_);
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<uint8_t> local_res(local_height_ * width_);
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel =
      zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::create_gaussian_kernel(radius, sigma);

  if (world.size() == 1) {
    result_.clear();
    std::vector<float>& vertical_kernel = horizontal_kernel;
    std::vector<float> temp(height_ * width_, 0.0f);
    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(input_, height_, width_, horizontal_kernel,
                                                                           temp);

    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(temp, height_, width_, vertical_kernel,
                                                                              result_);
    return true;
  }

  std::vector<float>& vertical_kernel = horizontal_kernel;
  std::vector<float> temp(local_height_ * width_, 0.0f);
  zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(local_input_, local_height_, width_,
                                                                         horizontal_kernel, temp);

  zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(temp, local_height_, width_,
                                                                            vertical_kernel, local_res);

  if (world.rank() == 0) {
    result_.clear();
    int base_height = height_ / world.size();
    int send_start = (base_height + (height_ % world.size())) * width_;
    std::copy(local_res.begin(), local_res.end() - width_, result_.begin());
    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<uint8_t> buffer((base_height + (proc == (world.size() - 1) ? 1 : 2)) * width_);
      world.recv(proc, 1, buffer);
      std::copy(buffer.begin() + width_, buffer.end() - (proc == world.size() - 1 ? 0 : width_),
                result_.begin() + send_start + (proc - 1) * base_height * width_);
    }
  } else {
    world.send(0, 1, local_res);
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    for (int i = 0; i < height_; i++) {
      for (int j = 0; j < width_; j++) {
        output_raw[i * width_ + j] = result_[i * width_ + j];
      }
    }
  }
  return true;
}