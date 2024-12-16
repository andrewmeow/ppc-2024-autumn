// Copyright 2023 Nesterov Alexander
// здесь писать саму задачу

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp>
#include <sstream>
#include <vector>
using namespace std;
/*
std::vector<float> zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::create_gaussian_kernel(int radius,
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

void zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_rows(const std::vector<uint8_t>& input,
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
      temp[y * width + x] = sum;
    }
  }
}

void zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_columns(const std::vector<float>& temp,
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
      output[y * width + x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(sum)), 0, 255));
      ;
    }
  }
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::pre_processing() {
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

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  std::vector<float> vertical_kernel = horizontal_kernel;
  std::vector<float> temp(height_ * width_, 0.0f);
  convolve_rows(input_, height_, width_, horizontal_kernel, temp);
  convolve_columns(temp, height_, width_, vertical_kernel, result_);

  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  uint8_t* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      output_raw[i * width_ + j] = result_[i * width_ + j];
    }
  }
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int world_size = world.size();
  if (world.rank() == 0) {
    height_ = taskData->inputs_count[0];
    width_ = taskData->inputs_count[1];
    input_.resize(height_ * width_);
  }
  boost::mpi::broadcast(world, height_, 0);
  boost::mpi::broadcast(world, width_, 0);
  local_height_ = height_ / world_size + (world.rank() == (world_size - 1) ? 1 : 2);
  if (world.rank() == 0) {
    const uint8_t* raw_data = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        input_[i * width_ + j] = raw_data[i * width_ + j];
      }
    }
    int base_height = height_ / world_size;
    int remainder = height_ % world_size;
    int send_start = (base_height + remainder) * width_;
    for (int proc = 1; proc < world_size - 1; proc++) {
      world.send(proc, 0, input_.data() + proc * send_start - 1, local_height_ * width_);
    }
    world.send(world_size - 1, 0, input_.data() + (world_size - 1) * send_start - 1, base_height * width_ + 1);

    local_height_ = local_height_ + remainder - 1;
    local_input_.resize(local_height_);
    for (int i = 0; i < local_height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        local_input_[i * width_ + j] = input_[i * width_ + j];
      }
    }
    result_.resize(height_ * width_);
  } else {
    local_input_.resize(local_height_ * width_);
    world.recv(0, 0, local_input_.data(), local_height_ * width_);
  }
  return true;
}
/*
bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<uint8_t> local_res(local_height_ * width_);
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel =
      zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::create_gaussian_kernel(radius, sigma);
  std::vector<float> vertical_kernel = horizontal_kernel;
  std::vector<float> temp(local_height_ * width_, 0.0f);
  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_rows(local_input_, local_height_, width_,
                                                                         horizontal_kernel, temp);
  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_columns(temp, local_height_, width_,
                                                                            vertical_kernel, local_res);
  for (int i = 0; i < local_height_; ++i) {
    cout << "Proc #" << world.rank() << ": temp[" << i << "] = " << static_cast<int>(local_res[i]) << endl;
  }
  std::vector<uint8_t> buff;
  buff.insert(buff.end(), local_res.begin() + (world.rank() > 0 ? width_ : 0),
              local_res.end() - (world.rank() == world.size() - 1 ? 0 : width_));

  std::vector<int> recv_counts(world.size());
  boost::mpi::gather(world, static_cast<int>(buff.size()), recv_counts, 0);

  std::vector<int> displs(world.size(), 0);
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    result_.resize(displs.back() + recv_counts.back());
  }

  if (world.rank() == 0) {
    std::copy(buff.begin(), buff.end(), result_.begin());
    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<uint8_t> recv_buffer(recv_counts[proc]);
      world.recv(proc, 0, recv_buffer);
      std::copy(recv_buffer.begin(), recv_buffer.end(), result_.begin() + displs[proc]);
    }
  } else {
    world.send(0, 0, buff);
  }

  /* if (world.rank() == 0) {
     int base_height = height_ / world.size();
     std::copy(local_res.begin(), local_res.end(), result_.begin());
     for (int proc = 1; proc < world.size(); ++proc) {
       std::vector<uint8_t> buffer((base_height + (proc == world.size() - 1 ? 1 : 2)) * width_);
       world.recv(proc, 0, buffer);
       for (int i = width_; i < base_height * width_; i++) result_.push_back(buffer[i]);
     }
   } else {
     world.send(0, 0, local_res);
   }
   /*  if (world.rank() == 0) {
   result_.resize(height_ * width_);
   boost::mpi::gather(world, local_uint8.data(), local_uint8.size(), result_, 0);
 }
 else {
   boost::mpi::gather(world, local_uint8.data(), local_uint8.size(), 0);
 }
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<uint8_t> local_res(local_height_ * width_);
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel =
      zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::create_gaussian_kernel(radius, sigma);
  std::vector<float> vertical_kernel = horizontal_kernel;
  std::vector<float> temp(local_height_ * width_, 0.0f);
  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_rows(local_input_, local_height_, width_,
                                                                         horizontal_kernel, temp);
  for (int i = 0; i < local_height_ * width_; ++i) {
    cout << "Proc #" << world.rank() << ": temp[" << i << "] = " << temp[i] << endl;
  }
  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::convolve_columns(temp, local_height_, width_,
                                                                            vertical_kernel, local_res);

  if (world.rank() == 0) {
    int base_height = height_ / world.size();
    std::copy(local_res.begin(), local_res.end(), result_.begin());
    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<uint8_t> buffer((base_height + (proc == world.size() - 1 ? 1 : 2)) * width_);
      world.recv(proc, 0, buffer);
      for (int i = width_; i < base_height * width_; i++) result_.push_back(buffer[i]);
    }
  } else {
    world.send(0, 0, local_res);
  }
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    uint8_t* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    for (size_t i = 0; i < height_; i++) {
      for (size_t j = 0; j < width_; j++) {
        output_raw[i * width_ + j] = result_[i * width_ + j];
      }
    }
  }
  return true;
}
/*
if (world.rank() == 0) {
// Объединение собранных данных в итоговое изображение
uint8_t* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
int base_height = height_ / world.size();
int remainder = height_ % world.size();
int offset = 0;

// Перенос строк с учетом возможного остатка для первого процесса
for (int proc = 0; proc < world.size(); proc++) {
  int rows_to_copy = base_height + (proc == 0 ? remainder + 1 : (proc == world.size() - 1) ? 1 : 2);
  for (int i = 0; i < rows_to_copy; i++) {
    std::copy(result_.begin() + offset + i * width_, result_.begin() + offset + (i + 1) * width_,
              output_raw + (base_height * proc +
                            (proc == 0                    ? remainder + 1
                             : (proc == world.size() - 1) ? 1
                                                          : 2) +
                            i) *
                               width_);
  }
  offset += rows_to_copy * width_;
}
}
return true;
}*/
