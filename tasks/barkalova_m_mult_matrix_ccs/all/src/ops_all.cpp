/*
#include "barkalova_m_mult_matrix_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include "barkalova_m_mult_matrix_ccs/common/include/common.hpp"

namespace barkalova_m_mult_matrix_ccs {

BarkalovaMMultMatrixCcsALL::BarkalovaMMultMatrixCcsALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCSMatrix{};
}

bool BarkalovaMMultMatrixCcsALL::ValidationImpl() {
  const auto &[A, B] = GetInput();
  if (A.cols != B.rows) {
    return false;
  }
  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  if (A.col_ptrs.size() != static_cast<size_t>(A.cols) + 1) {
    return false;
  }
  if (B.col_ptrs.size() != static_cast<size_t>(B.cols) + 1) {
    return false;
  }
  if (A.col_ptrs[0] != 0 || B.col_ptrs[0] != 0) {
    return false;
  }
  if (static_cast<size_t>(A.nnz) != A.values.size()) {
    return false;
  }
  if (static_cast<size_t>(B.nnz) != B.values.size()) {
    return false;
  }
  return true;
}

bool BarkalovaMMultMatrixCcsALL::PreProcessingImpl() {
  return true;
}

namespace {
constexpr double kEpsilon = 1e-10;

void TransponirMatr(const CCSMatrix &a, CCSMatrix &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      Complex val = a.values[i];
      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

bool IsNonZero(const Complex &val) {
  return std::abs(val.real()) > kEpsilon || std::abs(val.imag()) > kEpsilon;
}

Complex ComputeScalarProduct(const CCSMatrix &at, const CCSMatrix &b, int row_a, int col_b) {
  Complex sum = Complex(0.0, 0.0);

  int ks = at.col_ptrs[row_a];
  int ls = b.col_ptrs[col_b];
  int kf = at.col_ptrs[row_a + 1];
  int lf = b.col_ptrs[col_b + 1];

  while ((ks < kf) && (ls < lf)) {
    if (at.row_indices[ks] < b.row_indices[ls]) {
      ks++;
    } else if (at.row_indices[ks] > b.row_indices[ls]) {
      ls++;
    } else {
      sum += at.values[ks] * b.values[ls];
      ks++;
      ls++;
    }
  }

  return sum;
}

void ProcessColumn(int j, const CCSMatrix &at, const CCSMatrix &b, std::vector<int> &out_rows,
                   std::vector<Complex> &out_vals) {
  out_rows.reserve(100);
  out_vals.reserve(100);

  for (int i = 0; i < at.cols; i++) {
    Complex sum = ComputeScalarProduct(at, b, i, j);
    if (IsNonZero(sum)) {
      out_rows.push_back(i);
      out_vals.push_back(sum);
    }
  }
}

}  // namespace

bool BarkalovaMMultMatrixCcsALL::RunImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;

  try {
    // Получаем rank и size (MPI уже инициализирован тестовым фреймворком)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Транспонирование матрицы A
    CCSMatrix at;
    TransponirMatr(a, at);

    const int total_cols = b.cols;
    const int total_rows = a.rows;

    // Распределение столбцов между MPI процессами
    int cols_per_process = total_cols / size;
    int remainder = total_cols % size;

    int start_col = rank * cols_per_process + std::min(rank, remainder);
    int local_cols = cols_per_process + (rank < remainder ? 1 : 0);

    // Локальные данные для каждого MPI процесса
    std::vector<std::vector<int>> col_rows(local_cols);
    std::vector<std::vector<Complex>> col_vals(local_cols);

    // TBB параллелизация внутри MPI процесса
    tbb::parallel_for(tbb::blocked_range<int>(0, local_cols),
        [&](const tbb::blocked_range<int> &range) {
          for (int j_local = range.begin(); j_local < range.end(); ++j_local) {
            int global_col = start_col + j_local;
            ProcessColumn(global_col, at, b, col_rows[j_local], col_vals[j_local]);
          }
        });

    // Сборка локальных данных
    std::vector<int> local_row_indices;
    std::vector<Complex> local_values;

    for (int j = 0; j < local_cols; ++j) {
      local_row_indices.insert(local_row_indices.end(), col_rows[j].begin(), col_rows[j].end());
      local_values.insert(local_values.end(), col_vals[j].begin(), col_vals[j].end());
    }
    int local_nnz = static_cast<int>(local_values.size());

    // Сбор количества ненулевых элементов для каждого столбца
    std::vector<int> global_col_nnz(total_cols, 0);
    for (int j = 0; j < local_cols; ++j) {
      global_col_nnz[start_col + j] = static_cast<int>(col_rows[j].size());
    }

    std::vector<int> recv_col_nnz(total_cols);
    MPI_Allreduce(global_col_nnz.data(), recv_col_nnz.data(), total_cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Формируем глобальные col_ptrs (одинаковые для всех процессов)
    std::vector<int> global_col_ptrs = {0};
    for (int j = 0; j < total_cols; ++j) {
      global_col_ptrs.push_back(global_col_ptrs.back() + recv_col_nnz[j]);
    }
    int total_nnz = global_col_ptrs.back();

    // Сбор всех данных на процесс 0
    std::vector<int> global_row_indices;
    std::vector<Complex> global_values;

    if (rank == 0) {
      global_row_indices.resize(total_nnz);
      global_values.resize(total_nnz);

      // Копируем данные процесса 0
      std::copy(local_row_indices.begin(), local_row_indices.end(), global_row_indices.begin());
      std::copy(local_values.begin(), local_values.end(), global_values.begin());

      int offset = local_nnz;

      // Принимаем данные от остальных процессов
      for (int p = 1; p < size; ++p) {
        int p_nnz = 0;
        MPI_Recv(&p_nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> p_row_indices(p_nnz);
        MPI_Recv(p_row_indices.data(), p_nnz, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<double> p_values_real(p_nnz), p_values_imag(p_nnz);
        MPI_Recv(p_values_real.data(), p_nnz, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p_values_imag.data(), p_nnz, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::copy(p_row_indices.begin(), p_row_indices.end(), global_row_indices.begin() + offset);
        for (int i = 0; i < p_nnz; ++i) {
          global_values[offset + i] = Complex(p_values_real[i], p_values_imag[i]);
        }
        offset += p_nnz;
      }
    } else {
      // Отправляем данные на процесс 0
      MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(local_row_indices.data(), local_nnz, MPI_INT, 0, 1, MPI_COMM_WORLD);

      std::vector<double> local_values_real(local_nnz), local_values_imag(local_nnz);
      for (int i = 0; i < local_nnz; ++i) {
        local_values_real[i] = local_values[i].real();
        local_values_imag[i] = local_values[i].imag();
      }
      MPI_Send(local_values_real.data(), local_nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
      MPI_Send(local_values_imag.data(), local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    // Широковещательная рассылка результата всем процессам
    int bcast_rows = total_rows;
    int bcast_cols = total_cols;
    int bcast_nnz = total_nnz;

    MPI_Bcast(&bcast_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bcast_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bcast_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // col_ptrs
    int col_ptrs_size = static_cast<int>(global_col_ptrs.size());
    MPI_Bcast(&col_ptrs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      global_col_ptrs.resize(col_ptrs_size);
    }
    MPI_Bcast(global_col_ptrs.data(), col_ptrs_size, MPI_INT, 0, MPI_COMM_WORLD);

    // row_indices
    if (rank != 0) {
      global_row_indices.resize(bcast_nnz);
    }
    MPI_Bcast(global_row_indices.data(), bcast_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    // values (complex)
    std::vector<double> global_values_real(bcast_nnz), global_values_imag(bcast_nnz);
    if (rank == 0) {
      for (int i = 0; i < bcast_nnz; ++i) {
        global_values_real[i] = global_values[i].real();
        global_values_imag[i] = global_values[i].imag();
      }
    }
    MPI_Bcast(global_values_real.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_values_imag.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      global_values.resize(bcast_nnz);
      for (int i = 0; i < bcast_nnz; ++i) {
        global_values[i] = Complex(global_values_real[i], global_values_imag[i]);
      }
    }

    // Формируем итоговую матрицу
    CCSMatrix c;
    c.rows = bcast_rows;
    c.cols = bcast_cols;
    c.nnz = bcast_nnz;
    c.values = std::move(global_values);
    c.row_indices = std::move(global_row_indices);
    c.col_ptrs = std::move(global_col_ptrs);

    GetOutput() = c;
    return true;

  } catch (const std::exception &) {
    return false;
  }
}

bool BarkalovaMMultMatrixCcsALL::PostProcessingImpl() {
  const auto &c = GetOutput();

  // Базовая проверка
  if (c.rows <= 0 || c.cols <= 0) {
    return false;
  }

  // Проверка col_ptrs
  if (c.col_ptrs.size() != static_cast<size_t>(c.cols) + 1) {
    return false;
  }

  if (c.col_ptrs.empty()) {
    return false;
  }

  // Проверка монотонного возрастания col_ptrs
  for (size_t i = 1; i < c.col_ptrs.size(); ++i) {
    if (c.col_ptrs[i] < c.col_ptrs[i - 1]) {
      return false;
    }
  }

  // Проверка соответствия nnz
  if (c.nnz != static_cast<int>(c.values.size()) ||
      c.nnz != static_cast<int>(c.row_indices.size())) {
    return false;
  }

  // Проверка, что первый элемент col_ptrs = 0
  if (c.col_ptrs[0] != 0) {
    return false;
  }

  // Проверка, что последний элемент col_ptrs = nnz
  if (c.col_ptrs.back() != c.nnz) {
    return false;
  }

  return true;
}

}  // namespace barkalova_m_mult_matrix_ccs
*/

// mpi + omp без gather
/*
#include "barkalova_m_mult_matrix_ccs/all/include/ops_all.hpp"

#include <mpi.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include "barkalova_m_mult_matrix_ccs/common/include/common.hpp"

namespace barkalova_m_mult_matrix_ccs {

BarkalovaMMultMatrixCcsALL::BarkalovaMMultMatrixCcsALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCSMatrix{};
}

bool BarkalovaMMultMatrixCcsALL::ValidationImpl() {
  const auto &[A, B] = GetInput();
  if (A.cols != B.rows) {
    return false;
  }
  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  if (A.col_ptrs.size() != static_cast<size_t>(A.cols) + 1) {
    return false;
  }
  if (B.col_ptrs.size() != static_cast<size_t>(B.cols) + 1) {
    return false;
  }
  if (A.col_ptrs[0] != 0 || B.col_ptrs[0] != 0) {
    return false;
  }
  if (static_cast<size_t>(A.nnz) != A.values.size()) {
    return false;
  }
  if (static_cast<size_t>(B.nnz) != B.values.size()) {
    return false;
  }
  return true;
}

bool BarkalovaMMultMatrixCcsALL::PreProcessingImpl() {
  return true;
}

namespace {
constexpr double kEpsilon = 1e-10;

void TransponirMatr(const CCSMatrix &a, CCSMatrix &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      Complex val = a.values[i];
      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

bool IsNonZero(const Complex &val) {
  return std::abs(val.real()) > kEpsilon || std::abs(val.imag()) > kEpsilon;
}

Complex ComputeScalarProduct(const CCSMatrix &at, const CCSMatrix &b, int row_a, int col_b) {
  Complex sum = Complex(0.0, 0.0);

  int ks = at.col_ptrs[row_a];
  int ls = b.col_ptrs[col_b];
  int kf = at.col_ptrs[row_a + 1];
  int lf = b.col_ptrs[col_b + 1];

  while ((ks < kf) && (ls < lf)) {
    if (at.row_indices[ks] < b.row_indices[ls]) {
      ks++;
    } else if (at.row_indices[ks] > b.row_indices[ls]) {
      ls++;
    } else {
      sum += at.values[ks] * b.values[ls];
      ks++;
      ls++;
    }
  }

  return sum;
}

}  // namespace

bool BarkalovaMMultMatrixCcsALL::RunImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;

  try {
    // Получаем rank и size (MPI уже инициализирован тестовым фреймворком)
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Транспонирование матрицы A
    CCSMatrix at;
    TransponirMatr(a, at);

    const int total_cols = b.cols;
    const int total_rows = a.rows;

    // Распределение столбцов между MPI процессами
    int cols_per_process = total_cols / size;
    int remainder = total_cols % size;

    int start_col = rank * cols_per_process + std::min(rank, remainder);
    int local_cols = cols_per_process + (rank < remainder ? 1 : 0);

    // Локальные данные для каждого MPI процесса
    std::vector<std::vector<int>> col_rows(local_cols);
    std::vector<std::vector<Complex>> col_vals(local_cols);

    // OMP параллелизация внутри MPI процесса
    #pragma omp parallel for schedule(static) default(none) shared(local_cols, start_col, at, b, col_rows, col_vals)
    for (int j_local = 0; j_local < local_cols; ++j_local) {
      int global_col = start_col + j_local;

      std::vector<int> rows;
      std::vector<Complex> vals;
      rows.reserve(100);
      vals.reserve(100);

      for (int i = 0; i < at.cols; i++) {
        Complex sum = ComputeScalarProduct(at, b, i, global_col);
        if (IsNonZero(sum)) {
          rows.push_back(i);
          vals.push_back(sum);
        }
      }

      col_rows[j_local] = std::move(rows);
      col_vals[j_local] = std::move(vals);
    }

    // Сборка локальных данных
    std::vector<int> local_row_indices;
    std::vector<Complex> local_values;

    for (int j = 0; j < local_cols; ++j) {
      local_row_indices.insert(local_row_indices.end(), col_rows[j].begin(), col_rows[j].end());
      local_values.insert(local_values.end(), col_vals[j].begin(), col_vals[j].end());
    }
    int local_nnz = static_cast<int>(local_values.size());

    // Сбор количества ненулевых элементов для каждого столбца
    std::vector<int> global_col_nnz(total_cols, 0);
    for (int j = 0; j < local_cols; ++j) {
      global_col_nnz[start_col + j] = static_cast<int>(col_rows[j].size());
    }

    std::vector<int> recv_col_nnz(total_cols);
    MPI_Allreduce(global_col_nnz.data(), recv_col_nnz.data(), total_cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Формируем глобальные col_ptrs (одинаковые для всех процессов)
    std::vector<int> global_col_ptrs = {0};
    for (int j = 0; j < total_cols; ++j) {
      global_col_ptrs.push_back(global_col_ptrs.back() + recv_col_nnz[j]);
    }
    int total_nnz = global_col_ptrs.back();

    // Сбор всех данных на процесс 0
    std::vector<int> global_row_indices;
    std::vector<Complex> global_values;

    if (rank == 0) {
      global_row_indices.resize(total_nnz);
      global_values.resize(total_nnz);

      // Копируем данные процесса 0
      std::copy(local_row_indices.begin(), local_row_indices.end(), global_row_indices.begin());
      std::copy(local_values.begin(), local_values.end(), global_values.begin());

      int offset = local_nnz;

      // Принимаем данные от остальных процессов
      for (int p = 1; p < size; ++p) {
        int p_nnz = 0;
        MPI_Recv(&p_nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> p_row_indices(p_nnz);
        MPI_Recv(p_row_indices.data(), p_nnz, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<double> p_values_real(p_nnz), p_values_imag(p_nnz);
        MPI_Recv(p_values_real.data(), p_nnz, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p_values_imag.data(), p_nnz, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::copy(p_row_indices.begin(), p_row_indices.end(), global_row_indices.begin() + offset);
        for (int i = 0; i < p_nnz; ++i) {
          global_values[offset + i] = Complex(p_values_real[i], p_values_imag[i]);
        }
        offset += p_nnz;
      }
    } else {
      // Отправляем данные на процесс 0
      MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(local_row_indices.data(), local_nnz, MPI_INT, 0, 1, MPI_COMM_WORLD);

      std::vector<double> local_values_real(local_nnz), local_values_imag(local_nnz);
      for (int i = 0; i < local_nnz; ++i) {
        local_values_real[i] = local_values[i].real();
        local_values_imag[i] = local_values[i].imag();
      }
      MPI_Send(local_values_real.data(), local_nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
      MPI_Send(local_values_imag.data(), local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    // Широковещательная рассылка результата всем процессам
    int bcast_rows = total_rows;
    int bcast_cols = total_cols;
    int bcast_nnz = total_nnz;

    MPI_Bcast(&bcast_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bcast_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bcast_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // col_ptrs
    int col_ptrs_size = static_cast<int>(global_col_ptrs.size());
    MPI_Bcast(&col_ptrs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      global_col_ptrs.resize(col_ptrs_size);
    }
    MPI_Bcast(global_col_ptrs.data(), col_ptrs_size, MPI_INT, 0, MPI_COMM_WORLD);

    // row_indices
    if (rank != 0) {
      global_row_indices.resize(bcast_nnz);
    }
    MPI_Bcast(global_row_indices.data(), bcast_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    // values (complex)
    std::vector<double> global_values_real(bcast_nnz), global_values_imag(bcast_nnz);
    if (rank == 0) {
      for (int i = 0; i < bcast_nnz; ++i) {
        global_values_real[i] = global_values[i].real();
        global_values_imag[i] = global_values[i].imag();
      }
    }
    MPI_Bcast(global_values_real.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_values_imag.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      global_values.resize(bcast_nnz);
      for (int i = 0; i < bcast_nnz; ++i) {
        global_values[i] = Complex(global_values_real[i], global_values_imag[i]);
      }
    }

    // Формируем итоговую матрицу
    CCSMatrix c;
    c.rows = bcast_rows;
    c.cols = bcast_cols;
    c.nnz = bcast_nnz;
    c.values = std::move(global_values);
    c.row_indices = std::move(global_row_indices);
    c.col_ptrs = std::move(global_col_ptrs);

    GetOutput() = c;
    return true;

  } catch (const std::exception &) {
    return false;
  }
}

bool BarkalovaMMultMatrixCcsALL::PostProcessingImpl() {
  const auto &c = GetOutput();

  // Базовая проверка
  if (c.rows <= 0 || c.cols <= 0) {
    return false;
  }

  // Проверка col_ptrs
  if (c.col_ptrs.size() != static_cast<size_t>(c.cols) + 1) {
    return false;
  }

  if (c.col_ptrs.empty()) {
    return false;
  }

  // Проверка монотонного возрастания col_ptrs
  for (size_t i = 1; i < c.col_ptrs.size(); ++i) {
    if (c.col_ptrs[i] < c.col_ptrs[i - 1]) {
      return false;
    }
  }

  // Проверка соответствия nnz
  if (c.nnz != static_cast<int>(c.values.size()) ||
      c.nnz != static_cast<int>(c.row_indices.size())) {
    return false;
  }

  // Проверка, что первый элемент col_ptrs = 0
  if (c.col_ptrs[0] != 0) {
    return false;
  }

  // Проверка, что последний элемент col_ptrs = nnz
  if (c.col_ptrs.back() != c.nnz) {
    return false;
  }

  return true;
}

}  // namespace barkalova_m_mult_matrix_ccs
*/

#include "barkalova_m_mult_matrix_ccs/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include "barkalova_m_mult_matrix_ccs/common/include/common.hpp"

namespace barkalova_m_mult_matrix_ccs {

BarkalovaMMultMatrixCcsALL::BarkalovaMMultMatrixCcsALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCSMatrix{};
}

bool BarkalovaMMultMatrixCcsALL::ValidationImpl() {
  const auto &[A, B] = GetInput();
  if (A.cols != B.rows) {
    return false;
  }
  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  if (A.col_ptrs.size() != static_cast<size_t>(A.cols) + 1) {
    return false;
  }
  if (B.col_ptrs.size() != static_cast<size_t>(B.cols) + 1) {
    return false;
  }
  if (A.col_ptrs[0] != 0 || B.col_ptrs[0] != 0) {
    return false;
  }
  if (static_cast<size_t>(A.nnz) != A.values.size()) {
    return false;
  }
  if (static_cast<size_t>(B.nnz) != B.values.size()) {
    return false;
  }
  return true;
}

bool BarkalovaMMultMatrixCcsALL::PreProcessingImpl() {
  return true;
}

namespace {
constexpr double kEpsilon = 1e-10;

void TransponirMatr(const CCSMatrix &a, CCSMatrix &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      Complex val = a.values[i];
      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

bool IsNonZero(const Complex &val) {
  return std::abs(val.real()) > kEpsilon || std::abs(val.imag()) > kEpsilon;
}

Complex ComputeScalarProduct(const CCSMatrix &at, const CCSMatrix &b, int row_a, int col_b) {
  Complex sum = Complex(0.0, 0.0);

  int ks = at.col_ptrs[row_a];
  int ls = b.col_ptrs[col_b];
  int kf = at.col_ptrs[row_a + 1];
  int lf = b.col_ptrs[col_b + 1];

  while ((ks < kf) && (ls < lf)) {
    if (at.row_indices[ks] < b.row_indices[ls]) {
      ks++;
    } else if (at.row_indices[ks] > b.row_indices[ls]) {
      ls++;
    } else {
      sum += at.values[ks] * b.values[ls];
      ks++;
      ls++;
    }
  }

  return sum;
}

// Выделяем вычисление локальных столбцов в отдельную функцию
void ComputeLocalColumns(int start_col, int local_cols, const CCSMatrix &at, const CCSMatrix &b,
                         std::vector<std::vector<int>> &col_rows, std::vector<std::vector<Complex>> &col_vals) {
#pragma omp parallel for schedule(static) default(none) shared(start_col, local_cols, at, b, col_rows, col_vals)
  for (int j_local = 0; j_local < local_cols; ++j_local) {
    int global_col = start_col + j_local;

    std::vector<int> rows;
    std::vector<Complex> vals;
    rows.reserve(100);
    vals.reserve(100);

    for (int i = 0; i < at.cols; i++) {
      Complex sum = ComputeScalarProduct(at, b, i, global_col);
      if (IsNonZero(sum)) {
        rows.push_back(i);
        vals.push_back(sum);
      }
    }

    col_rows[j_local] = std::move(rows);
    col_vals[j_local] = std::move(vals);
  }
}

// Выделяем сбор локальных данных в плоские векторы
void BuildLocalVectors(int local_cols, const std::vector<std::vector<int>> &col_rows,
                       const std::vector<std::vector<Complex>> &col_vals, std::vector<int> &local_row_indices,
                       std::vector<Complex> &local_values) {
  for (int j = 0; j < local_cols; ++j) {
    local_row_indices.insert(local_row_indices.end(), col_rows[j].begin(), col_rows[j].end());
    local_values.insert(local_values.end(), col_vals[j].begin(), col_vals[j].end());
  }
}

// Выделяем MPI сбор данных
void GatherResults(int rank, int size, int local_nnz, const std::vector<int> &local_row_indices,
                   const std::vector<Complex> &local_values, std::vector<int> &global_row_indices,
                   std::vector<double> &global_values_real, std::vector<double> &global_values_imag, int total_nnz) {
  std::vector<int> recv_counts(size, 0);
  MPI_Gather(&local_nnz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(size, 0);
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  // Сбор row_indices
  if (rank == 0) {
    global_row_indices.resize(total_nnz);
  }
  MPI_Gatherv(local_row_indices.data(), local_nnz, MPI_INT, global_row_indices.data(), recv_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  // Подготовка complex значений
  std::vector<double> local_values_real(local_nnz);
  std::vector<double> local_values_imag(local_nnz);
  for (int i = 0; i < local_nnz; ++i) {
    local_values_real[i] = local_values[i].real();
    local_values_imag[i] = local_values[i].imag();
  }

  // Сбор real частей
  if (rank == 0) {
    global_values_real.resize(total_nnz);
  }
  MPI_Gatherv(local_values_real.data(), local_nnz, MPI_DOUBLE, global_values_real.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Сбор imag частей
  if (rank == 0) {
    global_values_imag.resize(total_nnz);
  }
  MPI_Gatherv(local_values_imag.data(), local_nnz, MPI_DOUBLE, global_values_imag.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Выделяем широковещательную рассылку результата
void BroadcastResult(int rank, int total_rows, int total_cols, int total_nnz, std::vector<int> &global_col_ptrs,
                     std::vector<int> &global_row_indices, std::vector<double> &global_values_real,
                     std::vector<double> &global_values_imag) {
  int bcast_rows = total_rows;
  int bcast_cols = total_cols;
  int bcast_nnz = total_nnz;

  MPI_Bcast(&bcast_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // col_ptrs
  int col_ptrs_size = static_cast<int>(global_col_ptrs.size());
  MPI_Bcast(&col_ptrs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    global_col_ptrs.resize(col_ptrs_size);
  }
  MPI_Bcast(global_col_ptrs.data(), col_ptrs_size, MPI_INT, 0, MPI_COMM_WORLD);

  // row_indices
  if (rank != 0) {
    global_row_indices.resize(bcast_nnz);
  }
  MPI_Bcast(global_row_indices.data(), bcast_nnz, MPI_INT, 0, MPI_COMM_WORLD);

  // values_real и values_imag
  if (rank != 0) {
    global_values_real.resize(bcast_nnz);
    global_values_imag.resize(bcast_nnz);
  }
  MPI_Bcast(global_values_real.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_values_imag.data(), bcast_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

}  // namespace

bool BarkalovaMMultMatrixCcsALL::RunImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;

  try {
    // Получаем rank и size (инициализируем нулями)
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Транспонирование матрицы A
    CCSMatrix at;
    TransponirMatr(a, at);

    const int total_cols = b.cols;
    const int total_rows = a.rows;

    // Распределение столбцов между MPI процессами
    int cols_per_process = total_cols / size;
    int remainder = total_cols % size;

    // Добавляем скобки для порядка операций
    int start_col = (rank * cols_per_process) + std::min(rank, remainder);
    int local_cols = cols_per_process + (rank < remainder ? 1 : 0);

    // Локальные данные для каждого MPI процесса
    std::vector<std::vector<int>> col_rows(local_cols);
    std::vector<std::vector<Complex>> col_vals(local_cols);

    // Вычисление локальных столбцов
    ComputeLocalColumns(start_col, local_cols, at, b, col_rows, col_vals);

    // Сборка локальных данных в плоские векторы
    std::vector<int> local_row_indices;
    std::vector<Complex> local_values;
    BuildLocalVectors(local_cols, col_rows, col_vals, local_row_indices, local_values);
    int local_nnz = static_cast<int>(local_values.size());

    // Сбор количества ненулевых элементов для каждого столбца
    std::vector<int> global_col_nnz(total_cols, 0);
    for (int j = 0; j < local_cols; ++j) {
      global_col_nnz[start_col + j] = static_cast<int>(col_rows[j].size());
    }

    std::vector<int> recv_col_nnz(total_cols);
    MPI_Allreduce(global_col_nnz.data(), recv_col_nnz.data(), total_cols, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Формируем глобальные col_ptrs
    std::vector<int> global_col_ptrs = {0};
    for (int j = 0; j < total_cols; ++j) {
      global_col_ptrs.push_back(global_col_ptrs.back() + recv_col_nnz[j]);
    }
    int total_nnz = global_col_ptrs.back();

    // Сбор данных через MPI
    std::vector<int> global_row_indices;
    std::vector<double> global_values_real;
    std::vector<double> global_values_imag;
    GatherResults(rank, size, local_nnz, local_row_indices, local_values, global_row_indices, global_values_real,
                  global_values_imag, total_nnz);

    // Широковещательная рассылка результата
    BroadcastResult(rank, total_rows, total_cols, total_nnz, global_col_ptrs, global_row_indices, global_values_real,
                    global_values_imag);

    // Собираем complex значения на всех процессах
    std::vector<Complex> global_values(total_nnz);
    for (int i = 0; i < total_nnz; ++i) {
      global_values[i] = Complex(global_values_real[i], global_values_imag[i]);
    }

    // Формируем итоговую матрицу
    CCSMatrix c;
    c.rows = total_rows;
    c.cols = total_cols;
    c.nnz = total_nnz;
    c.values = std::move(global_values);
    c.row_indices = std::move(global_row_indices);
    c.col_ptrs = std::move(global_col_ptrs);

    GetOutput() = c;
    return true;

  } catch (const std::exception &) {
    return false;
  }
}

bool BarkalovaMMultMatrixCcsALL::PostProcessingImpl() {
  const auto &c = GetOutput();

  // Базовая проверка
  if (c.rows <= 0 || c.cols <= 0) {
    return false;
  }

  // Проверка col_ptrs
  if (c.col_ptrs.size() != static_cast<size_t>(c.cols) + 1) {
    return false;
  }

  if (c.col_ptrs.empty()) {
    return false;
  }

  // Проверка монотонного возрастания col_ptrs
  for (size_t i = 1; i < c.col_ptrs.size(); ++i) {
    if (c.col_ptrs[i] < c.col_ptrs[i - 1]) {
      return false;
    }
  }

  // Проверка соответствия nnz (используем std::cmp_not_equal)
  if (std::cmp_not_equal(c.nnz, c.values.size()) || std::cmp_not_equal(c.nnz, c.row_indices.size())) {
    return false;
  }

  // Проверка, что первый элемент col_ptrs = 0
  if (c.col_ptrs[0] != 0) {
    return false;
  }

  // Проверка, что последний элемент col_ptrs = nnz
  if (c.col_ptrs.back() != c.nnz) {
    return false;
  }

  return true;
}

}  // namespace barkalova_m_mult_matrix_ccs
