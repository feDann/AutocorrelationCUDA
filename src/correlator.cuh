#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define M 2 // Number of lag coalesced

namespace MultiTau {

    template <typename T>
    __global__ void 
    correlate (T * new_values, const size_t timepoints, size_t instants_processed, T * shift_register, int * shift_positions, T * accumulators, int * num_accumulators, T * correlation,  const size_t num_bins);

}

template<typename T>
class Correlator {

    public:
        Correlator(size_t num_bins, size_t bin_size, size_t num_sensors, size_t num_sensors_per_block = 8, int device = 0, bool debug = false);
        ~Correlator();

        void alloc();
        void correlate(T * new_values, size_t timepoints);
        void transfer();
        T get(size_t sensor, size_t lag);
        void reset();

    private:
        dim3 number_of_blocks;
        dim3 threads_per_block;

        size_t num_bins;
        size_t bin_size;
        size_t num_sensors;
        size_t num_sensors_per_block;

        size_t shared_memory_per_block;

        uint32_t max_tau;
        uint32_t num_taus;

        size_t instants_processed = 0;

        bool debug;
        bool transfered;

        T * correlation = nullptr;
        size_t * taus = nullptr;

        // Device variables
        T * d_shift_register = nullptr;
        int * d_shift_positions = nullptr;

        T * d_accumulators = nullptr;
        int * d_num_accumulators = nullptr;

        T * d_correlation = nullptr;
        T * d_new_values = nullptr;

};