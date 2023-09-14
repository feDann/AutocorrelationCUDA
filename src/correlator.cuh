#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define M 2 // Number of lag coalesced
#define MIN_SENSORS_PER_BOOCK 4
namespace MultiTau {


    __inline__ __device__ size_t
    insert_until_bin(const size_t instants, const size_t num_bins);

    template <typename T>
    __global__ void 
    correlate ( T * new_values, 
                const size_t timepoints, 
                size_t instants_processed, 
                T * shift_register, 
                int * shift_positions, 
                T * accumulators, 
                T * correlation,  
                const size_t num_bins,
                const size_t num_sensors
                );

}

template<typename T>
class Correlator {

    public:
        Correlator(const size_t num_bins, const size_t bin_size, const size_t num_sensors, const size_t packet_size, const int device = 0, const bool debug = false);
        ~Correlator();

        void alloc();
        void correlate(const T * new_values, const size_t timepoints);
        void transfer();
        T get(const size_t sensor, const size_t lag);
        void reset();

    private:
        dim3 number_of_blocks;
        dim3 threads_per_block;

        size_t num_bins;
        size_t bin_size;
        size_t num_sensors;
        size_t num_sensors_per_block;
        size_t packet_size;

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

        T * d_correlation = nullptr;
        T * d_new_values = nullptr;

};