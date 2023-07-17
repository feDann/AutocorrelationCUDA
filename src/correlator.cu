#include "correlator.cuh"
#include <cassert>
#include <cmath>


#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ <<": ERROR: " << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ <<": ERROR: " << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }


template <typename T>
__global__ void MultiTau::correlate<T>(T * new_values, size_t packet_size, T * shift_register, T * accumulator, T * insert_indexes, T * correlation){

};

template <typename T>
Correlator<T>::Correlator(size_t _num_bins, size_t _bin_size, size_t _num_sensors,size_t _num_sensors_per_block, int device, bool _verbose){    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device);

    //TODO assert that the max shared memory is not exceeded
    
    num_bins = _num_bins;
    bin_size = _bin_size;
    num_sensors = _num_sensors;
    num_sensors_per_block = _num_sensors_per_block;
    verbose = _verbose;

    max_tau = bin_size * std::pow(2, num_bins);
    num_taus = bin_size * num_bins;

    if (verbose){
        std::cout << "Number of bins: " << num_bins << std::endl;
        std::cout << "Size of bins: " << bin_size << std::endl;
        std::cout << "Number of sensors: " << num_sensors << std::endl;
        std::cout << "Number of sensors per block: " << num_sensors_per_block << std::endl;
        std::cout << "Max tau possible: " << max_tau << std::endl;
        std::cout << "Number of taus possible: " << num_taus << std::endl;
        //TODO add shared memory occcupancy info
    }    
};

template <typename T>
Correlator<T>::~Correlator(){
    free(correlation);
    free(taus);

    cudaDeviceReset();
};

template <typename T>
void Correlator<T>::alloc(){
    if (verbose) std::cout << "Allocating gpu arrays into global memory" << std::endl;

    CHECK(cudaMalloc(&d_shift_register, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_accumulator, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_insert_indexes, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMalloc(&d_correlation, num_taus * sizeof(T)));

    if (verbose) std::cout << "Initializing device arrays" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_insert_indexes, 0, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMemset(d_correlation, 0 , num_taus * sizeof(T)));

};

template <typename T>
void Correlator<T>::correlate(T * new_values, size_t packet_size){

};

template <typename T>
void Correlator<T>::outputs(T * correlations, uint32_t* t){

};

template <typename T>
void Correlator<T>::reset(){

    if (verbose) std::cout:: << "Resetting all gpu arrays to zero" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_insert_indexes, 0, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMemset(d_correlation, 0 , num_taus * sizeof(T)));

};

template <typename T>
T Correlator<T>::get(size_t sensor, size_t lag){

};