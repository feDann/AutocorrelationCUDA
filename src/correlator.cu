#include "correlator.cuh"
#include <cassert>
#include <cmath>

#ifdef _DEBUG_BUILD
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

#else
#define CHECK(call)                                                                      
#define CHECK_KERNELCALL()                                                         
#endif // _DEBUG_BUILD


template <typename T>
__global__ void 
MultiTau::correlate<T>(T * new_values, const size_t timepoints, size_t instants_processed, T * shift_register, int * shift_positions, T * accumulators, int * num_accumulators, T * correlation, const size_t num_bins){

    size_t first_sensor_of_block = blockIdx.x * blockDim.y;
    size_t group_channel = threadIdx.x + threadIdx.y * blockDim.x;

    size_t num_sensors_per_block = blockDim.y;
    size_t bin_size = blockDim.x;
    size_t num_sensors = gridDim.x * num_sensors_per_block;
    size_t sensor_relative_position = threadIdx.y;
    size_t group_relative_position = threadIdx.x;

    // Way to handle templates and dynamic shared memory
    extern __shared__  unsigned char total_shared_memory[];

    // int * block_accumulator_positions = reinterpret_cast<int *>(&total_shared_memory);
    // T * block_shift_register = reinterpret_cast<T *>(&block_accumulator_positions[num_sensors_per_block * num_bins]);
    // T * block_zero_delays = &block_shift_register[num_sensors_per_block * num_bins * bin_size];
    // T * block_output = &block_zero_delays[num_sensors_per_block * num_bins];


};


template <typename T>
Correlator<T>::Correlator(size_t t_num_bins, size_t t_bin_size, size_t t_num_sensors,size_t t_num_sensors_per_block, int t_device, bool t_debug){    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, t_device);
    
    num_bins = t_num_bins;
    bin_size = t_bin_size;
    num_sensors = t_num_sensors;
    num_sensors_per_block = t_num_sensors_per_block;
    debug = t_debug;

    max_tau = bin_size * std::pow(2, num_bins);
    num_taus = bin_size * num_bins;

    //                          accumulators                            shift registers and outputs                                     accumulator and num accumulator 
    shared_memory_per_block = ((num_sensors_per_block * num_bins) + 2 * (num_sensors_per_block * num_bins * bin_size) ) * sizeof(T) + 2 * (num_sensors_per_block * num_bins) * sizeof(int);

    assert(shared_memory_per_block <= device_properties.sharedMemPerBlock && "ERROR: current configuration exceed device shared memory limits");

    number_of_blocks = dim3(num_sensors / num_sensors_per_block, 1 , 1);
    threads_per_block = dim3(bin_size, num_sensors_per_block, 1);

    if (debug){
        std::cout << "Number of bins: " << num_bins << std::endl;
        std::cout << "Size of bins: " << bin_size << std::endl;
        std::cout << "Number of sensors: " << num_sensors << std::endl;
        std::cout << "Number of sensors per block: " << num_sensors_per_block << std::endl;
        std::cout << "Max tau possible: " << max_tau << std::endl;
        std::cout << "Number of taus possible: " << num_taus << std::endl;
        std::cout << "Shared Memory per block: " << shared_memory_per_block << " B" << std::endl;
        std::cout << "Number of blocks: (" << number_of_blocks.x << "," << number_of_blocks.y << "," << number_of_blocks.z << ")" << std::endl;
        std::cout << "Threads per blocks: (" << threads_per_block.x << "," << threads_per_block.y << "," << threads_per_block.z << ")" << std::endl;
    }    
};

template <typename T>
Correlator<T>::~Correlator(){
    // Free host memory
    if (correlation != nullptr){
        free(correlation);
    }
    if (taus != nullptr){
        free(taus);
    }

    // Free device memory
    if (d_shift_register != nullptr){
        CHECK(cudaFree(d_shift_register));
    }

    if (d_shift_positions != nullptr){
        CHECK(cudaFree(d_shift_positions));
    }

    if (d_accumulators != nullptr){
        CHECK(cudaFree(d_accumulators));
    }

    if (d_num_accumulators != nullptr){
        CHECK(cudaFree(d_num_accumulators));
    }

    if (d_correlation != nullptr){
        CHECK(cudaFree(d_correlation));
    }

    cudaDeviceReset();
};

template <typename T>
void Correlator<T>::alloc(){
    if (debug) std::cout << "Allocating device arrays into global memory" << std::endl;

    CHECK(cudaMalloc(&d_shift_register, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_shift_positions, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMalloc(&d_accumulators, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_num_accumulators, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMalloc(&d_correlation, num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "Initializing device arrays" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_shift_positions, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_accumulators, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_num_accumulators, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "Alocating device arrays" << std::endl;

    correlation = (T*)malloc(num_taus * num_sensors * sizeof(T));

};

template <typename T>
void Correlator<T>::correlate(T * new_values, size_t timepoints){

    if (debug) std::cout << "Allocating and copying new values to gpu array" << std::endl;

    CHECK(cudaMalloc(&d_new_values, timepoints * num_sensors * sizeof(T)));
    CHECK(cudaMemcpy(d_new_values, new_values, timepoints * num_sensors * sizeof(T), cudaMemcpyHostToDevice));

    if (debug) std::cout << "Starting correlation" << std::endl;

    MultiTau::correlate<T><<<number_of_blocks, threads_per_block, shared_memory_per_block>>>(d_new_values, timepoints, instants_processed, d_shift_register, d_shift_positions, d_accumulators, d_num_accumulators, d_correlation, num_bins);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
    
    if (d_new_values != nullptr){
        cudaFree(d_new_values);
    }

    if (debug) std::cout << "Instant Processed: " << instants_processed << std::endl;

    instants_processed += timepoints;
};

template <typename T>
void Correlator<T>::transfer(){    

    if (debug) std::cout << "Transfering data from device memory to host memory" << std::endl;

    CHECK(cudaMemcpy(correlation, d_correlation, num_taus * num_sensors * sizeof(T), cudaMemcpyDeviceToHost));
    transfered = true;

    if (debug) std::cout << "Data transfered" << std::endl;
}


template <typename T>
T Correlator<T>::get(size_t sensor, size_t lag){
    assert(transfered && "ERROR: Data not transfered from device memory to host memory");
    return correlation[lag * num_sensors + sensor];
};

template <typename T>
void Correlator<T>::reset(){

    if (debug) std::cout << "Resetting all device arrays to zero" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_shift_positions, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_accumulators, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_num_accumulators, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "Resetting all host arrays to zero" << std::endl;

    memset(correlation, 0, num_taus * num_sensors * sizeof(T));

    instants_processed = 0;
    transfered = false;
};

// Needed for the template
template class Correlator<int8_t>;
template class Correlator<int16_t>;
template class Correlator<int32_t>;
template class Correlator<int64_t>;

template class Correlator<uint8_t>;
template class Correlator<uint16_t>;
template class Correlator<uint32_t>;
template class Correlator<uint64_t>;

template class Correlator<double>;
template class Correlator<float>;