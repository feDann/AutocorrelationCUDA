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
__global__ void 
MultiTau::correlate<T>(T * new_values, const size_t timepoints, size_t instants_processed, T * shift_register, int * accumulator_positions, T * zero_delays, T * correlation, const size_t num_sensors_per_block, const size_t num_bins, const size_t bin_size){

    size_t first_sensor_of_block = blockIdx.x * blockDim.y;
    size_t group_position = threadIdx.x + threadIdx.y * blockDim.x;

    // Way to handle templates and dynamic shared memory
    extern __shared__ __align__(sizeof(T)) unsigned char total_shared_memory[];

    T * block_shift_register = reinterpret_cast<T *>(total_shared_memory);
    T * block_accumulator_positions = &block_shift_register[num_sensors_per_block * num_bins * bin_size];
    T * block_zero_delays = &block_accumulator_positions[num_sensors_per_block * num_bins];
    T * block_output = & block_zero_delays[num_sensors_per_block * num_bins];


    // Copy data from global to shared memory

    for (size_t i = 0; i < num_bins; ++i){
        block_shift_register[group_position + (i * num_sensors_per_block * bin_size)] = shift_register[group_position +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)];
    }

    for (size_t i = 0; i < num_bins; ++i){
        block_output[group_position + (i * num_sensors_per_block * bin_size)] = correlation[group_position +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)];
    }

    if (group_position < num_bins * num_sensors_per_block) {
        block_accumulator_positions[group_position] = accumulator_positions[group_position +  (blockIdx.x * num_bins * num_sensors_per_block )];
        block_zero_delays[group_position] = zero_delays[group_position +  (blockIdx.x * num_bins * num_sensors_per_block )];
    }

    __syncthreads();

    // Start the computation of the autocorrelation for the instants of time

    for (size_t i = 0 ; i < timepoints; ++i){

        instants_processed++;
        printf("Istants processed: %d ThreadIdx.x: %d\n", instants_processed, threadIdx.x);

        if (threadIdx.x == 0) {
            // Only one thread per sensor insert new datum
            // TODO Add new datum to shift register
        }
        __syncthreads();

    }
};


template <typename T>
Correlator<T>::Correlator(size_t _num_bins, size_t _bin_size, size_t _num_sensors,size_t _num_sensors_per_block, int device, bool _debug){    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device);
    
    num_bins = _num_bins;
    bin_size = _bin_size;
    num_sensors = _num_sensors;
    num_sensors_per_block = _num_sensors_per_block;
    debug = _debug;

    max_tau = bin_size * std::pow(2, num_bins);
    num_taus = bin_size * num_bins;

    shared_memory_per_block = ( 2 * (num_sensors_per_block * num_bins) + 2 *(num_sensors_per_block * num_bins * bin_size) ) * sizeof(T);

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
        std::cout << "Shared Memory per block: " << shared_memory_per_block << std::endl;
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
    if (d_accumulator_positions != nullptr){
        CHECK(cudaFree(d_accumulator_positions));
    }
    if (d_correlation != nullptr){
        CHECK(cudaFree(d_correlation));
    }
    if (d_zero_delays != nullptr){
        CHECK(cudaFree(d_zero_delays));
    }


    cudaDeviceReset();
};

template <typename T>
void Correlator<T>::alloc(){
    if (debug) std::cout << "Allocating gpu arrays into global memory" << std::endl;

    CHECK(cudaMalloc(&d_shift_register, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_accumulator_positions, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMalloc(&d_zero_delays, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_correlation, num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "Initializing device arrays" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator_positions, 0, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMemset(d_zero_delays, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

};

template <typename T>
void Correlator<T>::correlate(T * new_values, size_t timepoints){

    if (debug) std::cout << "Allocating and copying new values to gpu array" << std::endl;

    CHECK(cudaMalloc(&d_new_values, timepoints * num_sensors * sizeof(T)));
    CHECK(cudaMemcpy(d_new_values, new_values, timepoints * num_sensors * sizeof(T), cudaMemcpyHostToDevice));

    if (debug) std::cout << "Starting correlation" << std::endl;

    MultiTau::correlate<T><<<number_of_blocks, threads_per_block, shared_memory_per_block>>>(d_new_values, timepoints, instants_processed, d_shift_register, d_accumulator_positions, d_zero_delays, d_correlation, num_sensors_per_block, num_bins, bin_size);
    cudaDeviceSynchronize();
    
    if (d_new_values != nullptr){
        cudaFree(d_new_values);
    }

    if (debug) std::cout << "Instant Processed: " << instants_processed << std::endl;

    instants_processed += timepoints;
};

template <typename T>
T Correlator<T>::get(size_t sensor, size_t lag){
    assert(0 && "Not implemented");
};

template <typename T>
void Correlator<T>::reset(){

    if (debug) std::cout << "Resetting all gpu arrays to zero" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator_positions, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_zero_delays, 0, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

    instants_processed = 0;
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