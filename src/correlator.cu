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


#define CHANNEL_SHARED_AT(array, position, bin, num_sensors, bin_size) array[position + (bin * num_sensors * bin_size)]
#define CHANNEL_GLOBAL_AT(array, position, bin, num_sensors, bin_size, num_bins, global_offset) array[position + (bin * num_sensors * bin_size) + (global_offset * num_sensors * bin_size * num_bins)]

__inline__ __device__ size_t
MultiTau::repeatTimes(size_t instant){
    size_t number_of_bits = sizeof(instant) * 8;

    size_t mask = 1;

    for (size_t i = 0; i < number_of_bits; ++i) {
        if ((instant & mask) != 0){
            return i + 1;
        }
        mask = mask << 1;
    }
    return number_of_bits;
};

template <typename T>
__global__ void 
MultiTau::correlate<T>(T * new_values, const size_t timepoints, size_t instants_processed, T * shift_register, T * accumulator, int * accumulator_positions, T * zero_delays, T * correlation, const size_t num_bins){

    size_t first_sensor_of_block = blockIdx.x * blockDim.y;
    size_t group_channel = threadIdx.x + threadIdx.y * blockDim.x;

    size_t num_sensors_per_block = blockDim.y;
    size_t bin_size = blockDim.x;
    size_t num_sensors = gridDim.x * num_sensors_per_block;
    size_t sensor_relative_position = threadIdx.y;
    size_t group_relative_position = threadIdx.x;

    // Way to handle templates and dynamic shared memory
    extern __shared__ __align__(sizeof(T)) unsigned char total_shared_memory[];

    T * block_shift_register = reinterpret_cast<T *>(&total_shared_memory);
    T * block_accumulator = &block_shift_register[num_sensors_per_block * num_bins * bin_size];
    int * block_accumulator_positions = reinterpret_cast<int *>(&block_accumulator[num_sensors_per_block * num_bins]);
    T * block_zero_delays = reinterpret_cast<T *>(&block_accumulator_positions[num_sensors_per_block * num_bins]);
    T * block_output = &block_zero_delays[num_sensors_per_block * num_bins];


    // Copy data from global to shared memory

    for (size_t i = 0; i < num_bins; ++i){
        block_shift_register[group_channel + (i * num_sensors_per_block * bin_size)] = shift_register[group_channel +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)];
    }

    for (size_t i = 0; i < num_bins; ++i){
        block_output[group_channel + (i * num_sensors_per_block * bin_size)] = correlation[group_channel +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)];
    }

    if (group_channel < num_bins * num_sensors_per_block) {
        block_accumulator[group_channel] = accumulator[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )];
        block_accumulator_positions[group_channel] = accumulator_positions[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )];
        block_zero_delays[group_channel] = zero_delays[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )];
    }

    __syncthreads();

    // Start the computation of the autocorrelation for the instants of time

    for (size_t i = 0 ; i < timepoints; ++i){

        instants_processed++;
        
        if (threadIdx.x < num_bins) {
            // Only ${num_bins} of threads per sensors insert the new datum in the correlator bins
            // Start old BinGroupsMultiSensorMemory::insertNew
            if (threadIdx.x == 0) {
                // Set the new datum to the accumulator
                block_accumulator[block_accumulator_positions[sensor_relative_position + group_relative_position * num_sensors_per_block]] = new_values[i * num_sensors + (sensor_relative_position + first_sensor_of_block )];
            }
            // Add the new value to the zero delay
            block_zero_delays[sensor_relative_position + group_relative_position * num_sensors_per_block] = new_values[i * num_sensors + (sensor_relative_position + first_sensor_of_block )];
            // End BinGroupsMultiSensorMemory::insertNew
        }
        __syncthreads();

        size_t repeat_times = MultiTau::repeatTimes(instants_processed);
        size_t last_group = repeat_times < num_bins ? repeat_times - 1 : num_bins - 1;
        
        for(size_t j = 0; j < repeat_times && j < num_bins; ++j){

            size_t group = last_group - j;
            // Add stuff
            block_output[sensor_relative_position * (num_bins + bin_size) + group*bin_size + group_relative_position] += block_zero_delays[sensor_relative_position + group * num_sensors_per_block] * block_shift_register[sensor_relative_position * bin_size + group * num_sensors_per_block * bin_size];
            __syncthreads();

            // Shift procedure
            if (group_channel < num_sensors_per_block) {
                //Update accumulator positions
                block_accumulator_positions[group_channel + group*num_sensors_per_block] = (block_accumulator_positions[group_channel + group*num_sensors_per_block] -1) & (bin_size -1);

                if (last_group - j < num_bins - 1){
                    block_accumulator[block_accumulator_positions[group_channel + (group +1) * num_sensors_per_block]] = block_accumulator[block_accumulator_positions[group_channel + group * num_sensors_per_block]];
                }

                block_accumulator[block_accumulator_positions[group_channel + group * num_sensors_per_block]] = 0;
                block_zero_delays[group_channel + group * num_sensors_per_block] = 0;

            }
            __syncthreads();

        }  

    }
    // Copy data from shared memory to global memory
    for (size_t i = 0; i < num_bins; ++i){
        shift_register[group_channel +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)] = block_shift_register[group_channel + (i * num_sensors_per_block * bin_size)];
    }

    for (size_t i = 0; i < num_bins; ++i){
        correlation[group_channel +(i * num_sensors_per_block * bin_size) + (blockIdx.x * num_sensors_per_block * bin_size * num_bins)] = block_output[group_channel + (i * num_sensors_per_block * bin_size)];
    }

    if (group_channel < num_bins * num_sensors_per_block) {
        accumulator[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )] = block_accumulator[group_channel];
        accumulator_positions[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )] = block_accumulator_positions[group_channel];
        zero_delays[group_channel +  (blockIdx.x * num_bins * num_sensors_per_block )] = block_zero_delays[group_channel];
    }
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

    shared_memory_per_block = ( 2 * (num_sensors_per_block * num_bins) + 2 *(num_sensors_per_block * num_bins * bin_size) ) * sizeof(T) + (num_sensors_per_block * num_bins) * sizeof(int);

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
    if (d_accumulator != nullptr){
        CHECK(cudaFree(d_accumulator));
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

    // In the previous version of the code the accumulator was the first position of every group in the shift_register_array
    // here they are divided, the position in memory thus remain the same
    CHECK(cudaMalloc(&d_shift_register, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_accumulator, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_accumulator_positions, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMalloc(&d_zero_delays, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_correlation, num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "Initializing device arrays" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator, 0, num_bins * num_sensors * sizeof(T)));
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

    MultiTau::correlate<T><<<number_of_blocks, threads_per_block, shared_memory_per_block>>>(d_new_values, timepoints, instants_processed, d_shift_register, d_accumulator, d_accumulator_positions, d_zero_delays, d_correlation, num_bins);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
    
    if (d_new_values != nullptr){
        cudaFree(d_new_values);
    }

    if (debug) std::cout << "Instant Processed: " << instants_processed << std::endl;

    instants_processed += timepoints;
};

template <typename T>
T Correlator<T>::get(size_t sensor, size_t lag){
    return correlation[lag * num_sensors + sensor];
};

template <typename T>
void Correlator<T>::reset(){

    if (debug) std::cout << "Resetting all gpu arrays to zero" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_accumulator_positions, 0, num_bins * num_sensors * sizeof(int)));
    CHECK(cudaMemset(d_zero_delays, 0, num_bins * num_sensors * sizeof(T)));
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