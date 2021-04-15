﻿#ifdef _WIN32
#include <WinSock2.h>
#include <Windows.h>
#include <stdint.h>

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

#else
#include <sys/time.h>
#endif


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ResultArray.h"
#include "DataFile.h"
#include "CudaInput.h"
#include "InputVector.h"
#include "BinGroupsMultiSensorMemory.h"
#include "SensorsDataPacket.h"
#include <iostream>
#include <vector>
#include <memory>

#define SENSORS_EXP2_DEFAULT 8
#define GROUPS_DEFAULT 10
#define INSTANTS_PER_PACKET_DEFAULT 80000
#define REPETITIONS_DEFAULT 4
#define GROUP_SIZE_EXP2 5


template <typename Contained, int SizeExp2>
__global__ void autocorrelate(AutocorrelationCUDA::SensorsDataPacket<Contained> packet, AutocorrelationCUDA::BinGroupsMultiSensorMemory<Contained, SizeExp2> binStructure, std::uint_fast32_t instantsProcessed, AutocorrelationCUDA::ResultArray<Contained> out);


namespace AutocorrelationCUDA {

	template<typename Integer>
	__device__ static Integer repeatTimes(Integer x, std::int8_t bits) {
		Integer pow2 = 1;
		for (std::uint8_t i = 0; i < bits; ++i) {
			if ((x & pow2) != 0) {
				return i + 1;
			}
			pow2 = pow2 << 1;
		}
		return 0;
	}

}


std::vector<std::uint_fast32_t> askParameters() {
	std::vector<std::uint_fast32_t> result;
	std::uint_fast32_t tmp;

	std::cout << "\nInsert 0 for default parameter\n";
	std::cout << "\nnumber of sensors --> insert a number x, you'll have 2^x sensors: ";std::cin >> tmp;	result.emplace_back(tmp <= 0 ? SENSORS_EXP2_DEFAULT : tmp);
	std::cout << "\nnumber of bin groups: ";											std::cin >> tmp;	result.emplace_back(tmp <= 0 ? GROUPS_DEFAULT : tmp);
	std::cout << "\ninstants in a sigle packet: ";										std::cin >> tmp;	result.emplace_back(tmp <= 0 ? INSTANTS_PER_PACKET_DEFAULT : tmp);
	std::cout << "\nrepetitions: ";														std::cin >> tmp;	result.emplace_back(tmp <= 0 ? REPETITIONS_DEFAULT : tmp);

	return result;
}



constexpr int pow(int base, int exp) {
	if (exp == 0) {
		return 1;
	}

	int res = base;
	for (int i = 1; i < exp; ++i) {
		res *= base;
	}
	return res;
}



int main() {
	
	//ask parameters to user
	std::vector<std::uint_fast16_t> params = askParameters();
	const std::uint_fast32_t sensorsExp2 = params[0];
	const std::uint_fast32_t groups = params[1];
	const std::uint_fast32_t instantsPerPacket = params[2];
	const std::uint_fast32_t repetitions = params[3];

	std::uint_fast32_t sensors = std::pow(2, sensorsExp2);
	dim3 numberOfBlocks = sensors; //number of blocks active on the GPU
	constexpr int threadsPerBlock = pow(2, GROUP_SIZE_EXP2);

	//open file where data is stored
	std::unique_ptr<AutocorrelationCUDA::CudaInput<int>> dataFile = std::make_unique<AutocorrelationCUDA::InputVector<int>>("", "test1");
	

	//create array where to put new data for the GPU
	AutocorrelationCUDA::SensorsDataPacket<int> inputArray(sensorsExp2, instantsPerPacket);

	//array in GPU of the bin groups structure
	AutocorrelationCUDA::BinGroupsMultiSensorMemory<int, GROUP_SIZE_EXP2> binStructure(sensors, groups);

	//output array to store results in GPU
	auto out = binStructure.generateResultArray();



	
	//timer
	AutocorrelationCUDA::Timer timer{[](std::vector<double> data){AutocorrelationCUDA::DataFile<double>::write(data, "out_timer.txt");},
									 [](){struct timeval tp;
									      gettimeofday(&tp, NULL);
									      return ((double)tp.tv_sec + (double)tp.tv_usec * 0.000001);}};
	
	std::uint_fast32_t timesCalled; //counter
	timer.start();
	for(timesCalled = 0; timesCalled < repetitions; ++timesCalled) {
		inputArray.setNewDataPacket(dataFile->read(sensors * instantsPerPacket)); //store in GPU memory a new block of data to be processed
		//cudaDeviceSynchronize();
		//timer.getInterval();
		//timer.start();
		autocorrelate <<< numberOfBlocks, threadsPerBlock-1 >>> (inputArray, binStructure, timesCalled * instantsPerPacket, out);
		//cudaDeviceSynchronize();	
		timer.getInterval();
	}
	

	/*std::cout << timesCalled << "\n";
	for (int sensor = 0; sensor < out.getSensors(); ++sensor) {
		std::cout << "\n\n\t======= SENSOR " << sensor << " =======\n";

		for (int lag = 0; lag < out.getMaxLagv(); ++lag) {
			int curr = out.get(sensor, lag);
			int div = (timesCalled*instantsPerPacket) - lag;
			float print = (float) curr / div;
			std::cout << "\n\t" << lag+1 << " --> " << print;
		}
	}*/

	//write output to file
	//AutocorrelationCUDA::DataFile<std::uint_fast32_t>::write(out);

}


template <typename Contained, int SizeExp2>
__global__ void autocorrelate(AutocorrelationCUDA::SensorsDataPacket<Contained> packet, AutocorrelationCUDA::BinGroupsMultiSensorMemory<Contained, SizeExp2> binStructure, std::uint_fast32_t instantsProcessed, AutocorrelationCUDA::ResultArray<Contained> out) {
	std::uint_fast32_t absoluteThreadsIdx = blockIdx.x * blockDim.x + threadIdx.x;
	Contained instantsNum = packet.instantsNum();

	//cycle over all of the new data, where i is the instant in time processed
	for (int i = 0; i < instantsNum; ++i) {
		instantsProcessed++;

		//only one thread per sensor adds the new datum to the bin structure
		if (absoluteThreadsIdx < binStructure.sensorsNum()) {
			binStructure.insertNew(absoluteThreadsIdx, packet.get(absoluteThreadsIdx, i));
		}

		//calculate autocorrelation for that instant
		//Decides how many group to calculate, based on how many instants have been already processed (i.e. 1 instant-->0; 2-->0,1; 3-->0; 4-->0,1,2; 5-->0; 6-->0,1; ...)
		std::uint_fast32_t repeatTimes = AutocorrelationCUDA::repeatTimes(instantsProcessed, 32);
		for (std::uint_fast8_t j = 0; j < repeatTimes; ++j) {
			out.addTo(blockIdx.x, threadIdx.x + j * blockDim.x, binStructure.getZeroDelay(blockIdx.x, j) * binStructure.get(blockIdx.x, j, threadIdx.x)); //given that each CUDA block has k threads, where k = groupSize()-1
			binStructure.shift(blockIdx.x, j);
		}
	}

}

