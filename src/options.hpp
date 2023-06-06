#pragma once

#include <getopt.h>
#include <string>


#define DEBUG false
#define ITERATIONS 1
#define PARSE_FILE false
#define OUTPUT_FILE "out"


struct Options {
    uint iterations = ITERATIONS;
    int debug = DEBUG;
    int parse_file = PARSE_FILE;
    std::string input_file;
    std::string output_file = OUTPUT_FILE;
    int packets=-1;

    Options(int argc, char* argv[]){
        int opt;
        int option_index = 0;
        static struct option long_options[] = { {"debug", no_argument, 0, 'd'},
                                                {"packets", required_argument, 0, 'p'},
                                                {"input_file", required_argument, 0, 'i'},
                                                {"output_file", optional_argument, 0, 'o'},
                                                {"iterations", optional_argument, 0, 'I'},
                                                {"help", no_argument, 0, 'h'},
                                                {0,0,0,0}};

        while(( opt = getopt_long(argc, argv, "dpi:o:I:h", long_options, &option_index) ) != EOF ) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'p':
                    packets = atoi(optarg);
                    break;
                case 'i':
                    input_file = optarg;
                    parse_file = true;
                    break;
                case 'o':
                    output_file = optarg;
                    break;
                case 'I':
                    iterations = atoi(optarg);
                    break;
                case 'h':
                    std::cout << "This program is designed to perform autocorrelation calculations on sensor data using CUDA, a parallel computing platform."<< std::endl <<std::endl;
                    std::cout << "\t--debug, -d\t\t Activate debug prints" << std::endl;
                    std::cout << "\t--packets, -p\t\t Number of instant used per packets" << std::endl;
                    std::cout << "\t--input_file, -i\t\t Name of the input file containing the sensor data. If not given random data will e used for the calculation of the autocorrelation" << std::endl;
                    std::cout << "\t--output-file, -o \t\t Name of the output file. Correlation result will be saved into a csv file. If not given a default name will be used \"out.csv\"" << std::endl;
                    std::cout << "\t--iterations, -I\t\t Number of times that the calculation is repeated. If it is greater than one the calculation of the autocorrelation will be repeated multiple times on the same data" << std::endl;
                    std::cout << "\t--help, -h\t\t Print this message" << std::endl;
                    exit(0);
                    break;
                default:
                    break;
            }
        }

        if( packets == -1){
            std::cout << "ERROR: --packets is a required argument. Use --help for more details" << std::endl;
            exit(1);
        }


    }
};

