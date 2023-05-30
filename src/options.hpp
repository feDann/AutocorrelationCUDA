#pragma once

#include <getopt.h>
#include <string>


#define DEBUG false
#define ITERATIONS 1
#define PARSE_FILE true
#define INPUT_FILE "../input/100.csv"
#define OUTPUT_FILE "out.csv"


struct Options {
    uint iterations = ITERATIONS;
    int debug = DEBUG;
    std::string input_file = INPUT_FILE;
    std::string output_file = OUTPUT_FILE;

    Options(int argc, char* argv[]){
        int opt;
        int option_index = 0;
        static struct option long_options[] = { {"debug", no_argument, 0, 'd'},
                                                {"input_file", required_argument, 0, 'i'},
                                                {"output_file", optional_argument, 0, 'o'},
                                                {"iterations", optional_argument, 0, 'I'},
                                                {0,0,0,0}};

        while(( opt = getopt_long(argc, argv, "di:o:I:", long_options, &option_index) ) != EOF ) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'i':
                    input_file = optarg;
                    break;
                case 'o':
                    output_file = optarg;
                    break;
                case 'I':
                    iterations = atoi(optarg);
                    break;
                default:
                    break;
            }
        }
    }
};

