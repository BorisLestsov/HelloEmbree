#pragma once

#include <random>
#include <climits>

#include "vector3.h"

/****************************************************** Constants *****************************************************/
const double D_PI                           = 3.14159265358979323846;
const double D_INV_PI                       = 0.31830988618379067154;
const double D_INV2_PI                      = 0.15915494309189533577;
const double D_INV4_PI                      = 0.07957747154594766788;
const double D_EPSILON                      = 0.0000000001;

const double D_OFFSET_CONSTANT              = 0.0001;

const double DEFAULT_REFRACTIVE_INDEX_OUT   = 1.0;
const double DEFAULT_REFRACTIVE_INDEX_IN    = 1.5;

const int    MAX_NUMBER_OF_EVENTS           = 20;
const int    MAX_NUMBER_OF_RNDS             = 2 * MAX_NUMBER_OF_EVENTS;

/*********************************************** Random Numbers Generator *********************************************/
std::default_random_engine generator = std::default_random_engine();
std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);

inline double get_uniform1(){
    return distribution(generator);
}


static unsigned long x=123456789, y=362436069, z=521288629;

double get_uniform(void) {          //period 2^96-1
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return ((double) z / (double) ULONG_MAX);

}

/************************************************ Image I/O and Clamping **********************************************/
inline  double clamp(double x, double low = 0.0, double high = 1.0)  {
    return (x < high) ? ((x > low) ? x : low) : high;
}

inline uint8_t to_byte(double x, double gamma = 2.2)  {
    return static_cast<uint8_t>(clamp(255.0 * std::pow(x, 1 / gamma), 0.0, 255.0));
}

inline void save_ppm(int w, int h, const Vec3 *pixels, int samps)  {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y_%I-%M-%S_", timeinfo);

    std::string sppstring = std::to_string(samps);
    std::string offsetstring = std::to_string(D_OFFSET_CONSTANT);
    std::string timestring(buffer);
    std::string filename = timestring + offsetstring + "_" + sppstring + "spp.ppm";

    FILE *fp = fopen(filename.c_str(), "w");

    fprintf(fp, "P3\n%u %u\n%u\n", w, h, 255u);
    for (int i = 0; i < w * h; ++i) {
        fprintf(fp, "%u %u %u ", to_byte(pixels[i].x), to_byte(pixels[i].y), to_byte(pixels[i].z));
    }

    fclose(fp);
}

#define _SEP " "

template<typename T>
inline void print(const T& arg){
    std::cout << arg << _SEP
              << std::endl;
}
template<typename T1, typename T2>
inline void print(const T1& arg1, const T2& arg2){
    std::cout << arg1 << _SEP
              << arg2 << _SEP
              << std::endl;
}
template<typename T1, typename T2, typename T3>
inline void print(const T1& arg1, const T2& arg2, const T3& arg3){
    std::cout << arg1 << _SEP
              << arg2 << _SEP
              << arg3 << _SEP
              << std::endl;
}
template<typename T1, typename T2, typename T3, typename T4>
inline void print(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4){
    std::cout << arg1 << _SEP
              << arg2 << _SEP
              << arg3 << _SEP
              << arg4 << _SEP
              << std::endl;
}
