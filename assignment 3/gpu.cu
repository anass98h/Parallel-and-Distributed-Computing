#include <iostream>

int main() {
    daDeviceProp devProps;
    udaGetDeviceProperties(&devProps, 0);
    std::cout << devProps.totalGlobalMem << "\n" << devProps.regsPerBlock << "\n";
}