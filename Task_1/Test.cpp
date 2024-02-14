#include <iostream>
#include <cmath>

#define arr_elem 10000000

#define d_double

#ifdef d_double
using nspace = double;
#else
using nspace = float;
#endif // d_double


const nspace pi = std::acos(-1);

int main(int argc, char const* argv[])
{
    nspace sum = 0;
    nspace* array;
    array = new nspace[arr_elem];

    for (int i = 0; i < arr_elem; ++i)
    {
        array[i] = sin(2*pi*i/arr_elem);
    }

    for (int i = 0; i < arr_elem; ++i)
    {
        sum += array[i];
    }

    std::cout << sum << std::endl;
    
    delete(array);
    
    return 0;
}