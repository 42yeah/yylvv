#include <iostream>
#include <fstream>
#include <iomanip>


// Drop first element of data
int main()
{
    std::ifstream reader("../rectgrid2.csv");
    std::ofstream writer("rectgrid2.txt");
    float a, x, y, z;
    while (true)
    {
        if (!(reader >> a >> x >> y >> z))
        {
            break;
        }
        writer << std::setprecision(5);
        writer << (x * 10.0f) << " " << (y * 10.0f) << " " << (z * 10.0f) << std::endl;
    }
    return 0;
}
