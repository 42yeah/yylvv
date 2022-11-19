#include <iostream>
#include <fstream>
#include <iomanip>


int main()
{
    // Drop first element of data
    // std::ifstream reader("../rectgrid2.csv");
    // std::ofstream writer("rectgrid2.txt");
    // float a, x, y, z;
    // while (true)
    // {
    //     if (!(reader >> a >> x >> y >> z))
    //     {
    //         break;
    //     }
    //     writer << std::setprecision(5);
    //     writer << (x * 10.0f) << " " << (y * 10.0f) << " " << (z * 10.0f) << std::endl;
    // }

    std::ifstream reader("../tierny2.csv");
    std::ofstream writer("tierny2.txt");
    float a, x, y, z;
    while (true)
    {
        if (!(reader >> x >> y >> z >> a >> a))
        {
            break;
        }
        writer << std::setprecision(5);
        writer << x << " " << y << " " << z << std::endl;
    }
    return 0;
}
