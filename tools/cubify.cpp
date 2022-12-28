#include <iostream>
#include <fstream>
#include <vector>
#include <glm/glm.hpp>


int main()
{
    std::ifstream reader("separator_hires.csv");
    int w, h, d;

    reader >> w >> h >> d;

    std::vector<std::vector<std::vector<glm::vec3> > > cube;
    cube.resize(d);

    for (int z = 0; z < d; z++)
    {
        cube[z].resize(h);
        for (int y = 0; y < h; y++)
        {
            cube[z][y].resize(w);
            for (int x = 0; x < w; x++)
            {
                cube[z][y][x] = glm::vec3(0.0f);
            }
        }
    }

    int x, y, z;
    float entry[3];

    while (reader)
    {
        reader >> x >> y >> z;
        reader >> entry[0] >> entry[1] >> entry[2];

        cube[z][y][x] = glm::vec3(entry[0], entry[1], entry[2]);
    }

    std::ofstream writer("separator_hires.txt");

    writer << w << h << d << std::endl;
    for (int z = 0; z < d; z++)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                writer << cube[z][y][x].x << " " << cube[z][y][x].y << " " << cube[z][y][x].z << std::endl;
            }
        }
    }

    return 0;
}
