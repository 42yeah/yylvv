// Counter
// Counts how many floats are there in a file
// This is used to test-parse VTK file.

#include <iostream>
#include <fstream>

int count_floats(const std::string &path)
{
    std::ifstream reader(path);
    if (!reader.good())
    {
        return -1;
    }

    int count = 0;
    float trash = 0.0f;
    while (reader >> trash)
    {
        count++;
    }
    return count;
}

int count_binary(const std::string &path)
{
    std::ifstream bin_reader(path, std::ios::binary);

    if (!bin_reader.good())
    {
        return -1;
    }

    float f;
    int num_trials = 6;
    int count = 0;
    while (bin_reader.read((char *) &f, sizeof(f)))
    {
        count++;
        if (num_trials > 0)
        {
            std::cout << f << " ";
            num_trials--;
        }
    }
    std::cout << std::endl;
    return count;
}

int main()
{
    std::cout << count_floats("floats.txt") << std::endl;

    return 0;
}
