#include "FeatureUtils.h"


template<typename T>
std::vector<float> NormVector(std::vector<T> vec, float normVal) {
    std::vector<float> normedVec(vec.size(), 0);
    for (unsigned int i = 0; i < vec.size(); i++) {
        normedVec.at(i) = vec.at(i) / normVal;
    }
    return normedVec;
}

std::vector<unsigned int> PascalsTriangleRow(const unsigned int n) {
    std::vector<unsigned int> row(n + 1, 1);
    unsigned int entry = 1;
    for (unsigned int i = 1; i < n + 1; i++) {
        entry = (unsigned int)(entry * (n + 1 - i) / i);
        row.at(i) = entry;
    }
    return row;
}

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> BinomialKernel2D(const unsigned int width, norm_vec norm) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");

    std::vector<unsigned int> bino1D = PascalsTriangleRow(width - 1);
    std::vector<float> bino2D(width * width, -1);

    // helper for normalization
    int sum = 0;
    int max = 0;

    // outter product
    for (unsigned int row = 0; row < width; row++) {
        for (unsigned int col = 0; col < width; col++) {
            bino2D.at(row*width + col) = bino1D.at(row) * bino1D.at(col);

            // helper for normalization
            sum += +bino2D.at(row*width + col);
            if (bino2D.at(row*width + col) > (float)max)
                max = bino2D.at(row*width + col);
        }
    }

    // normalization
    if (norm == 1)
        bino2D = NormVector(bino2D, max);
    else if (norm == 2)
        bino2D = NormVector(bino2D, sum);

    return bino2D;
}

std::vector<float> GaussianKernel1D(const unsigned int width, const float sd) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");
    if (sd < 0)
        throw std::invalid_argument("sd must be positive");

    std::vector<float> kernel(width, 0);
    int coutner = 0;
    for (int i = (-1 * ((int)width - 1) / 2); i <= ((int)width - 1) / 2; i++) {
        kernel.at(coutner) = std::exp(-1 * (i*i) / (2 * sd * sd));
        coutner++;
    }
    return kernel;

}

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> GaussianKernel2D(const unsigned int width, const float sd, norm_vec norm) {
    if (width % 2 == 0)
        throw std::invalid_argument("n must be odd");
    if (sd < 0)
        throw std::invalid_argument("sd must be positive");

    std::vector<float> gauss1D = GaussianKernel1D(width);
    std::vector<float> gauss2D(width * width, 0);

    // helper for normalization
    float sum = 0;
    float max = 0;

    // outter product
    for (unsigned int row = 0; row < width; row++) {
        for (unsigned int col = 0; col < width; col++) {
            gauss2D.at(row*width + col) = gauss1D.at(row) *  gauss1D.at(col);

            // helper for normalization
            sum += +gauss2D.at(row*width + col);
            if (gauss2D.at(row*width + col) > (float)max)
                max = gauss2D.at(row*width + col);
        }
    }

    // normalization
    if (norm == 1)
        gauss2D = NormVector(gauss2D, max);
    else if (norm == 2)
        gauss2D = NormVector(gauss2D, sum);

    return gauss2D;
}

unsigned int SqrtBinSize(unsigned int numItems) {
    return int(std::ceil(std::sqrt(numItems)));
}

unsigned int SturgesBinSize(unsigned int numItems) {
    return int(std::ceil(std::log2(numItems) + 1));
}

unsigned int RiceBinSize(unsigned int numItems) {
    return int(2 * std::ceil(std::pow(numItems, 1.0/3)));
}



