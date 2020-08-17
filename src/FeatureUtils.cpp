#include "FeatureUtils.h"


template<typename T>
std::vector<float> NormVector(std::vector<T> vec, float normVal) {
    std::vector<float> normedVec(vec.size(), 0);
    for (unsigned int i = 0; i < vec.size(); i++) {
        normedVec[i] = vec[i] / normVal;
    }
    return normedVec;
}

std::vector<unsigned int> PascalsTriangleRow(const unsigned int n) {
    std::vector<unsigned int> row(n + 1, 1);
    unsigned int entry = 1;
    for (unsigned int i = 1; i < n + 1; i++) {
        entry = (unsigned int)(entry * (n + 1 - i) / i);
        row[i] = entry;
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
            bino2D[row*width + col] = bino1D[row] * bino1D[col];

            // helper for normalization
            sum += +bino2D[row*width + col];
            if (bino2D[row*width + col] > (float)max)
                max = bino2D[row*width + col];
        }
    }

    // normalization
    if (norm == norm_vec::NORM_MAX)
        bino2D = NormVector(bino2D, max);
    else if (norm == norm_vec::NORM_SUM)
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
        kernel[coutner] = std::exp(-1 * (i*i) / (2 * sd * sd));
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
            gauss2D[row*width + col] = gauss1D[row] *  gauss1D[col];

            // helper for normalization
            sum += +gauss2D[row*width + col];
            if (gauss2D[row*width + col] > (float)max)
                max = gauss2D[row*width + col];
        }
    }

    // normalization
    if (norm == norm_vec::NORM_MAX)
        gauss2D = NormVector(gauss2D, max);
    else if (norm == norm_vec::NORM_SUM)
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

std::vector<int> neighborhoodIndices(const unsigned int pointInd, const size_t locNeighbors, const QSize imgSize, const std::vector<unsigned int>& pointIds) {
    size_t kernelWidth = (2 * locNeighbors) + 1;
    size_t neighborhoodSize = kernelWidth * kernelWidth;
    std::vector<int> neighborsIDs(neighborhoodSize, -1);
    int imWidth = imgSize.width();
    int rowID = int(pointInd / imWidth);

    // left and right neighbors
    std::vector<int> lrNeighIDs(kernelWidth, 0);
    std::iota(lrNeighIDs.begin(), lrNeighIDs.end(), pointInd - locNeighbors);

    // are left and right out of the picture?
    for (int& n : lrNeighIDs) {
        if (n < rowID * imWidth)
            n = -1;
        else if (n >= (rowID + 1) * imWidth)
            n = -1;
    }

    // above and below neighbors
    unsigned int localNeighCount = 0;
    for (int i = -1 * locNeighbors; i <= (int)locNeighbors; i++) {
        for (int ID : lrNeighIDs) {
            neighborsIDs[localNeighCount] = (ID != -1) ? ID + i * imgSize.width() : -1;  // if left or right is already out of image, above and below will be as well
            localNeighCount++;
        }
    }

    // Check if neighborhood IDs are in selected points
    for (int& ID : neighborsIDs) {
        // if neighbor is not in neighborhood, assign -1
        if (std::find(pointIds.begin(), pointIds.end(), ID) == pointIds.end()) {
            ID = -1;
        }
    }

    return neighborsIDs;
}

std::vector<float> getNeighborhoodValues(const std::vector<int>& neighborIDs, const std::vector<float>& attribute_data, const size_t neighborhoodSize, const size_t numDims) {
    std::vector<float> neighborValues;
    neighborValues.resize(neighborhoodSize * numDims);
    for (unsigned int neighbor = 0; neighbor < neighborhoodSize; neighbor++) {
        for (unsigned int dim = 0; dim < numDims; dim++) {
            neighborValues[neighbor * numDims + dim] = (neighborIDs[neighbor] != -1) ? attribute_data[neighborIDs[neighbor] * numDims + dim] : 0;
        }
    }
    return neighborValues;
}
