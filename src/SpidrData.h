#pragma once

#include <QObject>
#include <QMetaType>

#include <vector>
#include <cassert>

class TsneData : public QObject
{
    Q_OBJECT
public:
    TsneData() :
        _numPoints(0),
        _numDimensions(0)
    {
    }

    TsneData(const TsneData& other)
    {
        _numPoints = other._numPoints;
        _numDimensions = other._numDimensions;
        _data = other._data;
    }

    ~TsneData()
    {
    }

    unsigned int getNumPoints() const
    {
        return _numPoints;
    }

    unsigned int getNumDimensions() const
    {
        return _numDimensions;
    }

    const std::vector<float>& getData() const
    {
        return _data;
    }

    // Just here because computeJointProbabilityDistribution doesn't take a const vector
    std::vector<float>& getDataNonConst()
    {
        return _data;
    }

    void assign(unsigned int numPoints, unsigned int numDimensions, const std::vector<float>& inputData)
    {
        assert(inputData.size() == numPoints * numDimensions);
        
        _numPoints = numPoints;
        _numDimensions = numDimensions;
        _data = inputData;
    }

private:
    unsigned int _numPoints;
    unsigned int _numDimensions;

    std::vector<float> _data;
};

Q_DECLARE_METATYPE(TsneData);
