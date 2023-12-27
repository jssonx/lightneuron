#include "relu.h"

float relu(float x)
{
    return fmaxf(0.0f, x);
}