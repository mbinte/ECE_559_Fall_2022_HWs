#include <stddef.h>

static float score(const float *weights, const float *row) {
    float value = 0.0f;
    for (int j = 0; j < 784; ++j) value += weights[j] * row[j];
    return value;
}

void online_epoch(float *weights, const float *images, const unsigned char *labels,
                  size_t count, float eta) {
    for (size_t i = 0; i < count; ++i) {
        const float *row = images + i * 784;
        for (int k = 0; k < 10; ++k) {
            float *wk = weights + k * 784;
            int predicted = score(wk, row) >= 0.0f;
            float change = eta * ((k == labels[i]) - predicted);
            if (change != 0.0f)
                for (int j = 0; j < 784; ++j) wk[j] += change * row[j];
        }
    }
}

size_t errors(const float *weights, const float *images,
              const unsigned char *labels, size_t count) {
    size_t wrong = 0;
    for (size_t i = 0; i < count; ++i) {
        const float *row = images + i * 784;
        int best = 0;
        float largest = score(weights, row);
        for (int k = 1; k < 10; ++k) {
            float candidate = score(weights + k * 784, row);
            if (candidate > largest) {
                largest = candidate;
                best = k;
            }
        }
        wrong += best != labels[i];
    }
    return wrong;
}
