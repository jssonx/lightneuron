#include "unity/unity.h"
#include "test_conv.h"
#include "test_nn.h"
#include "test_functional.h"
#include "test_linear.h"
#include "test_matrix_ops.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();

    // Test conv
    RUN_TEST(test_conv);

    // Test nn
    RUN_TEST(test_flatten_basic);
    RUN_TEST(test_predict_simple_array);
    RUN_TEST(test_predict_all_same_values);
    RUN_TEST(test_predict_mix_of_negatives_and_positives);


    // Test functional
    RUN_TEST(test_softmax_basic);
    RUN_TEST(test_relu);

    // Test linear
    RUN_TEST(test_linear_basic);
    RUN_TEST(test_linear_basic2);

    // Test matrix_ops
    RUN_TEST(test_matmul_square_matrices);
    RUN_TEST(test_matmul_incompatible_dimensions);

    return UNITY_END();
}
