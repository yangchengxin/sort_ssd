#include <iostream>

#include "hungarian_optimizer.h"

/**
 * @brief Update the costs matrix of hungarian optimizer.
 *
 * @param association_mat The association matrix of tracks and objects, which
 * represents the bipartite graph to be optimized.
 * @param costs The costs matrix of hungarian optimizer.
 */
void UpdateCosts(const std::vector<std::vector<float>>& association_mat,
    SecureMat<float>* costs) {
    size_t rows_size = association_mat.size();
    size_t cols_size = rows_size > 0 ? association_mat.at(0).size() : 0;

    /* costs对象进行最大代价矩阵的属性的设置，以及对height和width的设置 */
    (*costs).Resize(rows_size, cols_size);

    /* 遍历代价矩阵，为(*costs)mat_对象进行赋值 */
    for (size_t row_idx = 0; row_idx < rows_size; ++row_idx) {
        for (size_t col_idx = 0; col_idx < cols_size; ++col_idx) {
            (*costs)(row_idx, col_idx) = association_mat.at(row_idx).at(col_idx);
        }
    }

    /* 打印赋值后的(*costs)的mat_元素 */
    printf("cost matrix：\n%s\n", (*costs).ToString().c_str());
}

/**
 * @brief Print the assignments result.
 *
 * @param assignments Assignments result to be printed.
 */
void PrintAssignments(
    const std::vector<std::pair<size_t, size_t>>& assignments) {
    std::cout << "\nThe assignments result are: \n" << std::endl;

    for (const auto& assignment : assignments) {
        std::cout << "    (" << assignment.first << ", " << assignment.second << ")"
            << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    /*std::vector<std::vector<float>> association_mat = { {82.0f, 83.0f, 69.0f},
                                                       {77.0f, 37.0f, 49.0f},
                                                       {11.0f, 69.0f, 5.0f},
                                                       {8.0f, 9.0f, 98.0f} };*/

                                                       /* 给定一个两个分布之间的cost matrix */
    std::vector<std::vector<float>> association_mat = { {10.0f, 15.0f, 9.0f},
                                                   {9.0f, 18.0f, 5.0f},
                                                   {6.0f, 14.0f, 3.0f} };

    /* 实例化一个KM算法对象 */
    HungarianOptimizer<float> optimizer;

    /* 创建一个用来存储匹配对象索引的pair数组 */
    std::vector<std::pair<size_t, size_t>> assignments;

    /* 更新匈牙利匹配对象的cost matrix属性 */
    UpdateCosts(association_mat, optimizer.costs());

    // entry of hungarian optimizer minimum-weighted matching
    optimizer.Minimize(&assignments);

    PrintAssignments(assignments);

    return 0;
}
