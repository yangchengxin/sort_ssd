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

    /* costs������������۾�������Ե����ã��Լ���height��width������ */
    (*costs).Resize(rows_size, cols_size);

    /* �������۾���Ϊ(*costs)mat_������и�ֵ */
    for (size_t row_idx = 0; row_idx < rows_size; ++row_idx) {
        for (size_t col_idx = 0; col_idx < cols_size; ++col_idx) {
            (*costs)(row_idx, col_idx) = association_mat.at(row_idx).at(col_idx);
        }
    }

    /* ��ӡ��ֵ���(*costs)��mat_Ԫ�� */
    printf("cost matrix��\n%s\n", (*costs).ToString().c_str());
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

                                                       /* ����һ�������ֲ�֮���cost matrix */
    std::vector<std::vector<float>> association_mat = { {10.0f, 15.0f, 9.0f},
                                                   {9.0f, 18.0f, 5.0f},
                                                   {6.0f, 14.0f, 3.0f} };

    /* ʵ����һ��KM�㷨���� */
    HungarianOptimizer<float> optimizer;

    /* ����һ�������洢ƥ�����������pair���� */
    std::vector<std::pair<size_t, size_t>> assignments;

    /* ����������ƥ������cost matrix���� */
    UpdateCosts(association_mat, optimizer.costs());

    // entry of hungarian optimizer minimum-weighted matching
    optimizer.Minimize(&assignments);

    PrintAssignments(assignments);

    return 0;
}
