#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace chrono;

int n;
bool has_conflicts;
int* q_pos;
int* row_conf;
int* diag1;
int* diag2;

void initQueens() {
    int col = 1;
    for (int row = 0; row < n; row++) {
        q_pos[col] = row;
        row_conf[row]++;
        diag1[col - row + n - 1]++;
        diag2[col + row]++;
        col += 2;
        if (col >= n) {
            col = 0;
        }
    }
}

int getMaxConfCol() {
    int max_conf = -1;
    vector<int> max_conf_cols;
    for (int col = 0; col < n; col++) {
        int row = q_pos[col];
        int conf = row_conf[row] + diag1[col - row + n - 1] + diag2[col + row] - 3;
        if (conf == max_conf) {
            max_conf_cols.push_back(col);
        }
        else if (conf > max_conf) {
            max_conf = conf;
            max_conf_cols = { col };
        }
    }
    if (max_conf == 0) {
        has_conflicts = false;
    }
    return max_conf_cols[rand() % max_conf_cols.size()];
}

int getMinConfRow(int col) {
    int min_conf = n + 1;
    vector<int> min_conf_rows;
    for (int row = 0; row < n; row++) {
        int conf = row_conf[row] + diag1[col - row + n - 1] + diag2[col + row];
        if (q_pos[col] == row) {
            conf -= 3;
        }
        if (conf == min_conf) {
            min_conf_rows.push_back(row);
        }
        else if (conf < min_conf) {
            min_conf = conf;
            min_conf_rows = { row };
        }
    }
    return min_conf_rows[rand() % min_conf_rows.size()];
}

void moveQueen(int row, int col) {
    int old_row = q_pos[col];
    row_conf[old_row]--;
    diag1[col - old_row + n - 1]--;
    diag2[col + old_row]--;

    q_pos[col] = row;
    row_conf[row]++;
    diag1[col - row + n - 1]++;
    diag2[col + row]++;
}

void minConflicts() {
    int steps = 0;
    int max_steps = n;
    while (steps++ <= max_steps) {
        int col = getMaxConfCol();
        if (!has_conflicts) break;
        int row = getMinConfRow(col);
        moveQueen(row, col);
    }
    if (has_conflicts) {
        minConflicts();
    }
}

void printQueens() {
    cout << "[";
    for (int i = 0; i < n; i++) {
        cout << q_pos[i];
        if (i < n - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

int main() {
    cin >> n;

    if (n < 4) {
        cout << -1 << endl;
        return 0;
    }

    q_pos = new int[n];
    row_conf = new int[n] {0};
    diag1 = new int[2 * n - 1]{ 0 };
    diag2 = new int[2 * n - 1]{ 0 };

    has_conflicts = true;
    auto start = high_resolution_clock::now();

    initQueens();
    minConflicts();

    auto end = high_resolution_clock::now();
    double exec_time = duration_cast<chrono::duration<double>>(end - start).count();

    if (n > 100) {
        if (exec_time < 0.01) {
            cout  << fixed << setprecision(5) << exec_time << endl;
        }
        else {
            cout << fixed << setprecision(2) << exec_time << endl;
        }
    }
    else {
        printQueens();
    }

    delete[] q_pos;
    delete[] row_conf;
    delete[] diag1;
    delete[] diag2;
    return 0;
}
