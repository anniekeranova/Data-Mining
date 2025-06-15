#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <string>
#include <unordered_set>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;


const vector<int> DIR_ROW = { 0, 0, -1, 1 };
const vector<int> DIR_COL = { -1, 1, 0, 0 };
const vector<string> DIR_NAME = { "right", "left", "down", "up" };


vector<int> goalState;
vector<int> goalPos;
int dim;
vector<string> finalPath;
bool foundSolution = false;


vector<int> buildGoalState(int N, int zeroIndex) {
    if (zeroIndex == -1) {
        zeroIndex = N;
    }
    vector<int> g(N + 1);
    int val = 1;
    for (int i = 0; i <= N; i++) {
        if (i == zeroIndex) {
            g[i] = 0;
        }
        else {
            g[i] = val++;
        }
    }
    return g;
}


vector<int> buildGoalPos(const vector<int>& goal) {
    int sz = (int)goal.size();
    vector<int> gPos(sz, -1);
    for (int i = 0; i < sz; i++) {
        int val = goal[i];
        if (val != 0) {
            gPos[val] = i;
        }
    }
    return gPos;
}


bool isGoal(const vector<int>& board) {
    return (board == goalState);
}


int manhattanDistance(const vector<int>& board) {
    int dist = 0;
    int sz = (int)board.size();
    for (int i = 0; i < sz; i++) {
        int val = board[i];
        if (val == 0) continue;
        int goalIdx = goalPos[val];
        int r1 = i / dim, c1 = i % dim;
        int r2 = goalIdx / dim, c2 = goalIdx % dim;
        dist += abs(r1 - r2) + abs(c1 - c2);
    }
    return dist;
}


string encode(const vector<int>& board) {
    ostringstream oss;
    for (int i = 0; i < (int)board.size(); i++) {
        oss << board[i] << ',';
    }
    return oss.str();
}

bool dfs(vector<int>& board, int g, int threshold, int& minNextThreshold,
    vector<string>& path, int lastMove) {
    int h = manhattanDistance(board);
    int f = g + h;
    if (f > threshold) {
        minNextThreshold = min(minNextThreshold, f);
        return false;
    }
    if (isGoal(board)) {
        finalPath = path;
        foundSolution = true;
        return true;
    }

    int zeroPos = -1;
    for (int i = 0; i < (int)board.size(); i++) {
        if (board[i] == 0) {
            zeroPos = i;
            break;
        }
    }
    int r = zeroPos / dim;
    int c = zeroPos % dim;

    for (int i = 0; i < 4; i++) {
        if (i == (lastMove ^ 1)) continue;

        int rr = r + DIR_ROW[i];
        int cc = c + DIR_COL[i];
        if (rr < 0 || rr >= dim || cc < 0 || cc >= dim)
            continue;

        int newPos = rr * dim + cc;
        swap(board[zeroPos], board[newPos]);
        path.push_back(DIR_NAME[i]);

        if (dfs(board, g + 1, threshold, minNextThreshold, path, i)) {
            return true;
        }

        path.pop_back();
        swap(board[zeroPos], board[newPos]);
    }

    return false;
}

bool idaStar(vector<int>& startBoard) {
    int threshold = manhattanDistance(startBoard);

    while (true) {
        int minNextThreshold = INT_MAX;
        vector<string> path;

        if (dfs(startBoard, 0, threshold, minNextThreshold, path, -1)) {
            return foundSolution;
        }
        if (minNextThreshold == INT_MAX) {
            return false;
        }
        threshold = minNextThreshold;

        if (threshold > 100000) {
            return false;
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, I;
    cin >> N >> I;

    dim = (int)sqrt(N + 1);

    vector<int> startBoard(N + 1);
    for (int i = 0; i < N + 1; i++) {
        cin >> startBoard[i];
    }

    goalState = buildGoalState(N, I);
    goalPos = buildGoalPos(goalState);

    auto start = high_resolution_clock::now(); 

    foundSolution = false;
    bool ok = idaStar(startBoard);

    auto end = high_resolution_clock::now(); 
    double exec_time = duration_cast<chrono::duration<double>>(end - start).count();

    if (!ok) {
        cout << -1 << "\n";
    }
    else {
        cout << finalPath.size() << "\n";
        for (auto& step : finalPath) {
            cout << step << "\n";
        }
    }

    cerr << "Execution Time: " << fixed << setprecision(3) << exec_time << " seconds" << endl;

    return 0;
}