#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
using namespace std;

class Data {
private:
    vector<vector<string>> rows;
    vector<string> cols;

public:
    Data(const vector<vector<string>>& r, const vector<string>& c) : rows(r), cols(c) {}

    void validate() {
        for (auto it = rows.begin(); it != rows.end();) {
            if (it->size() != cols.size()) {
                it = rows.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    void handleMissing(bool treatAsNeutral) {
        for (size_t col = 1; col < cols.size(); ++col) {
            map<string, int> counts;
            for (const auto& row : rows) {
                if (row[col] != "?") {
                    counts[row[col]]++;
                }
            }
            string mode = "neutral";
            if (!counts.empty()) {
                mode = max_element(counts.begin(), counts.end(),
                    [](const pair<string, int>& a, const pair<string, int>& b) {
                        return a.second < b.second;
                    })->first;
            }
            for (auto& row : rows) {
                if (row[col] == "?") {
                    row[col] = treatAsNeutral ? "neutral" : mode;
                }
            }
        }
    }

    pair<Data, Data> split(float ratio) const {
        vector<vector<string>> train, test;
        vector<vector<string>> shuffled = rows;
        shuffle(shuffled.begin(), shuffled.end(), random_device());

        size_t trainSize = static_cast<size_t>(shuffled.size() * ratio);
        train.insert(train.end(), shuffled.begin(), shuffled.begin() + trainSize);
        test.insert(test.end(), shuffled.begin() + trainSize, shuffled.end());

        return { Data(train, cols), Data(test, cols) };
    }

    const vector<vector<string>>& getRows() const { return rows; }
    const vector<string>& getCols() const { return cols; }
};


class NBC {
private:
    map<string, map<string, map<string, double>>> probs;
    map<string, int> classCounts;
    int totalSamples;

public:
    void train(const Data& data, double laplace) {
        const auto& rows = data.getRows();
        totalSamples = rows.size();
        int numFeatures = rows[0].size() - 1;

        for (const auto& row : rows) {
            string label = row[0];
            classCounts[label]++;

            for (size_t i = 1; i <= numFeatures; ++i) {
                probs[label][to_string(i)][row[i]]++;
            }
        }

        for (auto& cls : probs) {
            string label = cls.first;
            for (auto& feat : cls.second) {
                auto& counts = feat.second;
                int total = classCounts[label] + laplace * counts.size();
                for (auto& count : counts) {
                    count.second = (count.second + laplace) / total;
                }
            }
        }
    }

    string predict(const vector<string>& instance, double laplace = 1.0) const {
        map<string, double> logProbs;
        for (const auto& cls : classCounts) {
            string label = cls.first;
            logProbs[label] = log(static_cast<double>(cls.second) / totalSamples);

            for (size_t i = 1; i < instance.size(); ++i) {
                auto it = probs.at(label).find(to_string(i));
                if (it != probs.at(label).end() && it->second.count(instance[i])) {
                    logProbs[label] += log(it->second.at(instance[i]));
                }
                else {
                    logProbs[label] += log(laplace /
                        (classCounts.at(label) + laplace * probs.at(label).at(to_string(i)).size()));
                }
            }
        }

        return max_element(logProbs.begin(), logProbs.end(),
            [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second < b.second;
            })->first;
    }

    double accuracy(const Data& data) const {
        const auto& rows = data.getRows();
        int correct = 0;
        for (const auto& row : rows) {
            if (predict(row) == row[0]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / rows.size() * 100;
    }
};

double stdDev(const vector<double>& values, double mean) {
    double variance = 0.0;
    for (double v : values) {
        variance += (v - mean) * (v - mean);
    }
    return sqrt(variance / values.size());
}

int main() {
    ifstream file("house-votes-84.data");
    if (!file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        return 1;
    }

    vector<vector<string>> rows;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }
        rows.push_back(row);
    }

    if (rows.empty()) {
        cerr << "Error: The file is empty or contains invalid data." << endl;
        return 1;
    }

    vector<string> cols = { "Class" };
    for (int i = 1; i <= 16; ++i) {
        cols.push_back("Attr" + to_string(i));
    }

    Data data(rows, cols);
    data.validate();

    if (data.getRows().empty()) {
        cerr << "Error: Invalid rows." << endl;
        return 1;
    }

    int input;
    cout << "Enter 0 or 1: ";
    cin >> input;

    data.handleMissing(input == 0);

    auto [train, test] = data.split(0.8);

    NBC nbc;
    nbc.train(train, 1.0);


    double trainAcc = nbc.accuracy(train);
    cout << "1. Train Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << trainAcc << "%" << endl;

    vector<double> foldAccs;
    const auto& trainRows = train.getRows();
    size_t foldSize = trainRows.size() / 10;

    for (size_t i = 0; i < 10; ++i) {
        vector<vector<string>> valFold(trainRows.begin() + i * foldSize,
            (i == 9) ? trainRows.end() : trainRows.begin() + (i + 1) * foldSize);

        vector<vector<string>> trainFold;
        trainFold.insert(trainFold.end(), trainRows.begin(), trainRows.begin() + i * foldSize);
        if (i != 9) {
            trainFold.insert(trainFold.end(), trainRows.begin() + (i + 1) * foldSize, trainRows.end());
        }

        Data trainData(trainFold, train.getCols());
        Data valData(valFold, train.getCols());

        NBC foldNBC;
        foldNBC.train(trainData, 1.0);

        double acc = foldNBC.accuracy(valData);
        foldAccs.push_back(acc);

        cout << "    Accuracy Fold " << i + 1 << ": " << fixed << setprecision(2) << acc << "%" << endl;
    }

    double meanAcc = accumulate(foldAccs.begin(), foldAccs.end(), 0.0) / foldAccs.size();
    double stdDevAcc = stdDev(foldAccs, meanAcc);

    cout << "\n10-Fold Cross-Validation Results:\n";
    cout << "    Average Accuracy: " << fixed << setprecision(2) << meanAcc << "%" << endl;
    cout << "    Standard Deviation: " << fixed << setprecision(2) << stdDevAcc << "%" << endl;


    double testAcc = nbc.accuracy(test);
    cout << "\n2. Test Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << testAcc << "%" << endl;

    return 0;
}
