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
#include <string>
using namespace std;


void validateData(vector<vector<string>>& rows, size_t colsSize) {
    for (auto it = rows.begin(); it != rows.end();) {
        if (it->size() != colsSize) {
            it = rows.erase(it);
        }
        else {
            ++it;
        }
    }
}


void handleMissing(vector<vector<string>>& rows, bool treatAsNeutral, size_t colsSize) {
    for (size_t col = 1; col < colsSize; ++col) {
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
                })
                ->first;
        }
        for (auto& row : rows) {
            if (row[col] == "?") {
                row[col] = treatAsNeutral ? "neutral" : mode;
            }
        }
    }
}


pair<vector<vector<string>>, vector<vector<string>>> stratifiedSplit(
    const vector<vector<string>>& rows, float ratio) {
    map<string, vector<vector<string>>> classes;
    for (const auto& row : rows) {
        classes[row[0]].push_back(row);
    }

    vector<vector<string>> train, test;
    for (auto& cls : classes) {
        auto& samples = cls.second;
        size_t trainSize = static_cast<size_t>(samples.size() * ratio);
        train.insert(train.end(), samples.begin(), samples.begin() + trainSize);
        test.insert(test.end(), samples.begin() + trainSize, samples.end());
    }

    shuffle(train.begin(), train.end(), random_device());
    shuffle(test.begin(), test.end(), random_device());
    return { train, test };
}


map<string, map<string, map<string, double>>> trainNBC(
    const vector<vector<string>>& rows, double laplace, map<string, int>& classCounts, int& totalSamples) {
    map<string, map<string, map<string, double>>> probs;
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

    return probs;
}


string predictNBC(const vector<string>& instance, const map<string, map<string, map<string, double>>>& probs,
    const map<string, int>& classCounts, int totalSamples, double laplace) {
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
                logProbs[label] += log(laplace / (classCounts.at(label) + laplace * probs.at(label).at(to_string(i)).size()));
            }
        }
    }

    return max_element(logProbs.begin(), logProbs.end(),
        [](const pair<string, double>& a, const pair<string, double>& b) {
            return a.second < b.second;
        })
        ->first;
}


double calculateAccuracy(const vector<vector<string>>& rows,
    const map<string, map<string, map<string, double>>>& probs,
    const map<string, int>& classCounts, int totalSamples, double laplace) {
    int correct = 0;
    for (const auto& row : rows) {
        if (predictNBC(row, probs, classCounts, totalSamples, laplace) == row[0]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / rows.size() * 100;
}


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

    validateData(rows, cols.size());

    if (rows.empty()) {
        cerr << "Error: Invalid rows." << endl;
        return 1;
    }

    int input;
    cout << "Enter 0 or 1: ";
    cin >> input;

    handleMissing(rows, input == 0, cols.size());

    auto [train, test] = stratifiedSplit(rows, 0.8);

    double lambda = 1.0;
    map<string, int> classCounts;
    int totalSamples;

    auto probs = trainNBC(train, lambda, classCounts, totalSamples);

    double trainAcc = calculateAccuracy(train, probs, classCounts, totalSamples, lambda);
    cout << "1. Train Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << trainAcc << "%" << endl;

    vector<double> foldAccs;
    size_t foldSize = train.size() / 10;

    for (size_t i = 0; i < 10; ++i) {
        vector<vector<string>> valFold(train.begin() + i * foldSize,
            (i == 9) ? train.end() : train.begin() + (i + 1) * foldSize);

        vector<vector<string>> trainFold;
        trainFold.insert(trainFold.end(), train.begin(), train.begin() + i * foldSize);
        if (i != 9) {
            trainFold.insert(trainFold.end(), train.begin() + (i + 1) * foldSize, train.end());
        }

        map<string, int> foldClassCounts;
        int foldTotalSamples;
        auto foldProbs = trainNBC(trainFold, lambda, foldClassCounts, foldTotalSamples);

        double acc = calculateAccuracy(valFold, foldProbs, foldClassCounts, foldTotalSamples, lambda);
        foldAccs.push_back(acc);

        cout << "    Accuracy Fold " << i + 1 << ": " << fixed << setprecision(2) << acc << "%" << endl;
    }

    double meanAcc = accumulate(foldAccs.begin(), foldAccs.end(), 0.0) / foldAccs.size();
    double stdDevAcc = stdDev(foldAccs, meanAcc);

    cout << "\n10-Fold Cross-Validation Results:\n";
    cout << "    Average Accuracy: " << fixed << setprecision(2) << meanAcc << "%" << endl;
    cout << "    Standard Deviation: " << fixed << setprecision(2) << stdDevAcc << "%" << endl;

    double testAcc = calculateAccuracy(test, probs, classCounts, totalSamples, lambda);
    cout << "\n2. Test Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << testAcc << "%" << endl;
    cout << endl << "Lambda: " << lambda << endl;

    return 0;
}
