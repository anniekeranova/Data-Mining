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
#include <memory>
using namespace std;

const vector<string> attributes = { "Class", "age", "menopause", "tumor-size", "inv-nodes",
                             "node-caps", "deg-malig", "breast", "breast-quad", "irradiat" };

struct Node {
    string attribute;
    map<string, shared_ptr<Node>> children;
    bool isLeaf = false;
    string classification;
};

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

void handleMissing(vector<vector<string>>& rows) {
    map<string, map<string, int>> classAttributeCounts;
    for (const auto& row : rows) {
        string cls = row[0];
        for (size_t i = 1; i < row.size(); ++i) {
            if (row[i] != "?") {
                classAttributeCounts[cls][row[i]]++;
            }
        }
    }
    for (auto& row : rows) {
        string cls = row[0];
        for (size_t i = 1; i < row.size(); ++i) {
            if (row[i] == "?") {
                row[i] = max_element(classAttributeCounts[cls].begin(), classAttributeCounts[cls].end(),
                    [](const pair<string, int>& a, const pair<string, int>& b) {
                        return a.second < b.second;
                    })
                    ->first;
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



double calculateEntropy(const vector<vector<string>>& rows) {
    map<string, int> classCounts;
    for (const auto& row : rows) {
        classCounts[row[0]]++;
    }

    double entropy = 0.0;
    for (const auto& [label, count] : classCounts) {
        double p = static_cast<double>(count) / rows.size();
        entropy -= p * log2(p);
    }
    return entropy;
}

double informationGain(const vector<vector<string>>& rows, const string& attribute) {
    int attributeIndex = find(attributes.begin(), attributes.end(), attribute) - attributes.begin();
    map<string, vector<vector<string>>> subsets;

    for (const auto& row : rows) {
        subsets[row[attributeIndex]].push_back(row);
    }

    double totalEntropy = calculateEntropy(rows);
    double subsetEntropy = 0.0;

    for (const auto& [value, subset] : subsets) {
        double p = static_cast<double>(subset.size()) / rows.size();
        subsetEntropy += p * calculateEntropy(subset);
    }

    return totalEntropy - subsetEntropy;
}


string majorityClass(const vector<vector<string>>& rows) {
    map<string, int> classCounts;
    for (const auto& row : rows) {
        classCounts[row[0]]++;
    }

    return max_element(classCounts.begin(), classCounts.end(),
        [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second < b.second;
        })
        ->first;
}

shared_ptr<Node> buildTree(const vector<vector<string>>& rows, const vector<string>& attributes, int minSamples) {
    if (rows.empty()) return nullptr;

    map<string, int> classCounts;
    for (const auto& row : rows) {
        classCounts[row[0]]++;
    }
    if (classCounts.size() == 1) {
        auto leaf = make_shared<Node>();
        leaf->isLeaf = true;
        leaf->classification = rows[0][0];
        return leaf;
    }

    if (rows.size() < minSamples) {
        auto leaf = make_shared<Node>();
        leaf->isLeaf = true;
        leaf->classification = majorityClass(rows);
        return leaf;
    }

    string bestAttribute;
    double bestGain = -1;
    for (const auto& attribute : attributes) {
        if (attribute != "Class") {
            double gain = informationGain(rows, attribute);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attribute;
            }
        }
    }

    if (bestGain == -1) {
        auto leaf = make_shared<Node>();
        leaf->isLeaf = true;
        leaf->classification = majorityClass(rows);
        return leaf;
    }

    auto node = make_shared<Node>();
    node->attribute = bestAttribute;

    int attributeIndex = find(attributes.begin(), attributes.end(), bestAttribute) - attributes.begin();
    map<string, vector<vector<string>>> subsets;
    for (const auto& row : rows) {
        subsets[row[attributeIndex]].push_back(row);
    }

    vector<string> remainingAttributes = attributes;
    remainingAttributes.erase(find(remainingAttributes.begin(), remainingAttributes.end(), bestAttribute));

    for (const auto& [value, subset] : subsets) {
        node->children[value] = buildTree(subset, remainingAttributes, minSamples);
    }

    return node;
}


string predict(const shared_ptr<Node>& tree, const vector<string>& instance) {
    if (tree->isLeaf) return tree->classification;

    auto it = find(attributes.begin(), attributes.end(), tree->attribute);
    if (it != attributes.end()) {
        int index = distance(attributes.begin(), it);
        string value = instance[index];
        if (tree->children.find(value) != tree->children.end()) {
            return predict(tree->children.at(value), instance);
        }
    }

    return "Unknown";
}

double calculateAccuracy(const shared_ptr<Node>& tree, const vector<vector<string>>& dataset) {
    int correct = 0;
    for (const auto& instance : dataset) {
        if (predict(tree, instance) == instance[0]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / dataset.size() * 100;
}

double stdDev(const vector<double>& values, double mean) {
    double variance = 0.0;
    for (double v : values) {
        variance += (v - mean) * (v - mean);
    }
    return sqrt(variance / values.size());
}


void reducedErrorPruning(shared_ptr<Node> root, const vector<vector<string>>& validationSet) {
    if (root == nullptr || root->isLeaf) return;

    for (auto& [value, child] : root->children) {
        reducedErrorPruning(child, validationSet);
    }

    bool allChildrenAreLeaves = true;
    for (const auto& [value, child] : root->children) {
        if (!child->isLeaf) {
            allChildrenAreLeaves = false;
            break;
        }
    }

    if (allChildrenAreLeaves) {
        string majority = majorityClass(validationSet);
        int correctBefore = 0, correctAfter = 0;

        for (const auto& instance : validationSet) {
            string predictedBefore = predict(root, instance);
            if (predictedBefore == instance[0]) correctBefore++;

            if (majority == instance[0]) correctAfter++;
        }

        if (correctAfter >= correctBefore) {
            root->isLeaf = true;
            root->classification = majority;
            root->children.clear();
        }
    }
}



int main() {
    ifstream file("breast-cancer.data");
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

    validateData(rows, attributes.size());
    handleMissing(rows);

    auto [train, test] = stratifiedSplit(rows, 0.8);

    cout << "Enter 0, 1 or 2: ";
    int pruningType;
    cin >> pruningType;

    int minSamples = 10;
    shared_ptr<Node> decisionTree = nullptr;

    if (pruningType == 0 || pruningType == 2) {
        cout << "\nPre-pruning (min samples = " << minSamples << ") " << endl;
        decisionTree = buildTree(train, attributes, minSamples);
    }
    else {
        cout << "Without pre-pruning" << endl;
        decisionTree = buildTree(train, attributes, 1);
    }


    double trainAcc = calculateAccuracy(decisionTree, train);
    cout << "\n1. Train Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << trainAcc << "%" << endl;

    cout << "\n10-Fold Cross-Validation Results:" << endl;
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

        auto foldTree = buildTree(trainFold, attributes, (pruningType == 0 || pruningType == 2) ? minSamples : 1);
        double acc = calculateAccuracy(foldTree, valFold);
        foldAccs.push_back(acc);

        cout << "    Accuracy Fold " << i + 1 << ": " << fixed << setprecision(2) << acc << "%" << endl;
    }

    double meanAcc = accumulate(foldAccs.begin(), foldAccs.end(), 0.0) / foldAccs.size();
    double stdDevAcc = stdDev(foldAccs, meanAcc);

    cout << "\n    Average Accuracy: " << fixed << setprecision(2) << meanAcc << "%" << endl;
    cout << "    Standard Deviation: " << fixed << setprecision(2) << stdDevAcc << "%" << endl;

    if (pruningType == 1 || pruningType == 2) {
        cout << "\nPost-pruning: Reduced Error Pruning" << endl;
        reducedErrorPruning(decisionTree, test);
    }

    double testAcc = calculateAccuracy(decisionTree, test);
    cout << "\n2. Test Set Accuracy:\n    Accuracy: " << fixed << setprecision(2) << testAcc << "%" << endl;

    return 0;
}

