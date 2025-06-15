#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;
using namespace chrono;

struct Item {
    int weight;
    int value;
};

int m, n;
vector<Item> items;

int randomInt(int min, int max) {
    return min + (rand() % (max - min + 1));
}

int fitness(const vector<int>& solution) {
    int totalWeight = 0, totalValue = 0;
    for (int i = 0; i < n; i++) {
        if (solution[i]) {
            totalWeight += items[i].weight;
            totalValue += items[i].value;
        }
    }
    return (totalWeight <= m) ? totalValue : 0;
}

vector<int> selectParent(const vector<vector<int>>& population, const vector<int>& fitnessVal) {
    int bestIdx = randomInt(0, population.size() - 1);
    for (int i = 1; i < 3; i++) {
        int idx = randomInt(0, population.size() - 1);
        if (fitnessVal[idx] > fitnessVal[bestIdx]) {
            bestIdx = idx;
        }
    }
    return population[bestIdx];
}

vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int point = randomInt(0, n - 1);
    vector<int> child(n);
    for (int i = 0; i < n; i++) {
        child[i] = (i < point) ? parent1[i] : parent2[i];
    }
    return child;
}

void mutate(vector<int>& solution) {
    int idx = randomInt(0, n - 1);
    solution[idx] = 1 - solution[idx];
}

void initializePopulation(vector<vector<int>>& population, int populationSize) {
    for (int i = 0; i < populationSize; i++) {
        vector<int> individual(n);
        int currentWeight = 0;
        for (int j = 0; j < n; j++) {
            individual[j] = randomInt(0, 1);
            if (individual[j]) {
                currentWeight += items[j].weight;
                if (currentWeight > m) {
                    individual[j] = 0; 
                }
            }
        }
        population.push_back(individual);
    }
}

void geneticAlgorithm(int generations, int populationSize) {
    vector<vector<int>> population;
    initializePopulation(population, populationSize);

    vector<int> fitnessValues(populationSize);
    int bestValue = 0;

    for (int gen = 0; gen <= generations; gen++) {

        for (int i = 0; i < populationSize; i++) {
            fitnessValues[i] = fitness(population[i]);
        }


        int currentBest = 0;
        for (int value : fitnessValues) {
            if (value > currentBest) {
                currentBest = value;
            }
        }
        bestValue = max(bestValue, currentBest);


        if (gen == 0 || gen == generations || gen % (generations / 8) == 0) {
            cout << bestValue << endl;
        }


        bool hasValidIndividual = false;
        for (int i = 0; i < populationSize; i++) {
            if (fitnessValues[i] > 0) {
                hasValidIndividual = true;
                break;
            }
        }
        if (!hasValidIndividual) {
            population.clear();
            initializePopulation(population, populationSize);
            continue;
        }


        vector<vector<int>> newPopulation;
        for (int i = 0; i < populationSize / 2; i++) {
            vector<int> parent1 = selectParent(population, fitnessValues);
            vector<int> parent2 = selectParent(population, fitnessValues);

            vector<int> child1 = crossover(parent1, parent2);
            vector<int> child2 = crossover(parent2, parent1);

            mutate(child1);
            mutate(child2);

            newPopulation.push_back(child1);
            newPopulation.push_back(child2);
        }
        population = newPopulation;
    }


    cout << "Best Value: " << bestValue << endl;
}

int main() {
    srand(time(0));

    cin >> m >> n;
    items.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> items[i].weight >> items[i].value;
    }

    int generations = 1000;
    int populationSize = 500;

    auto start = high_resolution_clock::now();
    cout << '\n';
    geneticAlgorithm(generations, populationSize);
    auto end = high_resolution_clock::now();

    double duration = duration_cast<chrono::duration<double>>(end - start).count();
    cerr << "Execution Time: " << fixed << duration << " seconds" << endl;

    return 0;
}

