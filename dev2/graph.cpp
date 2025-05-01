#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <fstream>

using namespace std;

// Convert vector to string (unique identifier for a permutation)
string vecToStr(const vector<int>& v) {
    string s;
    for (int num : v) s += to_string(num) + ",";
    return s;
}

// Generate neighbors by adjacent swaps
vector<string> generateNeighbors(const vector<int>& perm) {
    vector<string> neighbors;
    for (int i = 0; i < perm.size() - 1; i++) {
        vector<int> temp = perm;
        swap(temp[i], temp[i + 1]);
        neighbors.push_back(vecToStr(temp));
    }
    return neighbors;
}

int main() {
    int n;
    cout << "Enter n for Bn (bubble-sort network): ";
    cin >> n;

    string filename = "Bn_edges_n" + to_string(n) + ".txt";
    ofstream outfile(filename);

    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;

    unordered_map<string, int> idMap; // Map permutation string to unique node ID
    int idCounter = 0;

    do {
        string current = vecToStr(perm);
        if (idMap.find(current) == idMap.end()) {
            idMap[current] = idCounter++;
        }
        int u = idMap[current];

        vector<string> neighbors = generateNeighbors(perm);
        for (const string& neighbor : neighbors) {
            if (idMap.find(neighbor) == idMap.end()) {
                idMap[neighbor] = idCounter++;
            }
            int v = idMap[neighbor];
            if (u < v) // Avoid duplicate edges
                outfile << u << " " << v << "\n";
        }

    } while (next_permutation(perm.begin(), perm.end()));

    outfile.close();
    cout << "Finished writing edges to " << filename << " (" << idCounter << " nodes).\n";

    return 0;
}
