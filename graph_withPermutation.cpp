#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <fstream>

using namespace std;

// Helper function: convert vector<int> to string
string vecToStr(const vector<int>& v) {
    string s;
    for (int x : v) s += to_string(x);
    return s;
}

// Generate all permutations of size n
vector<string> generatePermutations(int n) {
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;

    vector<string> result;
    do {
        result.push_back(vecToStr(perm));
    } while (next_permutation(perm.begin(), perm.end()));
    return result;
}

// Check if two permutations differ by one adjacent swap
bool isAdjacentSwap(const string& a, const string& b) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        string temp = a;
        swap(temp[i], temp[i + 1]);
        if (temp == b) return true;
    }
    return false;
}

int main() {
    int n;
    cout << "Enter n for Bn (bubble-sort network): ";
    cin >> n;

    vector<string> nodes = generatePermutations(n);

    // Map node string to index
    unordered_map<string, int> nodeIndex;
    for (int i = 0; i < nodes.size(); i++) {
        nodeIndex[nodes[i]] = i;
    }

    string filename = "Bn_edges_n" + to_string(n) + ".txt";
    ofstream outfile(filename);

    if (!outfile.is_open()) {
        cerr << "Failed to open file for writing.\n";
        return 1;
    }

    for (int i = 0; i < nodes.size(); i++) {
        for (int j = i + 1; j < nodes.size(); j++) {
            if (isAdjacentSwap(nodes[i], nodes[j])) {
                outfile << i << " " << j << "  // " << nodes[i] << " - " << nodes[j] << "\n";
            }
        }
    }

    outfile.close();
    cout << "Edges written to " << filename << "\n";

    return 0;
}
