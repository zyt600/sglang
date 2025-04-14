#include <torch/extension.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>

// -----------------------------------------------------------------------------
// Helper function: build prefix key up to threshold integers.
// -----------------------------------------------------------------------------
static std::string buildPrefixKey(const std::vector<int>& request, int threshold) {
    int prefixLen = std::min(static_cast<int>(request.size()), threshold);

    std::string key;
    key.reserve(prefixLen * 4); // rough guess to reduce reallocation
    for (int i = 0; i < prefixLen; ++i) {
        key += std::to_string(request[i]);
        if (i < prefixLen - 1) {
            key += ",";
        }
    }
    return key;
}

// -----------------------------------------------------------------------------
// Main grouping function
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> groupRequests(
    const std::vector<std::vector<int>>& requests,
    int threshold = 100
) {
    // Data structure to hold information about each prefix group
    struct GroupInfo {
        std::vector<int> indices;
        int earliestIndex;
    };

    // Map: prefix_key -> GroupInfo (indices, earliest index)
    std::unordered_map<std::string, GroupInfo> prefixMap;
    prefixMap.reserve(requests.size());

    // 1. Build groups by prefix
    for (int i = 0; i < static_cast<int>(requests.size()); ++i) {
        std::string key = buildPrefixKey(requests[i], threshold);

        auto& groupInfo = prefixMap[key];
        groupInfo.indices.push_back(i);

        // If this is the first insertion, record the earliest index
        if (groupInfo.indices.size() == 1) {
            groupInfo.earliestIndex = i;
        }
    }

    // 2. Collect groups in a vector for sorting
    //    We'll store: (prefix_key, vector_of_indices, earliest_index).
    std::vector<std::tuple<std::string, std::vector<int>, int>> groups;
    groups.reserve(prefixMap.size());

    for (auto& kv : prefixMap) {
        groups.emplace_back(
            kv.first,
            std::move(kv.second.indices),
            kv.second.earliestIndex
        );
    }

    // 3. Sort the groups:
    //    (a) By size of the group (descending).
    //    (b) Tie-break by earliest index (ascending).
    std::sort(groups.begin(), groups.end(),
              [](auto& g1, auto& g2) {
                  const auto& indices1 = std::get<1>(g1);
                  const auto& indices2 = std::get<1>(g2);
                  const size_t size1   = indices1.size();
                  const size_t size2   = indices2.size();

                  if (size1 != size2) {
                      return size1 > size2; // larger group first
                  }
                  // tie-break: earliest index ascending
                  return std::get<2>(g1) < std::get<2>(g2);
              });

    // 4. Build the final result: each group is just the vector of indices
    std::vector<std::vector<int>> result;
    result.reserve(groups.size());
    for (auto& group : groups) {
        // std::get<1>(group) is the vector of indices
        result.push_back(std::move(std::get<1>(group)));
    }

    return result;
}

// -----------------------------------------------------------------------------
// Pybind portion for exposing to Python.  We'll accept a Python list of lists of
// integers, along with an optional threshold, and return a list of lists of ints.
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> group_requests_py(
    std::vector<std::vector<int>> requests,
    int threshold = 100
) {
    return groupRequests(requests, threshold);
}

// We create the Pybind module here.  The special macro name is TORCH_EXTENSION_NAME.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_requests", &group_requests_py,
          "Group requests by prefix up to 'threshold' length (default 100)",
          py::arg("requests"),
          py::arg("threshold") = 100);
}
