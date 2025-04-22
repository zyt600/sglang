#include "hiradixcache.hpp"  // Include the header file

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For automatic vector/tuple conversions
#include <torch/extension.h>  // For PyTorch C++ extension bindings

#include <algorithm>
#include <limits>
#include <stack>      // For potential non-recursive helpers (if needed)
#include <stdexcept>  // Include for std::runtime_error
#include <string>
#include <unordered_map>
#include <vector>

// Make pybind namespace accessible
namespace py = pybind11;

// ============================================================================
// TreeNode Method Implementations
// ============================================================================

// --- Constructor ---
TreeNode::TreeNode(NodeId id, std::vector<int> k, TreeNode* p)
    : id(id), key(std::move(k)), parent(p) {}  // Correct initializer list

// ============================================================================
// HiRadixCache Method Implementations
// ============================================================================

// --- Constructor ---
HiRadixCache::HiRadixCache(NodeId root_node_id) {
  root_node_ =
      std::make_shared<TreeNode>(root_node_id, std::vector<int>{}, nullptr);
  node_id_map_[root_node_->id] = root_node_.get();
}

// --- Private Helper: Find Node by ID ---
TreeNode* HiRadixCache::find_node_by_id(NodeId node_id) const {
  auto it = node_id_map_.find(node_id);
  return (it != node_id_map_.end()) ? it->second : nullptr;
}

// --- Private Helper: Get Child Key ---
int HiRadixCache::get_child_key(const std::vector<int>& key) const {
  if (key.empty()) {
    throw std::runtime_error("Cannot get child key from empty key segment");
  }
  return key[0];  // Use first element as the map key
}

// --- Private Helper: Key Match ---
size_t HiRadixCache::key_match(const std::vector<int>& node_key,
                               const std::vector<int>& query_key) const {
  size_t len = std::min(node_key.size(), query_key.size());
  size_t i = 0;
  while (i < len && node_key[i] == query_key[i]) {
    i++;
  }
  return i;
}

// --- Public Method Implementations ---

void HiRadixCache::reset() {
  NodeId root_id = root_node_ ? root_node_->id : 0;
  node_id_map_.clear();
  root_node_.reset();  // Release old tree
  // Recreate root node
  root_node_ = std::make_shared<TreeNode>(root_id, std::vector<int>{}, nullptr);
  node_id_map_[root_node_->id] = root_node_.get();
}

bool HiRadixCache::backup_node(NodeId node_id) {
  TreeNode* node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  node->backuped = true;
  return true;
}

bool HiRadixCache::evict_node(NodeId node_id) {
  TreeNode* node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  node->evicted = true;
  return true;
}

bool HiRadixCache::load_node(NodeId node_id) {
  TreeNode* node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  node->evicted = false;
  return true;
}

bool HiRadixCache::delete_node(NodeId node_id) {
  TreeNode* node_to_remove = find_node_by_id(node_id);
  if (!node_to_remove || node_to_remove == root_node_.get()) {
    throw std::runtime_error(
        "Attempted to delete root node or null node using delete_node.");
  }

  // --- Added Check for Leaf Node ---
  if (!node_to_remove->children.empty()) {
    // If node has children, it's not a leaf. Throw an error.
    throw std::runtime_error(
        "Attempted to delete a non-leaf node (ID: " + std::to_string(node_id) +
        ") using delete_node." +
        std::to_string(node_to_remove->children.size()) + " children");
  }
  // --- End Added Check ---

  TreeNode* parent_node = node_to_remove->parent;
  int child_map_key = get_child_key(node_to_remove->key);

  auto it = parent_node->children.find(child_map_key);
  if (it != parent_node->children.end() && it->second.get() == node_to_remove) {
    node_id_map_.erase(node_to_remove->id);  // Remove from ID map
    // Erase from parent's children map (releases shared_ptr)
    parent_node->children.erase(it);
    return true;
  }
  throw std::runtime_error(
      "Node ID not found in parent's children map. Node ID: " +
      std::to_string(node_id));
}

bool HiRadixCache::insert_node(NodeId node_id, std::vector<int>& key_segment,
                               NodeId parent_id) {
  TreeNode* parent_node = find_node_by_id(parent_id);
  if (!parent_node) {
    throw std::runtime_error(
        "Parent node not found for insertion. Parent ID: " +
        std::to_string(parent_id));
  }
  if (node_id_map_.count(node_id)) {
    throw std::runtime_error("Node ID already exists in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  int child_map_key = get_child_key(key_segment);
  if (parent_node->children.count(child_map_key)) {
    throw std::runtime_error(
        "Child map key already exists in parent node's children, insert "
        "failed.");
  }

  // Create the new node
  auto new_node =
      std::make_shared<TreeNode>(node_id, std::move(key_segment), parent_node);
  parent_node->children[child_map_key] = new_node;
  node_id_map_[node_id] = new_node.get();
  return true;
}

bool HiRadixCache::split_node(NodeId original_node_id,
                              std::vector<int>& key_segment_back,
                              NodeId new_node_id,
                              std::vector<int>& key_segment_front) {
  TreeNode* original_node = find_node_by_id(original_node_id);
  if (!original_node) {
    throw std::runtime_error("Original node not found for split. Node ID: " +
                             std::to_string(original_node_id));
  }
  TreeNode* parent_node = original_node->parent;

  int map_key_back = get_child_key(key_segment_back);
  int map_key_front = get_child_key(key_segment_front);
  if (map_key_front != get_child_key(original_node->key)) {
    throw std::runtime_error(
        "Child map key mismatch, split failed. Expected: " +
        std::to_string(get_child_key(original_node->key)) +
        ", got: " + std::to_string(map_key_front));
  }

  auto new_node = std::make_shared<TreeNode>(
      new_node_id, std::move(key_segment_front), parent_node);
  new_node->evicted = original_node->evicted;
  new_node->backuped = original_node->backuped;
  node_id_map_[new_node_id] = new_node.get();

  new_node->children[map_key_back] = parent_node->children[map_key_front];
  parent_node->children[map_key_front] = new_node;

  original_node->key = std::move(key_segment_back);
  original_node->parent = new_node.get();  // Update parent of the original node

  return true;
}

std::tuple<int, int, int> HiRadixCache::match_prefix(
    const std::vector<int>& key) const {
  int hit_device_len = 0;
  int hit_host_len = 0;
  size_t total_matched_len = 0;

  auto current_node = root_node_.get();  // Use raw pointer for traversal
  std::vector<int> key_remaining = key;

  while (!key_remaining.empty()) {
    int child_map_key = get_child_key(key_remaining);
    auto it = current_node->children.find(child_map_key);

    if (it == current_node->children.end()) {
      break;  // No child matches the next part of the key
    }

    auto child = it->second.get();
    size_t prefix_len = key_match(child->key, key_remaining);

    total_matched_len += prefix_len;

    // Add the length of this fully matched node's key to appropriate counter
    if (!child->evicted) {
      hit_device_len += prefix_len;
    } else if (child->backuped) {  // Only count if evicted AND backuped
      hit_host_len += prefix_len;
    } else {
      throw std::runtime_error(
          "Node is evicted but not backuped. This should not happen.");
    }

    if (prefix_len < child->key.size()) {
      // Partial match on the child's edge key, stop here
      break;
    }

    // Consume the matched part of the key and move to the child
    key_remaining.erase(key_remaining.begin(),
                        key_remaining.begin() + prefix_len);
    current_node = child;
  }

  // Calculate the length that needs to be computed
  int to_compute_len =
      static_cast<int>(key.size()) - static_cast<int>(total_matched_len);
  // Ensure to_compute_len is not negative (can happen if key is shorter than
  // path)
  if (to_compute_len < 0) to_compute_len = 0;

  return std::make_tuple(hit_device_len, hit_host_len, to_compute_len);
}

// -----------------------------------------------------------------------------
// Helper function: build prefix key up to threshold integers.
// -----------------------------------------------------------------------------
static std::string buildPrefixKey(const std::vector<int>& request,
                                  int threshold) {
  int prefixLen = std::min(static_cast<int>(request.size()), threshold);

  std::string key;
  key.reserve(prefixLen * 4);  // rough guess to reduce reallocation
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
    const std::vector<std::vector<int>>& requests, int threshold = 100) {
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
    groups.emplace_back(kv.first, std::move(kv.second.indices),
                        kv.second.earliestIndex);
  }

  // 3. Sort the groups:
  //    (a) By size of the group (descending).
  //    (b) Tie-break by earliest index (ascending).
  std::sort(groups.begin(), groups.end(), [](auto& g1, auto& g2) {
    const auto& indices1 = std::get<1>(g1);
    const auto& indices2 = std::get<1>(g2);
    const size_t size1 = indices1.size();
    const size_t size2 = indices2.size();

    if (size1 != size2) {
      return size1 > size2;  // larger group first
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
// Pybind portion for exposing to Python.  We'll accept a Python list of lists
// of integers, along with an optional threshold, and return a list of lists of
// ints.
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> group_requests_py(
    std::vector<std::vector<int>> requests, int threshold = 100) {
  return groupRequests(requests, threshold);
}

// ============================================================================
// Pybind11 Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pybind11 bindings for HiRadixCache C++ implementation";

  // Expose the HiRadixCache class to Python
  py::class_<HiRadixCache>(m, "HiRadixCache_CPP")
      .def(py::init<NodeId>(), py::arg("root_node_id") = 0)  // Constructor
      .def("reset", &HiRadixCache::reset,
           py::call_guard<py::gil_scoped_release>(),
           "Resets the cache to an empty state with only the root node.")
      .def("backup_node", &HiRadixCache::backup_node, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Marks a node as backuped (returns true on success).")
      .def("evict_node", &HiRadixCache::evict_node, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Marks a node as evicted (returns true on success).")
      .def("load_node", &HiRadixCache::load_node, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Marks a node as not evicted (returns true on success).")
      .def("delete_node", &HiRadixCache::delete_node, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Deletes a leaf node by ID (throws error if not leaf, returns true "
           "on success).")  // Updated docstring
      .def("insert_node", &HiRadixCache::insert_node, py::arg("node_id"),
           py::arg("key_segment"), py::arg("parent_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Inserts a new node with given ID and key segment as child of "
           "parent_id (returns true on success).")
      .def("split_node", &HiRadixCache::split_node, py::arg("node_id"),
           py::arg("key_segment_back"), py::arg("new_node_id"),
           py::arg("key_segment_front"),
           py::call_guard<py::gil_scoped_release>(),
           "Updates the key segment of a node (returns true on success). Use "
           "with caution.")
      .def("match_prefix", &HiRadixCache::match_prefix, py::arg("key"),
           py::call_guard<py::gil_scoped_release>(),
           "Matches key prefix, returns tuple (hit_device_len, hit_host_len, "
           "to_compute_len).");

  m.def("group_requests", &group_requests_py,
        "Group requests by prefix up to 'threshold' length (default 100)",
        py::arg("requests"), py::arg("threshold") = 100);
}
