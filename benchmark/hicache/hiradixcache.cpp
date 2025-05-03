#include "hiradixcache.hpp" // Include the header file

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For automatic vector/tuple conversions
#include <torch/extension.h> // For PyTorch C++ extension bindings

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stack>     // For potential non-recursive helpers (if needed)
#include <stdexcept> // Include for std::runtime_error
#include <string>
#include <unordered_map>
#include <vector>

// Make pybind namespace accessible
namespace py = pybind11;
const int CACHE_THRESHOLD = 100;

// ============================================================================
// TreeNode Method Implementations
// ============================================================================

// --- Constructor ---
TreeNode::TreeNode(NodeId id, std::vector<int> k, TreeNode *p)
    : id(id), key(std::move(k)), parent(p) {} // Correct initializer list

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
TreeNode *HiRadixCache::find_node_by_id(NodeId node_id) const {
  auto it = node_id_map_.find(node_id);
  return (it != node_id_map_.end()) ? it->second : nullptr;
}

// --- Private Helper: Get Child Key ---
int HiRadixCache::get_child_key(const std::vector<int> &key) const {
  if (key.empty()) {
    throw std::runtime_error("Cannot get child key from empty key segment");
  }
  return key[0]; // Use first element as the map key
}

// --- Private Helper: Key Match ---
size_t HiRadixCache::key_match(const std::vector<int> &node_key,
                               const std::vector<int> &query_key) const {
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
  root_node_.reset(); // Release old tree
  // Recreate root node
  root_node_ = std::make_shared<TreeNode>(root_id, std::vector<int>{}, nullptr);
  node_id_map_[root_node_->id] = root_node_.get();
}

bool HiRadixCache::backup_node(NodeId node_id) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  node->backuped = true;
  return true;
}

bool HiRadixCache::evict_node(NodeId node_id) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  if (node->ref_count > 0) {
    throw std::runtime_error(
        "Cannot evict node with non-zero reference count. Node ID: " +
        std::to_string(node_id));
  }
  node->evicted = true;
  return true;
}

bool HiRadixCache::load_node(NodeId node_id) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  node->evicted = false;
  return true;
}

bool HiRadixCache::delete_node(NodeId node_id) {
  TreeNode *node_to_remove = find_node_by_id(node_id);
  if (!node_to_remove || node_to_remove == root_node_.get()) {
    throw std::runtime_error(
        "Attempted to delete root node or null node using delete_node.");
  }

  if (node_to_remove->ref_count > 0) {
    throw std::runtime_error(
        "Cannot delete node with non-zero reference count. Node ID: " +
        std::to_string(node_id));
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

  TreeNode *parent_node = node_to_remove->parent;
  int child_map_key = get_child_key(node_to_remove->key);

  auto it = parent_node->children.find(child_map_key);
  if (it != parent_node->children.end() && it->second.get() == node_to_remove) {
    node_id_map_.erase(node_to_remove->id); // Remove from ID map
    // Erase from parent's children map (releases shared_ptr)
    parent_node->children.erase(it);
    return true;
  }
  throw std::runtime_error(
      "Node ID not found in parent's children map. Node ID: " +
      std::to_string(node_id));
}

void HiRadixCache::insert_pending_request(NodeId node_id,
                                          std::vector<int> &request) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  if (request.size() > CACHE_THRESHOLD) {
    node->pending_requests.push_back(std::move(request));
  }
}

bool HiRadixCache::insert_node(NodeId node_id, std::vector<int> &key_segment,
                               NodeId parent_id) {
  TreeNode *parent_node = find_node_by_id(parent_id);
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

  std::vector<std::vector<int>> new_pending_requests;
  for (int idx = 0; idx < parent_node->pending_requests.size(); ++idx) {
    // check and erase pending requests that match the key segment
    auto &pending_r = parent_node->pending_requests[idx];
    int match_len = key_match(pending_r, key_segment);
    if (match_len == key_segment.size()) {
      pending_r.erase(pending_r.begin(), pending_r.begin() + match_len);
      new_pending_requests.push_back(std::move(pending_r));
      parent_node->pending_requests.erase(
          parent_node->pending_requests.begin() + idx);
      idx--;
    } else if (pending_r.size() - match_len < CACHE_THRESHOLD) {
      // If the pending request mostly matches the key segment, remove it
      parent_node->pending_requests.erase(
          parent_node->pending_requests.begin() + idx);
      idx--;
    } else {
      // todo, partial match, need to handle
    }
  }

  // Create the new node
  auto new_node =
      std::make_shared<TreeNode>(node_id, std::move(key_segment), parent_node);
  parent_node->children[child_map_key] = new_node;
  node_id_map_[node_id] = new_node.get();
  new_node->pending_requests = std::move(new_pending_requests);
  return true;
}

bool HiRadixCache::split_node(NodeId original_node_id,
                              std::vector<int> &key_segment_back,
                              NodeId new_node_id,
                              std::vector<int> &key_segment_front) {
  TreeNode *original_node = find_node_by_id(original_node_id);
  if (!original_node) {
    throw std::runtime_error("Original node not found for split. Node ID: " +
                             std::to_string(original_node_id));
  }
  TreeNode *parent_node = original_node->parent;

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
  original_node->parent = new_node.get(); // Update parent of the original node

  return true;
}

void HiRadixCache::inc_lock_ref(NodeId node_id) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  while (node != nullptr) {
    node->ref_count++;
    node = node->parent;
  }
}

void HiRadixCache::dec_lock_ref(NodeId node_id) {
  TreeNode *node = find_node_by_id(node_id);
  if (!node) {
    throw std::runtime_error("Node ID not found in the cache. Node ID: " +
                             std::to_string(node_id));
  }
  while (node != nullptr) {
    node->ref_count--;
    node = node->parent;
  }
}

std::tuple<int, int, int, int>
HiRadixCache::match_prefix(const std::vector<int> &key,
                           bool lock_only = true) const {
  int hit_device_len = 0;
  int hit_host_len = 0;
  int pending_hit_len = 0;
  size_t total_matched_len = 0;

  auto current_node = root_node_.get(); // Use raw pointer for traversal
  std::vector<int> key_remaining = key;

  while (!key_remaining.empty()) {
    int child_map_key = get_child_key(key_remaining);
    auto it = current_node->children.find(child_map_key);

    if (it == current_node->children.end()) {
      // matching complete, check if any pending requests match
      if (current_node->pending_requests.size() > 0) {
        for (const auto &pending_request : current_node->pending_requests) {
          int match_len = key_match(pending_request, key_remaining);
          pending_hit_len = std::max(pending_hit_len, match_len);
        }
      }

      break;
    }

    auto child = it->second.get();
    size_t prefix_len = key_match(child->key, key_remaining);

    total_matched_len += prefix_len;

    if ((!child->evicted) && ((!lock_only) || (child->ref_count > 0))) {
      // if lock_only is true, only count the prefix if the node is locked
      hit_device_len += prefix_len;
    } else {
      hit_host_len += prefix_len;
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

  return std::make_tuple(hit_device_len, hit_host_len, to_compute_len,
                         pending_hit_len);
}

struct Request {
  int index;
  int length;
  int hit_device_len;
  int hit_host_len;
  int to_compute_len;
  int pending_hit_len;
  std::string prefix_key;
};

static std::string buildPrefixKey(const std::vector<int> &request,
                                  int threshold) {
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

struct GroupInfo {
  std::vector<int> indices;
  int earliestIndex;
};

std::vector<int> scheduling_lpm(const std::vector<Request> &request_list) {
  std::vector<int> reordered_indices(request_list.size());
  std::iota(reordered_indices.begin(), reordered_indices.end(), 0);

  std::sort(reordered_indices.begin(), reordered_indices.end(),
            [&request_list](int index1, int index2) {
              const Request &r1 = request_list[index1];
              const Request &r2 = request_list[index2];
              if (std::abs(r1.hit_device_len - r2.hit_device_len) >
                  CACHE_THRESHOLD) {
                return r1.hit_device_len > r2.hit_device_len;
              }
              return index1 < index2;
            });
  return reordered_indices;
}

std::vector<int> scheduling_glpm(const std::vector<Request> &request_list) {
  std::vector<int> reordered_indices(request_list.size());
  std::iota(reordered_indices.begin(), reordered_indices.end(), 0);

  std::sort(reordered_indices.begin(), reordered_indices.end(),
            [&request_list](int index1, int index2) {
              const Request &r1 = request_list[index1];
              const Request &r2 = request_list[index2];
              if (r1.hit_device_len + r1.hit_host_len !=
                  r2.hit_device_len + r2.hit_host_len) {
                return r1.hit_device_len + r1.hit_host_len >
                       r2.hit_device_len +
                           r2.hit_host_len; // longer global match first
              }
              return index1 < index2;
            });
  return reordered_indices;
}

std::vector<int>
scheduling_grouping(const std::vector<Request> &request_list,
                    std::unordered_map<std::string, GroupInfo> &prefixMap) {
  std::vector<int> reordered_indices;
  reordered_indices.reserve(request_list.size());
  using MapConstIterator =
      std::unordered_map<std::string, GroupInfo>::const_iterator;
  std::vector<MapConstIterator> group_iters;
  group_iters.reserve(prefixMap.size());

  for (auto it = prefixMap.begin(); it != prefixMap.end(); ++it) {
    if (!it->second.indices.empty()) {
      group_iters.push_back(it);
    }
  }

  std::sort(group_iters.begin(), group_iters.end(),
            [request_list](MapConstIterator it1, MapConstIterator it2) {
              // Access GroupInfo directly via the iterator
              const GroupInfo &g1_info = it1->second;
              const GroupInfo &g2_info = it2->second;
              const size_t size1 = g1_info.indices.size();
              const size_t size2 = g2_info.indices.size();

              if (size1 != size2) {
                return size1 > size2; // larger group first
              }
              if (size1 == 1) {
                // sort by hit_device_len
                const Request &r1 = request_list[g1_info.earliestIndex];
                const Request &r2 = request_list[g2_info.earliestIndex];
                if (std::abs(r1.hit_device_len - r2.hit_device_len) >
                    CACHE_THRESHOLD) {
                  return r1.hit_device_len > r2.hit_device_len;
                }
              }
              // tie-break: earliest index ascending
              return g1_info.earliestIndex < g2_info.earliestIndex;
            });

  for (MapConstIterator it : group_iters) {
    const GroupInfo &group_info = it->second;
    const std::vector<int> &indices = group_info.indices;
    const int earliest_req_idx = group_info.earliestIndex;
    const Request &leading_req = request_list[earliest_req_idx];

    // If the leading request has a pending hit, postpone the group
    if (leading_req.pending_hit_len > CACHE_THRESHOLD) {
      continue;
    }

    if (indices.size() > 1) {
      if (leading_req.hit_device_len + leading_req.hit_host_len >
          CACHE_THRESHOLD) {
        reordered_indices.insert(reordered_indices.end(), indices.begin(),
                                 indices.end());
        continue;
      }
    }
    reordered_indices.push_back(leading_req.index);
  }

  return reordered_indices;
}

std::vector<int>
scheduling_balance(std::vector<Request> &request_list,
                   std::unordered_map<std::string, GroupInfo> &prefixMap) {
  std::vector<int> reordered_indices;
  reordered_indices.reserve(request_list.size());
  std::vector<GroupInfo *> group_ptrs;
  group_ptrs.reserve(prefixMap.size());

  for (auto &[_, group] : prefixMap) {
    // if (group.indices.size() > 1) {
    //   Request& lead = request_list[group.earliestIndex];
    //   const std::size_t cached_len = lead.length - lead.to_compute_len;
    //   lead.hit_device_len = group.indices.size() * cached_len;
    // }
    group_ptrs.push_back(&group);
  }

  std::sort(group_ptrs.begin(), group_ptrs.end(),
            [&request_list](const GroupInfo *g1_ptr, const GroupInfo *g2_ptr) {
              const size_t size1 = g1_ptr->indices.size();
              const size_t size2 = g2_ptr->indices.size();
              if (size1 != size2) {
                return size1 > size2; // larger group first
              }

              const Request &r1 = request_list[g1_ptr->earliestIndex];
              const Request &r2 = request_list[g2_ptr->earliestIndex];

              const int hit_diff = r1.hit_device_len - r2.hit_device_len;
              if (std::abs(hit_diff) > CACHE_THRESHOLD)
                return hit_diff > 0;

              return r1.hit_host_len / static_cast<double>(r1.to_compute_len) >
                     r2.hit_host_len / static_cast<double>(r2.to_compute_len);

              const int cache_diff = (r1.length - r1.to_compute_len) -
                                     (r2.length - r2.to_compute_len);
              if (std::abs(cache_diff) > CACHE_THRESHOLD)
                return cache_diff > 0;

              // tie-break: earliest index ascending
              return g1_ptr->earliestIndex < g2_ptr->earliestIndex;
            });

  double sum_load = 1;
  double sum_compute = 1;
  const double load_compute_ratio = 100;

  int group_idx = 0;
  while (group_idx < group_ptrs.size()) {
    const GroupInfo *group_info = group_ptrs[group_idx];
    Request &leading_req = request_list[group_info->earliestIndex];

    // If the leading request has a pending hit, postpone the group
    if (leading_req.pending_hit_len > CACHE_THRESHOLD) {
      group_idx += 1;
      continue;
    }

    if (((group_info->indices.size() > 1) &&
         (leading_req.hit_device_len + leading_req.hit_host_len >
          CACHE_THRESHOLD)) ||
        (leading_req.hit_device_len > CACHE_THRESHOLD)) {
      for (int idx : group_info->indices) {
        reordered_indices.push_back(idx);
        sum_compute += request_list[idx].to_compute_len;
      }
      sum_load += leading_req.hit_host_len;
      group_idx += 1;
      continue;
    }

    if (sum_load / sum_compute > load_compute_ratio) {
      // If the load is too high, pick from the end
      Request &comp_req = request_list[group_ptrs.back()->earliestIndex];
      if (comp_req.index != leading_req.index) {
        group_ptrs.pop_back();
        if (comp_req.pending_hit_len > CACHE_THRESHOLD) {
          continue;
        }
        reordered_indices.push_back(comp_req.index);
        sum_load += comp_req.hit_host_len;
        sum_compute += comp_req.to_compute_len;
        continue;
      }
    }

    reordered_indices.push_back(leading_req.index);
    sum_load += leading_req.hit_host_len;
    sum_compute += leading_req.to_compute_len;
    group_idx += 1;
  }
  return reordered_indices;
}

std::vector<int>
HiRadixCache::scheduling(std::vector<std::vector<int>> requests) {
  std::vector<Request> request_list;
  request_list.reserve(requests.size());

  std::unordered_map<std::string, GroupInfo> prefixMap;
  std::vector<std::reference_wrapper<Request>> filtered_requests;

  for (int i = 0; i < static_cast<int>(requests.size()); ++i) {
    auto [hit_device_len, hit_host_len, to_compute_len, pending_hit_len] =
        match_prefix(requests[i]);
    std::string prefix_key = buildPrefixKey(requests[i], CACHE_THRESHOLD);
    request_list.push_back({i, static_cast<int>(requests[i].size()),
                            hit_device_len, hit_host_len, to_compute_len,
                            pending_hit_len, prefix_key});

    auto &groupInfo = prefixMap[prefix_key];
    groupInfo.indices.push_back(i);
    // If this is the first insertion, record the earliest index
    if (groupInfo.indices.size() == 1) {
      groupInfo.earliestIndex = i;
      filtered_requests.push_back(std::ref(request_list.back()));
    }
  }

  // return scheduling_lpm(request_list);
  // return scheduling_glpm(request_list);
  return scheduling_grouping(request_list, prefixMap);
  // return scheduling_balance(request_list, prefixMap);
}

// -----------------------------------------------------------------------------
// Main grouping function
// -----------------------------------------------------------------------------
std::vector<std::vector<int>>
groupRequests(const std::vector<std::vector<int>> &requests,
              int threshold = 100) {
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

    auto &groupInfo = prefixMap[key];
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

  for (auto &kv : prefixMap) {
    groups.emplace_back(kv.first, std::move(kv.second.indices),
                        kv.second.earliestIndex);
  }

  // 3. Sort the groups:
  //    (a) By size of the group (descending).
  //    (b) Tie-break by earliest index (ascending).
  std::sort(groups.begin(), groups.end(), [](auto &g1, auto &g2) {
    const auto &indices1 = std::get<1>(g1);
    const auto &indices2 = std::get<1>(g2);
    const size_t size1 = indices1.size();
    const size_t size2 = indices2.size();

    if (size1 != size2) {
      return size1 > size2; // larger group first
    }
    // tie-break: earliest index ascending
    return std::get<2>(g1) < std::get<2>(g2);
  });

  // 4. Build the final result: each group is just the vector of indices
  std::vector<std::vector<int>> result;
  result.reserve(groups.size());
  for (auto &group : groups) {
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
std::vector<std::vector<int>>
group_requests_py(std::vector<std::vector<int>> requests, int threshold = 100) {
  return groupRequests(requests, threshold);
}

// ============================================================================
// Pybind11 Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pybind11 bindings for HiRadixCache C++ implementation";

  // Expose the HiRadixCache class to Python
  py::class_<HiRadixCache>(m, "HiRadixCache_CPP")
      .def(py::init<NodeId>(), py::arg("root_node_id") = 0) // Constructor
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
           "on success).") // Updated docstring
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
      .def("inc_lock_ref", &HiRadixCache::inc_lock_ref, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Increments the lock reference count for a node.")
      .def("dec_lock_ref", &HiRadixCache::dec_lock_ref, py::arg("node_id"),
           py::call_guard<py::gil_scoped_release>(),
           "Decrements the lock reference count for a node.")
      .def("scheduling", &HiRadixCache::scheduling, py::arg("requests"),
           py::call_guard<py::gil_scoped_release>(),
           "Schedules requests based on prefix matching, returns reordered "
           "indices.")
      .def("insert_pending_request", &HiRadixCache::insert_pending_request,
           py::arg("node_id"), py::arg("request"),
           py::call_guard<py::gil_scoped_release>(),
           "Inserts pending requests into the node.");

  m.def("group_requests", &group_requests_py, py::arg("requests"),
        py::arg("threshold") = 100, py::call_guard<py::gil_scoped_release>(),
        "Group requests by prefix up to 'threshold' length (default 100)");
}
