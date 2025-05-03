#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using NodeId = int;
struct TreeNode;

// ============================================================================
// Data Structures
// ============================================================================

struct TreeNode : std::enable_shared_from_this<TreeNode> {
  NodeId id;
  std::unordered_map<int, std::shared_ptr<TreeNode>> children;
  TreeNode *parent = nullptr;

  std::vector<int> key;

  bool evicted = false;
  bool backuped = false;
  int ref_count = 0;
  std::vector<std::vector<int>> pending_requests;

  TreeNode(NodeId id, std::vector<int> k, TreeNode *p = nullptr);
  ~TreeNode() = default;
};

// ============================================================================
// Radix Tree Cache Class Definition
// ============================================================================

class HiRadixCache {
public:
  // --- Constructor & Destructor ---
  HiRadixCache(NodeId root_node_id = 0);
  ~HiRadixCache() = default; // Default destructor is okay with shared_ptr

  // --- Public Methods (Commands from Python) ---
  void reset();
  bool backup_node(NodeId node_id);
  bool evict_node(NodeId node_id);
  bool load_node(NodeId node_id);
  bool delete_node(NodeId node_id);
  bool insert_node(NodeId node_id, std::vector<int> &new_key_segment,
                   NodeId parent_id);
  bool split_node(NodeId node_id, std::vector<int> &key_segment,
                  NodeId new_node_id, std::vector<int> &new_key_segment);

  void inc_lock_ref(NodeId node_id);
  void dec_lock_ref(NodeId node_id);
  void insert_pending_request(NodeId node_id, std::vector<int> &request);

  // --- Public Query Method ---
  std::vector<int> scheduling(std::vector<std::vector<int>> requests);
  std::tuple<int, int, int, int> match_prefix(const std::vector<int> &key,
                                              bool lock_only) const;

private:
  // --- Private Data Members ---
  std::shared_ptr<TreeNode> root_node_;
  // Map for efficient Node ID -> Node Pointer lookup
  std::unordered_map<NodeId, TreeNode *> node_id_map_;

  // --- Private Helper Methods ---
  TreeNode *find_node_by_id(NodeId node_id) const;
  int get_child_key(const std::vector<int> &key) const;
  size_t key_match(const std::vector<int> &node_key,
                   const std::vector<int> &query_key) const;
};
