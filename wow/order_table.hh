#pragma once

#include <vector>
#include <mutex>
#include <random>
#include <unordered_set>
#include "ygg/ygg.hpp"
#include "utils.hh"
#include "disk.hh"

namespace wowlib {

template <typename att_t>
struct att_label_t{
  const att_t& att_;
  label_t label_;
  att_label_t(const att_t &att, label_t label) : att_(att), label_(label) {}
  att_label_t(const att_label_t &other) : att_(other.att_), label_(other.label_) {}
  bool operator<(const att_label_t &other) const { return this->att_ < other.att_ || (this->att_ == other.att_ && this->label_ < other.label_); }
  bool operator>(const att_label_t &other) const { return this->att_ > other.att_ || (this->att_ == other.att_ && this->label_ > other.label_); }
  bool operator==(const att_label_t &other) const { return this->att_ == other.att_ && this->label_ == other.label_; }
  bool operator!=(const att_label_t &other) const { return !(*this == other); }
  bool operator<=(const att_label_t &other) const { return *this < other || *this == other; }
  bool operator>=(const att_label_t &other) const { return *this > other || *this == other; }
};

template <typename att_t>
class OrderTable
{

public:
  OrderTable() = default;

  ~OrderTable() = default;

  virtual void InsertAttInid(const att_label_t<att_t>& att_label, tableint id) = 0;

  virtual auto GetWindowedFilterAndEntries(
      const att_label_t<att_t> &cur_att_label, int half_window_size, std::vector<tableint> &entry_points) -> wow_range<att_label_t<att_t>> = 0;

  virtual auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates,
      const std::vector<att_label_t<att_t>> &cand_att_label_vec, const att_label_t<att_t> &center_att_label, int half_window_size)
      -> std::vector<dist_id_pair> = 0;

  virtual auto GetRangeCardinality(const att_label_t<att_t> &l, const att_label_t<att_t> &u, std::vector<tableint> &OUT_eps) -> size_t = 0;

  virtual void Serialize(std::ostream &os) { std::cout << "Serialize is not implemented" << std::endl; };

  virtual void Deserialize(std::istream &is) { std::cout << "Deserialize is not implemented" << std::endl; };

protected:
  std::mutex lock_{};
  // std::unordered_set<att_t> unique_lookup_;
};

template <typename att_t>
class WBTreeOrderTable : public OrderTable<att_t>
{
  // multiple means
  using MyTreeOptions = ygg::TreeOptions<ygg::TreeFlags::WBT_SINGLE_PASS, ygg::TreeFlags::WBT_DELTA_NUMERATOR<3>,
      ygg::TreeFlags::WBT_DELTA_DENOMINATOR<1>, ygg::TreeFlags::WBT_GAMMA_NUMERATOR<2>,
      ygg::TreeFlags::WBT_GAMMA_DENOMINATOR<1>>;

  class WBNode : public ygg::WBTreeNodeBase<WBNode, MyTreeOptions>
  {
  public:
    WBNode() = default;
    WBNode(const att_label_t<att_t> &att_label, tableint id) : att_label_(att_label), id_(id) {}
    ~WBNode() = default;

    WBNode(const WBNode &other) = default;

    bool operator<(const WBNode &other) const { return this->att_label_ < other.att_label_; }

  public:
    const att_label_t<att_t> att_label_;
    tableint id_;
  };

  using MyTree = ygg::WBTree<WBNode, ygg::WBDefaultNodeTraits, MyTreeOptions>;

public:
  WBTreeOrderTable() = delete;

  explicit WBTreeOrderTable(size_t max_N) : max_N_(max_N) { node_store_ = new WBNode *[max_N]; }

  virtual ~WBTreeOrderTable()
  {
    for (int i = 0; i < tree_.size(); ++i) {
      delete node_store_[i];
    }
    delete[] node_store_;
  }

  void InsertAttInid(const att_label_t<att_t>& att_label, tableint id) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    auto node                 = new WBNode(att_label, id);
    node_store_[tree_.size()] = node;
    tree_.insert(*node);
  }

  auto GetWindowedFilterAndEntries(
      const att_label_t<att_t> &cur_att_label, int half_window_size, std::vector<tableint> &entry_points) -> wow_range<att_label_t<att_t>> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // if pos_l == 0 and pos_u == order_.size() - 1, the filter is the whole range just return
    if (2 * half_window_size >= tree_.size()) {
      entry_points.emplace_back(tree_.begin()->id_);
      return {tree_.begin()->att_label_, tree_.rbegin()->att_label_};
    }
    auto    it            = tree_.lower_bound(WBNode(cur_att_label, -1));
    WBNode *cur_node      = it == tree_.end() ? &*tree_.rbegin() : &*it;
    auto    boundary_node = GetWindowRangeLabel(cur_node, half_window_size);

    entry_points.emplace_back(boundary_node.l_->id_);
    if (boundary_node.l_->id_ != boundary_node.u_->id_) {
      entry_points.emplace_back(boundary_node.u_->id_);
    }
    return {boundary_node.l_->att_label_, boundary_node.u_->att_label_};
  }

  auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates,
      const std::vector<att_label_t<att_t>> &cand_att_label_vec, const att_label_t<att_t> &center_att_label, int half_window_size)
      -> std::vector<dist_id_pair> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    std::vector<dist_id_pair>   in_window_ids;
    // get the out of window id
    if (2 * half_window_size >= tree_.size()) {
      for (auto &c : candidates) {
        in_window_ids.emplace_back(c);
      }
      return in_window_ids;
    }
    auto it = tree_.find(WBNode(center_att_label, -1));
    if (it == tree_.end()) {
      throw std::runtime_error("Current node not found");
    }
    auto boundary_node = GetWindowRangeLabel(&*it, half_window_size);
    for (int i = 0; i < candidates.size(); ++i) {
      auto &c     = candidates[i];
      auto &c_att_label = cand_att_label_vec[i];
      auto  &l_att_label = boundary_node.l_->att_label_;
      auto  &u_att_label = boundary_node.u_->att_label_;
      if (c_att_label >= l_att_label && c_att_label <= u_att_label) {
        in_window_ids.emplace_back(c);
      }
    }
    return in_window_ids;
  }

  auto GetNodeIndex(WBNode *root, const att_label_t<att_t> &t_att_label) -> size_t
  {
    WBNode *cur = root;
    if (cur == nullptr)
      return 0;
    size_t index = 0;
    while (cur) {
      if (t_att_label < cur->att_label_) {
        cur = cur->get_left();
      } else if (t_att_label == cur->att_label_) {
        size_t left_size = cur->get_left() ? cur->get_left()->_wbt_size - 1 : 0;
        return index + left_size;
      } else {
        size_t left_size = cur->get_left() ? cur->get_left()->_wbt_size - 1 : 0;
        index += left_size + 1;
        cur = cur->get_right();
      }
    }
    throw std::runtime_error("Current node not found");
  }

  auto FindUpperBound(WBNode *root, const att_label_t<att_t> &t_att_label) -> WBNode *
  {
    WBNode *candidate = nullptr;
    WBNode *cur       = root;

    while (cur) {
      if (cur->att_label_ >= t_att_label) {
        candidate = cur;              // possible answer
        cur       = cur->get_left();  // find smaller answer
      } else {
        cur = cur->get_right();
      }
    }
    if (!candidate) {
      throw std::runtime_error("Target upper bound not exist, query range is likely empty");
    }

    return candidate;  // first node >= target
  }

  auto FindLowerBound(WBNode *root, const att_label_t<att_t> &t_att_label) -> WBNode *
  {
    WBNode *candidate = nullptr;
    WBNode *cur       = root;

    while (cur) {
      if (cur->att_label_ <= t_att_label) {
        candidate = cur;               // possible answer
        cur       = cur->get_right();  // find larger answer
      } else {
        cur = cur->get_left();
      }
    }
    if (!candidate) {
      throw std::runtime_error("Target lower bound not exist, query range is likely empty");
    }
    return candidate;  // first node <= target
  }

  auto GetRangeCardinality(const att_label_t<att_t> &l, const att_label_t<att_t> &u, std::vector<tableint> &OUT_eps) -> size_t override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // TODO: merge find bound and getnode index functions
    WBNode *root   = tree_.get_root();
    WBNode *node_l = FindUpperBound(root, l);
    WBNode *node_u = FindLowerBound(root, u);
    size_t  i      = GetNodeIndex(root, node_l->att_label_);
    size_t  j      = GetNodeIndex(root, node_u->att_label_);
    OUT_eps.emplace_back(node_l->id_);
    if (node_l != node_u) {
      OUT_eps.emplace_back(node_u->id_);
    }
    return j - i + 1;
  }

private:
  auto GetKthSmallestNode(WBNode *root, int k) -> WBNode *
  {
    if (root == nullptr || k <= 0 || k > root->_wbt_size - 1) {
      return nullptr;
    }

    WBNode *current = root;
    while (current != nullptr) {
      int leftSize = current->get_left() ? current->get_left()->_wbt_size - 1 : 0;

      if (k == leftSize + 1) {
        return current;
      } else if (k <= leftSize) {
        current = current->get_left();
      } else {
        k       = k - leftSize - 1;
        current = current->get_right();
      }
    }

    return nullptr;
  }
  auto GetKthLargestNode(WBNode *root, int k) -> WBNode *
  {
    if (root == nullptr || k <= 0 || k > root->_wbt_size - 1) {
      return nullptr;
    }

    WBNode *current = root;
    while (current != nullptr) {
      int rightSize = current->get_right() ? current->get_right()->_wbt_size - 1 : 0;

      if (k == rightSize + 1) {
        return current;
      } else if (k <= rightSize) {
        current = current->get_right();
      } else {
        k       = k - rightSize - 1;
        current = current->get_left();
      }
    }

    return nullptr;
  }

  auto GetWindowRangeLabel(WBNode *cur_node, int half_window_size) -> wow_range<WBNode *>
  {
    // get the index of the left boundary
    wow_range<WBNode *> boundary_node;
    int                 k        = half_window_size;
    WBNode             *current  = cur_node;
    int                 leftSize = current->get_left() ? current->get_left()->_wbt_size - 1 : 0;
    if (k <= leftSize) {
      // The kth smallest node is in the left subtree
      WBNode *left_boundry = GetKthLargestNode(current->get_left(), k);
      boundary_node.l_     = left_boundry;
    } else {
      // The kth smallest node may be in the parent path
      k -= leftSize;
      while (current) {
        WBNode *parent = current->get_parent();
        if (!parent) {
          // no parent to go back
          boundary_node.l_ = &(*tree_.begin());
          break;
        }
        if (current == parent->get_right()) {
          if ((parent->get_left() ? parent->get_left()->_wbt_size - 1 : 0) + 1 >= k) {
            if (k == 1) {
              boundary_node.l_ = parent;
              break;
            } else {
              WBNode *left_boundry = GetKthLargestNode(parent->get_left(), k - 1);
              boundary_node.l_     = left_boundry;
              break;
            }
          } else {
            k -= (parent->get_left() ? parent->get_left()->_wbt_size - 1 : 0) + 1;
            current = parent;
          }
        } else {
          current = parent;
        }
      }
    }
    // get the index of the right boundary
    k             = half_window_size;
    current       = cur_node;
    int rightSize = current->get_right() ? current->get_right()->_wbt_size - 1 : 0;
    if (k <= rightSize) {
      WBNode *right_boundry = GetKthSmallestNode(current->get_right(), k);
      boundary_node.u_      = right_boundry;
    } else {
      k -= rightSize;
      while (current) {
        WBNode *parent = current->get_parent();
        if (!parent) {
          boundary_node.u_ = &(*tree_.rbegin());
          break;
        }
        if (current == parent->get_left()) {
          if ((parent->get_right() ? parent->get_right()->_wbt_size - 1 : 0) + 1 >= k) {
            if (k == 1) {
              boundary_node.u_ = parent;
              break;
            } else {
              WBNode *right_boundry = GetKthSmallestNode(parent->get_right(), k - 1);
              boundary_node.u_      = right_boundry;
              break;
            }
          } else {
            k -= (parent->get_right() ? parent->get_right()->_wbt_size - 1 : 0) + 1;
            current = parent;
          }
        } else {
          current = parent;
        }
      }
    }
    return boundary_node;
  }

  void PrintFormatedTree(WBNode *node, const std::string &prefix, bool is_left)
  {
    if (node == nullptr) {
      return;
    }
    std::cout << prefix;
    std::cout << (is_left ? "├──" : "└──");
    // print the value of the node
    std::cout << node->label_ << std::endl;
    // enter the next tree level - left and right branch
    PrintFormatedTree(node->get_left(), prefix + (is_left ? "│   " : "    "), true);
    PrintFormatedTree(node->get_right(), prefix + (is_left ? "│   " : "    "), false);
  }

public:
  size_t   max_N_{};
  MyTree   tree_{};
  WBNode **node_store_;
};

}  // namespace wowlib
