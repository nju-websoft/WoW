#pragma once
#include "disk.hh"
#include "utils.hh"
#include "order_table.hh"
#include "visit_list.hh"
#include "space_dist.hh"
#include "memory.hh"

namespace wowlib {
// fixme: wow_bitset<int> is used only for benchmark, should remove it
#define should_check_filter(filter_type, att_type)                             \
  (std::is_same_v<filter_type, wowlib::wow_range<att_type>> ||                 \
      std::is_same_v<filter_type, wowlib::wow_range<att_label_t<att_type>>> || \
      std::is_same_v<filter_type, wowlib::wow_set<att_type>> ||                \
      std::is_same_v<filter_type, wowlib::wow_bitset<int>> ||                  \
      std::is_same_v<filter_type, wowlib::wow_bitset<label_t>>)

template <typename att_t = int, typename vec_t = float>
class WoWIndex
{

public:
  WoWIndex(size_t max_elements, size_t vec_d, size_t M, size_t efc, std::string space_name, size_t o = 4,
      size_t wp = 10, bool auto_raise_wp = true)
      : max_elements_(max_elements), vec_d_(vec_d), wp_(wp), o_(o), M_(M), efc_(efc)
  {
    if (space_name == "l2") {
      space_ = new wowlib::L2Space(vec_d_);
    } else if (space_name == "ip") {
      space_ = new wowlib::InnerProductSpace(vec_d_);
    } else {
      throw std::runtime_error("unsupported space type " + space_name + ", supported: l2, ip");
    }
    fstdistfunc_     = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    window_size_.emplace_back(2);
    while (window_size_.back() < max_elements_) {
      window_size_.emplace_back(o_ * window_size_.back());
    }
    if (wp_ + 1 < window_size_.size()) {
      if (auto_raise_wp) {
        std::cout << "auto_raise_wp is true. to cover all unique values, increased wp_ to " << window_size_.size() - 1
                  << std::endl;
        wp_ = window_size_.size() - 1;
      } else {
        std::cout << "auto_raise_wp is false. please make sure " << wp_ << " is good" << std::endl;
        window_size_.resize(wp_ + 1);
      }
    } else {
      if (wp_ + 1 != window_size_.size())
        std::cout << "consider using wp_<=" << wp_ << " to reduce memory usage" << std::endl;
      for (size_t i = window_size_.size(); i <= wp_ + 1; ++i) {
        window_size_.emplace_back(o_ * window_size_.back());
      }
    }
    sizelinks_per_element_ =
        sizeof(label_t) + sizeof(att_t) + sizeof(vec_t) * vec_d_ + sizeof(tableint) * (M_ + 1) * (wp_ + 1);
    sizelinklistsmem_ = max_elements_ * sizelinks_per_element_;
    offset_label_     = 0;
    offset_att_       = offset_label_ + sizeof(label_t);
    offset_vec_       = offset_att_ + sizeof(att_t);
    offset_linklists_ = offset_vec_ + sizeof(vec_t) * vec_d_;
    linklistsmemory_  = (char *)glass::alloc2M(sizelinklistsmem_);
    if (linklistsmemory_ == nullptr) {
      throw std::runtime_error("Not enough memory: WoWIndex failed to allocate linklist");
    }
    order_table_    = new WBTreeOrderTable<att_t>(max_elements_);
    linklist_locks_ = std::vector<std::mutex>(max_elements_);
    visited_pool_.Init(max_elements_);
  }

  WoWIndex(const WoWIndex &)            = delete;
  WoWIndex &operator=(const WoWIndex &) = delete;
  WoWIndex(WoWIndex &&)                 = delete;
  WoWIndex &operator=(WoWIndex &&)      = delete;
  WoWIndex()                            = delete;

  void save(const std::string &location)
  {
    std::ofstream ofs(location, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open index file for writing: " + location);
    }
    WriteBinaryPOD(ofs, max_elements_);
    WriteBinaryPOD(ofs, vec_d_);
    WriteBinaryPOD(ofs, wp_);
    WriteBinaryPOD(ofs, o_);
    WriteBinaryPOD(ofs, M_);
    WriteBinaryPOD(ofs, efc_);
    WriteBinaryPOD(ofs, curvec_num_);
    WriteBinaryPOD(ofs, cur_max_layer_);
    WriteBinaryPOD(ofs, sizelinks_per_element_);
    WriteBinaryPOD(ofs, sizelinklistsmem_);
    WriteBinaryPOD(ofs, offset_label_);
    WriteBinaryPOD(ofs, offset_att_);
    WriteBinaryPOD(ofs, offset_vec_);
    WriteBinaryPOD(ofs, offset_linklists_);

    ofs.write(linklistsmemory_, sizelinklistsmem_);
    ofs.close();
  }

  WoWIndex(const std::string &location, std::string space_name)
  {
    std::ifstream ifs(location, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open index file: " + location);
    }
    ReadBinaryPOD(ifs, max_elements_);
    ReadBinaryPOD(ifs, vec_d_);
    ReadBinaryPOD(ifs, wp_);
    ReadBinaryPOD(ifs, o_);
    ReadBinaryPOD(ifs, M_);
    ReadBinaryPOD(ifs, efc_);
    ReadBinaryPOD(ifs, curvec_num_);
    ReadBinaryPOD(ifs, cur_max_layer_);
    ReadBinaryPOD(ifs, sizelinks_per_element_);
    ReadBinaryPOD(ifs, sizelinklistsmem_);
    ReadBinaryPOD(ifs, offset_label_);
    ReadBinaryPOD(ifs, offset_att_);
    ReadBinaryPOD(ifs, offset_vec_);
    ReadBinaryPOD(ifs, offset_linklists_);

    if (sizelinks_per_element_ !=
        sizeof(label_t) + sizeof(att_t) + sizeof(vec_t) * vec_d_ + sizeof(tableint) * (M_ + 1) * (wp_ + 1)) {
      throw std::runtime_error("possible index file corruption, sizelinks_per_element_ is not equal to expected size");
    }
    linklistsmemory_ = (char *)glass::alloc2M(sizelinklistsmem_);
    if (linklistsmemory_ == nullptr) {
      throw std::runtime_error("Failed to allocate memory for linklistsmemory_");
    }
    ifs.read(linklistsmemory_, sizelinklistsmem_);
    order_table_ = new WBTreeOrderTable<att_t>(max_elements_);
    for (tableint i = 0; i < max_elements_; ++i) {
      auto att_mem = GetAttByInternalID(i);
      order_table_->InsertAttInid({*att_mem, *GetLabelByInternalID(i)}, i);
    }
    ifs.close();
    linklist_locks_ = std::vector<std::mutex>(max_elements_);
    visited_pool_.Init(max_elements_);
    visited_pool_.Return(visited_pool_.Get());

    if (space_name == "l2") {
      space_ = new wowlib::L2Space(vec_d_);
    } else if (space_name == "ip") {
      space_ = new wowlib::InnerProductSpace(vec_d_);
    } else {
      throw std::runtime_error("unsupported space type " + space_name + ", supported: l2, ip");
    }
    fstdistfunc_     = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();

    window_size_.emplace_back(2);
    while (window_size_.size() < wp_ + 1) {
      window_size_.emplace_back(o_ * window_size_.back());
    }
    std::cout << "========================== index summary =========================" << std::endl;
    std::cout << "max_elements_: " << max_elements_ << " vec_d_: " << vec_d_ << " wp_: " << wp_ << " o_: " << o_
              << " M_: " << M_ << " efc_: " << efc_ << std::endl;
    std::cout << "curvec_num_: " << curvec_num_ << " cur_max_layer_: " << cur_max_layer_ << std::endl;
    // calculate average out degree for each layer
    for (size_t layer = 0; layer <= cur_max_layer_; ++layer) {
      int M = 0;
      for (size_t i = 0; i < curvec_num_; ++i) {
        auto ll = GetLinkListByInternalID(i, layer);
        M += ll[M_];
      }
      std::cout << "Layer: " << layer << ", average M: " << M / curvec_num_ << std::endl;
    }
    std::cout << "===================================================================" << std::endl;
  }

  ~WoWIndex()
  {
    free(linklistsmemory_);
    linklistsmemory_ = nullptr;
    delete space_;
    delete order_table_;
  }

  void insert(const label_t label, const vec_t *v, const att_t &attribute, bool replace_deleted = false)
  {
    int      max_level_copy = -1;
    tableint cur_num        = -1;
    {
      std::unique_lock<std::mutex> lock(max_layer_lock_);
      cur_num = curvec_num_++;
      {
        if (cur_num == 0) {
          auto label_mem = GetLabelByInternalID(cur_num);
          auto att_mem   = GetAttByInternalID(cur_num);
          auto vec_mem   = GetVecByInternalID(cur_num);
          memcpy(label_mem, &label, sizeof(label_t));
          memcpy(att_mem, &attribute, sizeof(att_t));
          memcpy(vec_mem, v, sizeof(vec_t) * vec_d_);
          std::unique_lock<std::mutex> list_lock(linklist_locks_[cur_num]);
          for (layer_t layer = 0; layer <= wp_; ++layer) {
            auto ll = GetLinkListByInternalID(cur_num, layer);
            ll[M_]  = 0;
          }
          order_table_->InsertAttInid({*att_mem, *label_mem}, cur_num);
          return;
        }
      }

      if (curvec_num_ > window_size_[cur_max_layer_]) {
        if (this->cur_max_layer_ == wp_) {
          throw std::runtime_error("no enough space for new layer");
        }
        std::cout << "raise layer from " << this->cur_max_layer_ << " to " << this->cur_max_layer_ + 1 << std::endl;
        this->cur_max_layer_++;
        // copy
        for (tableint lower_id = 0; lower_id < curvec_num_; ++lower_id) {
          auto lower_link_list = GetLinkListByInternalID(lower_id, this->cur_max_layer_ - 1);
          if (lower_link_list[M_] == 0) {
            continue;
          }
          auto upper_link_list = GetLinkListByInternalID(lower_id, this->cur_max_layer_);
          memcpy(upper_link_list, lower_link_list, (M_ + 1) * sizeof(tableint));
        }
      }
      max_level_copy = cur_max_layer_;
    }
    if (max_level_copy == -1 || cur_num == -1) {
      throw std::runtime_error("-1: initilize failed");
    }
    std::vector<std::vector<dist_id_pair>> tmp_linklist(max_level_copy + 1);
    std::vector<dist_id_pair>              cur_allc;
    auto                                   curc_record = visited_pool_.Get();
    curc_record->Clear();
    for (layer_t layer = max_level_copy; layer >= 0; --layer) {
      size_t                half_window_size = window_size_[layer] / 2;
      std::vector<tableint> entry_points;

      auto query_rng = order_table_->GetWindowedFilterAndEntries({attribute, label}, half_window_size, entry_points);

      for (auto ep_id : entry_points) {
        auto d = fstdistfunc_(v, GetVecByInternalID(ep_id), dist_func_param_);
        metric_dist_comps_++;
        cur_allc.emplace_back(d, ep_id);
      }
      /**
       * @brief building optimization
       * we can simply use the following code to get the nearest candidates for all layers:
       *
       * auto allc = this->SearchCandidatesKNN(v, label, {s_pos, e_pos}, this->brparam_.efc_, status_, true);
       *
       * but the indexing time is log^2(n). The following code ensures in the worst case, the time is log^2 (n).
       * It first check the previously retrieved candidates, if the number of candidates is larger than M, we can
       * directly use them, otherwise, we need to search on the incomplete graph.
       *
       */
      std::vector<dist_id_pair> filtered_curc;
      for (const auto &[d, i] : cur_allc) {
        auto i_att_label = att_label_t{*GetAttByInternalID(i), *GetLabelByInternalID(i)};
        if (i_att_label >= query_rng.l_ && i_att_label <= query_rng.u_) {
          filtered_curc.emplace_back(d, i);
          curc_record->Set(i);
        }
      }
      cur_allc = std::move(filtered_curc);
      if (cur_allc.size() < M_) {
        auto new_c = SearchCandidates<true>(cur_allc, v, query_rng, {layer, max_level_copy}, efc_, cur_num);
        for (const auto &[d, i] : new_c) {
          if (i == cur_num) {
            throw std::runtime_error("repeated internal id");
          }
          if (!curc_record->Test(i)) {
            cur_allc.emplace_back(d, i);
          }
        }
      }
      auto pruned         = PruneByHeuristic(cur_allc, M_ / 2);
      tmp_linklist[layer] = std::move(pruned);
    }
    visited_pool_.Return(curc_record);

    // connect
    {
      auto label_mem = GetLabelByInternalID(cur_num);
      auto att_mem   = GetAttByInternalID(cur_num);
      auto vec_mem   = GetVecByInternalID(cur_num);
      memcpy(label_mem, &label, sizeof(label_t));
      memcpy(att_mem, &attribute, sizeof(att_t));
      memcpy(vec_mem, v, sizeof(vec_t) * vec_d_);
      std::unique_lock<std::mutex> lock_cur(linklist_locks_[cur_num]);
      for (layer_t layer = max_level_copy; layer >= 0; --layer) {
        auto ll = GetLinkListByInternalID(cur_num, layer);
        ll[M_]  = (tableint)tmp_linklist[layer].size();
        for (tableint i = 0; i < ll[M_]; ++i) {
          if (tmp_linklist[layer][i].id_ == cur_num) {
            throw std::runtime_error("pruned[i].id_ == cur_num");
          }
          if (ll[i]) {
            throw std::runtime_error("newly added point should have blank link list");
          }
          ll[i] = tmp_linklist[layer][i].id_;
        }
      }
    }
    for (layer_t layer = max_level_copy; layer >= 0; --layer) {
      // add and prune for neighbors in the same layer
      for (const auto &[nn_d, nn_i] : tmp_linklist[layer]) {
        std::lock_guard<std::mutex> lock_nn(this->linklist_locks_[nn_i]);
        auto                        nn_ll    = GetLinkListByInternalID(nn_i, layer);
        auto                        nn_ll_sz = nn_ll[M_];
        if (nn_ll_sz < M_) {
          nn_ll[nn_ll_sz] = cur_num;
          nn_ll[M_]++;
        } else {
          std::vector<dist_id_pair> nn_allc;
          nn_allc.reserve(nn_ll_sz + 1);
          for (tableint i = 0; i < nn_ll_sz; ++i) {
            nn_allc.emplace_back(
                fstdistfunc_(GetVecByInternalID(nn_i), GetVecByInternalID(nn_ll[i]), dist_func_param_), nn_ll[i]);
          }
          size_t half_window_size = window_size_[layer] / 2;
          /**********pruning 1 */
          std::vector<att_label_t<att_t>> cand_att_label_vec;
          for (int i = 0; i < nn_allc.size(); ++i) {
            cand_att_label_vec.emplace_back(*GetAttByInternalID(nn_allc[i].id_), *GetLabelByInternalID(nn_allc[i].id_));
          }
          nn_allc = order_table_->GetInWindowCandidates(
              nn_allc, cand_att_label_vec, {*GetAttByInternalID(nn_i), *GetLabelByInternalID(nn_i)}, half_window_size);
          nn_allc.emplace_back(nn_d, cur_num);
          /**********pruning 2 */
          auto nn_pruned = PruneByHeuristic(nn_allc, M_);
          nn_ll[M_]      = (tableint)nn_pruned.size();
          for (tableint i = 0; i < nn_ll[M_]; ++i) {
            nn_ll[i] = nn_pruned[i].id_;
          }
        }
      }
    }
    order_table_->InsertAttInid({*GetAttByInternalID(cur_num), *GetLabelByInternalID(cur_num)}, cur_num);
  }

  template <typename filter_t = wow_range<att_t>>
  auto searchKNN(const vec_t *query_vec, size_t efs, size_t k, const filter_t &filter)
      -> std::vector<std::pair<dist_t, label_t>>
  {
    // compiler check: filter should be one of the following types:
    // wow_range<att_t> wow_bitset<label_t> wow_bitset<int> wow_set<att_t>
    wow_range<layer_t>        layer_rng;
    std::vector<dist_id_pair> ep_dist_id_pairs;
    constexpr bool            check_filter = should_check_filter(filter_t, att_t);
    if constexpr (!check_filter) {
      // randomly select a ep from 0-curvec_num_-1
      auto ep_id = rand() % curvec_num_;
      auto d     = fstdistfunc_(query_vec, GetVecByInternalID(ep_id), dist_func_param_);
      metric_dist_comps_++;
      ep_dist_id_pairs.emplace_back(d, ep_id);
      layer_rng = {static_cast<layer_t>(cur_max_layer_), static_cast<layer_t>(cur_max_layer_)};
    } else if constexpr (std::is_same_v<filter_t, wow_range<att_t>>) {
      std::vector<tableint> eps;
      layer_rng = DecideLayerRange(filter, eps);
      for (auto ep_id : eps) {
        auto d = fstdistfunc_(query_vec, GetVecByInternalID(ep_id), dist_func_param_);
        metric_dist_comps_++;
        ep_dist_id_pairs.emplace_back(d, ep_id);
      }
    } else {  // wow_set<att_t>
      for (tableint i = 0; i < curvec_num_; ++i) {
        if (ep_dist_id_pairs.size() >= efs) {
          break;
        }
        if (filter.Test(*GetAttByInternalID(i))) {
          auto d = fstdistfunc_(query_vec, GetVecByInternalID(i), dist_func_param_);
          metric_dist_comps_++;
          ep_dist_id_pairs.emplace_back(d, i);
        }
      }
      layer_rng = {0, static_cast<layer_t>(cur_max_layer_)};
    }

    std::vector<dist_id_pair> result;
    if constexpr (std::is_same_v<filter_t, wow_range<att_t>>) {
      wow_range<att_label_t<att_t>> dedup_filter{{filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()}};
      result = SearchCandidates<false>(ep_dist_id_pairs, query_vec, dedup_filter, layer_rng, efs);
    } else {
      result = SearchCandidates<false>(ep_dist_id_pairs, query_vec, filter, layer_rng, efs);
    }

    while (result.size() > k) {
      POP_HEAP(result);
    }
    std::vector<std::pair<dist_t, label_t>> final_res(result.size());
    for (int i = 0; i < final_res.size(); ++i) {
      final_res[i].first  = result[i].dist_;
      final_res[i].second = *GetLabelByInternalID(result[i].id_);
    }
    return final_res;
  }

  inline __attribute__((always_inline)) auto GetDimension() const -> size_t { return vec_d_; }
  inline __attribute__((always_inline)) auto GetMaxElements() const -> size_t { return max_elements_; }
  inline __attribute__((always_inline)) auto GetCurNum() const -> size_t { return curvec_num_; }
  inline __attribute__((always_inline)) auto GetCurMaxLayer() const -> size_t { return cur_max_layer_; }
  inline __attribute__((always_inline)) auto GetM() const -> size_t { return M_; }
  inline __attribute__((always_inline)) auto GetEfc() const -> size_t { return efc_; }

private:
  // always inline
  inline __attribute__((always_inline)) auto GetLabelByInternalID(tableint internal_id) -> label_t *
  {
    return (label_t *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_label_);
  }

  inline __attribute__((always_inline)) auto GetAttByInternalID(tableint internal_id) -> att_t *
  {
    return (att_t *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_att_);
  }

  inline __attribute__((always_inline)) auto GetVecByInternalID(tableint internal_id) -> vec_t *
  {
    return (vec_t *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_vec_);
  }

  inline __attribute__((always_inline)) auto GetLinkListByInternalID(tableint internal_id, layer_t layer) -> tableint *
  {
    // store layers reversely to prefetch the next layer
    return (tableint *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_linklists_ +
                        (wp_ - layer) * (M_ + 1) * sizeof(tableint));
  }

  template <bool is_build, typename filter_t>
  auto SearchCandidates(std::vector<dist_id_pair> &eps, const vec_t *v, const filter_t &filter,
      const wow_range<layer_t> &layer_rng, const size_t ef, tableint ignore = -1) -> std::vector<dist_id_pair>
  {
    constexpr bool check_filter = should_check_filter(filter_t, att_t);
    if (eps.empty())
      return {};
    auto visited = visited_pool_.Get();
    visited->Clear();
    if (is_build && ignore != -1) {
      visited->Set(ignore);
    }
    // std::vector<dist_id_pair> visited_pairs;
    std::vector<dist_id_pair> result;
    std::vector<dist_id_pair> candidates;
    for (auto ep : eps) {
      PUSH_HEAP(candidates, -ep.dist_, ep.id_);
      PUSH_HEAP(result, ep.dist_, ep.id_);
      visited->Set(ep.id_);
      // if constexpr (is_build) {
      //   visited_pairs.emplace_back(ep.dist_, ep.id_);
      // }
    }
    auto res_max_dist = TOP_HEAP(result).dist_;
    while (!candidates.empty()) {
      auto [dist, id] = TOP_HEAP(candidates);
      if constexpr (is_build) {
        if (((-dist) > res_max_dist) && (result.size() == ef)) {
          break;
        }
      } else {
        if ((-dist) > res_max_dist) {
          break;
        }
      }

#ifdef USE_SSE
      _mm_prefetch((char *)GetLinkListByInternalID(id, layer_rng.u_), _MM_HINT_T2);
#endif
      POP_HEAP(candidates);
      metric_hops_++;

      if constexpr (is_build)
        linklist_locks_[id].lock();
      size_t neighbor_cnt = 0;

      for (layer_t layer = layer_rng.u_; layer >= layer_rng.l_; --layer) {
        if (neighbor_cnt >= M_) {
          break;
        }
        auto ll    = GetLinkListByInternalID(id, layer);
        auto ll_sz = ll[M_];
#ifdef USE_SSE
        _mm_prefetch((char *)(visited->GetData(ll[0])), _MM_HINT_T0);
        _mm_prefetch((char *)(visited->GetData(ll[0]) + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(GetAttByInternalID(ll[0])), _MM_HINT_T0);
        _mm_prefetch((char *)(ll + 1), _MM_HINT_T0);
#endif
        bool visit_next_layer = false;
        for (tableint i = 0; i < ll_sz; ++i) {
          if (neighbor_cnt >= M_) {
            break;
          }
          auto nn_id = ll[i];
#ifdef USE_SSE
          _mm_prefetch((char *)(visited->GetData(ll[i + 1])), _MM_HINT_T0);
          _mm_prefetch((char *)(GetAttByInternalID(ll[i + 1])), _MM_HINT_T0);
#endif
          auto &nn_att = *GetAttByInternalID(nn_id);
          if constexpr (check_filter) {
            if constexpr (std::is_same_v<filter_t, wow_range<att_label_t<att_t>>>) {
              if (!filter.Test({nn_att, *GetLabelByInternalID(nn_id)})) {
                visit_next_layer = true;
                continue;
              }
            } else {
              if (!filter.Test(nn_att)) {
                visit_next_layer = true;
                continue;
              }
            }
          }
          if (visited->Test(nn_id)) {
            continue;
          }
          visited->Set(nn_id);
          auto nn_dist = fstdistfunc_(v, GetVecByInternalID(nn_id), dist_func_param_);
          metric_dist_comps_++;
          neighbor_cnt++;
          // if constexpr (is_build) {
          //   visited_pairs.emplace_back(nn_dist, nn_id);
          // }
          if (result.size() < ef || nn_dist < res_max_dist) {
            PUSH_HEAP(candidates, -nn_dist, nn_id);
#ifdef USE_SSE
            _mm_prefetch((char *)linklistsmemory_ + TOP_HEAP(candidates).id_ * sizelinks_per_element_, _MM_HINT_T2);
#endif
            PUSH_HEAP(result, nn_dist, nn_id);
            if (result.size() > ef) {
              POP_HEAP(result);
            }
            res_max_dist = TOP_HEAP(result).dist_;
          }
        }
        if (!is_build && !visit_next_layer) {
          break;
        }
      }
      if (is_build)
        linklist_locks_[id].unlock();
    }
    visited_pool_.Return(visited);
    // if constexpr (is_build) {
    //   return visited_pairs;
    // } else {
    return result;
    // }
  }

  auto PruneByHeuristic(std::vector<dist_id_pair> &candidates, const size_t M) -> std::vector<dist_id_pair>
  {
    /**
     * @brief
     *   dist is the distance from query q to a, id is the id of a
     *   prune metric: scan the candidates sequentially, currently visited node xa, if there is a node xb in the
     *   result set requires: dist(q, xa) < dist(q, xb) and dist(xa, xb) < dist(q, xa), then it should be pruned,
     *   as candidates are ordered by  dist(q, a), we can prune the candidates by just checking the latter condition
     */

    if (candidates.size() <= M) {
      return candidates;
    }
    if (M == 0) {
      return {};
    }
    if (M == 1) {
      return {candidates[0]};
    }
    // ensure the candidates are sorted by distance
    std::sort(candidates.begin(), candidates.end());
    std::vector<dist_id_pair> pruned;
    for (const auto &[db, ib] : candidates) {
      if (pruned.size() >= M) {
        break;
      }
      bool good = true;
      for (const auto &[da, ia] : pruned) {
        auto curdist = fstdistfunc_(GetVecByInternalID(ib), GetVecByInternalID(ia), dist_func_param_);
        metric_dist_comps_++;
        if (curdist < db) {  // i == ib is to avoid repeated points
          good = false;
          break;
        }
      }
      if (good) {
        pruned.emplace_back(db, ib);
      }
    }
    return pruned;
  }

  auto DecideLayerRange(const wow_range<att_t> &filter_range, std::vector<tableint> &OUT_eps) -> wow_range<layer_t>
  {
    wow_range<layer_t> new_layer_rng;
    size_t             filter_card = order_table_->GetRangeCardinality(
        {filter_range.l_, 0}, {filter_range.u_, std::numeric_limits<label_t>::max()}, OUT_eps);
    auto c_it = std::lower_bound(window_size_.begin(), window_size_.end(), filter_card);
    if (c_it == this->window_size_.end() || *c_it > filter_card) {
      c_it--;
    }
    int c_it_idx = std::distance(window_size_.begin(), c_it);
    if (c_it_idx == 0) {
      new_layer_rng.u_ = c_it_idx + 1;
    } else if (c_it_idx == wp_) {
      new_layer_rng.u_ = c_it_idx;
    } else {
      int c_l = c_it_idx - 1;
      int c_u = c_it_idx + 1;
      // find the largest fraction
      float frac_l = 1.0 * this->window_size_[c_l] / filter_card;
      float frac_u = 1.0 * filter_card / std::min((int)this->window_size_[c_u], (int)max_elements_);
      if (frac_l > frac_u) {
        new_layer_rng.u_ = c_it_idx;
      } else {
        new_layer_rng.u_ = c_u;
      }
    }
    new_layer_rng.l_ = 0;
    return new_layer_rng;
  }

public:
  // single thread profiling
  size_t metric_dist_comps_{0};
  size_t metric_hops_{0};

private:
  size_t max_elements_{0};
  size_t vec_d_;
  size_t wp_{0};
  size_t o_{4};
  size_t M_{24};
  size_t efc_{256};

  size_t curvec_num_{0};
  size_t cur_max_layer_{0};

  size_t sizelinks_per_element_{0};
  size_t sizelinklistsmem_{0};

  size_t offset_label_{0};
  size_t offset_att_{0};
  size_t offset_vec_{0};
  size_t offset_linklists_{0};

  char *linklistsmemory_{nullptr};

  std::mutex              max_layer_lock_;
  std::vector<std::mutex> linklist_locks_;
  // function pointer to float (const float *, const float *, size_t d)
  wowlib::SpaceInterface<vec_t> *space_{nullptr};
  wowlib::DISTFUNC<vec_t>        fstdistfunc_{nullptr};
  void                          *dist_func_param_{nullptr};

  WBTreeOrderTable<att_t> *order_table_{nullptr};

  VisitedPool<VisitedList<tableint>> visited_pool_;
  std::vector<size_t>                window_size_;
};
}  // namespace wowlib