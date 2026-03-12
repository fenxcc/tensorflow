// SPDX-License-Identifier: Apache-2.0
#include "tensorflow/core/grappler/optimizers/shape_only_optimizer.h"

#include <deque>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace grappler {

namespace {

// Utility: normalize an input spec in NodeDef.input: remove '^' and port suffix.
string NodeNameFromInput(const string& input) {
  if (input.empty()) return input;
  string s = input;
  if (s[0] == '^') s = s.substr(1);
  // If contains ":" suffix that is a number (port), strip it.
  size_t pos = s.rfind(':');
  if (pos != string::npos) {
    bool digits = true;
    for (size_t i = pos + 1; i < s.size(); ++i) {
      if (!isdigit(s[i])) {
        digits = false;
        break;
      }
    }
    if (digits) return s.substr(0, pos);
  }
  return s;
}

// Build consumer adjacency: map node name -> set of consumer node names.
void BuildConsumers(const GraphDef& gdef,
                    absl::flat_hash_map<string, absl::flat_hash_set<string>>* out) {
  out->clear();
  for (const NodeDef& n : gdef.node()) {
    const string& name = n.name();
    // Ensure node exists in map
    (*out)[name];  // default construct empty set
  }
  for (const NodeDef& n : gdef.node()) {
    for (const string& inp : n.input()) {
      string pname = NodeNameFromInput(inp);
      if (pname.empty()) continue;
      (*out)[pname].insert(n.name());
    }
  }
}

// Helper: check if op is shape-like terminal
bool IsShapeLikeOpName(const string& op) {
  return op == "Shape" || op == "ShapeN" || op == "Rank" || op == "Size";
}

// Helper: whether op is explicitly disallowed for shape-only replacement
bool IsForbiddenOpName(const string& op) {
  // Unique(UniqueV2), stateful ops, resource ops should be excluded elsewhere.
  return op == "Unique" || op == "UniqueV2";
}

// Helper: whether op is supported by the runtime ShapeOnly kernel and safe to
// replace. Keep this list in sync with shape_only_kernels.cc dispatch.
bool IsSupportedOpName(const string& op) {
  static const absl::flat_hash_set<string> kSupported = {
      // passthrough-like
      "Identity", "Relu", "Neg", "Abs", "Sigmoid", "Tanh",
      // elementwise / broadcast
      "Add", "Mul", "Sub", "Div", "Maximum", "Minimum", "FloorMod",
      // reduces
      "Sum", "Mean", "ReduceSum", "ReduceMean",
      // value-dependent / shape-manipulating
      "Reshape", "GatherV2", "ConcatV2", "ParallelDynamicStitch",
      "DynamicPartition",
      // sparse-segment variants supported
      "SparseSegmentMean", "SparseSegmentSum",
  };
  return kSupported.contains(op);
}

// Helper: common typed attr keys used to carry dtype
bool GetDtypeFromNodeDef(const NodeDef& nd, DataType* out_type) {
  auto it = nd.attr().find("T");
  if (it != nd.attr().end() && it->second.type() != DT_INVALID) {
    *out_type = it->second.type();
    return true;
  }
  it = nd.attr().find("dtype");
  if (it != nd.attr().end() && it->second.type() != DT_INVALID) {
    *out_type = it->second.type();
    return true;
  }
  // fallthrough: not found
  return false;
}

// Convert a PartialTensorShape (here represented by TensorShapeProto-like info)
// into TensorShapeProto if fully defined; returns true on success.
bool PartialShapeToTensorShapeProto(const PartialTensorShape& pts,
                                    TensorShapeProto* out) {
  if (!pts.IsFullyDefined()) return false;
  out->Clear();
  for (int i = 0; i < pts.dims(); ++i) {
    out->add_dim()->set_size(pts.dim_size(i));
  }
  return true;
}

// Protect producers of inputs whose VALUES are read by ShapeOnly kernels at
// runtime
static void ProtectValueDependentProducers(const GraphDef& gdef,
                                           absl::flat_hash_set<string>* blocked) {
  for (const NodeDef& n : gdef.node()) {
    string pname = "";
    if (n.op() == "DynamicPartition") {
      if (n.input_size() >= 2) {
        pname = NodeNameFromInput(n.input(1));
      }
    } else if (n.op() == "ParallelDynamicStitch" || n.op() == "DynamicStitch") {
      if (n.input_size() >= 2 && (n.input_size() % 2 == 0)) {
        int k = n.input_size() / 2;
        for (int i = 0; i < k; ++i) {
          // Protect each indices input
          string p = NodeNameFromInput(n.input(i));
          if (!p.empty()) {
            VLOG(1) << "Protecting value-dependent producer: " << p;
            blocked->insert(p);
          }
        }
        continue;
      }
    } else if (absl::StartsWith(n.op(), "SparseSegment")) {
      if (n.input_size() >= 3) {
        pname = NodeNameFromInput(n.input(2));
      }
    } else if (n.op() == "Reshape") {
      if (n.input_size() >= 2) {
        pname = NodeNameFromInput(n.input(1));
      }
    } else if (n.op() == "ConcatV2") {
      if (n.input_size() >= 1) {
        pname = NodeNameFromInput(n.input(n.input_size() - 1));
      }
    } else if (n.op() == "GatherV2") {
      // GatherV2(params, indices, ...): protect indices producer at input 1
      if (n.input_size() >= 2) {
        pname = NodeNameFromInput(n.input(1));
      }
    }

    if (!pname.empty()) {
      VLOG(1) << "Protecting value-dependent producer: " << pname;
      blocked->insert(pname);
    }
  }
}

// File-scoped helper used by other free functions to detect stateful/side-effect ops.
static bool IsStatefulOrSideEffectNode(const NodeDef& nd) {
  static const absl::flat_hash_set<string> kBlocklist = {
      "VariableV2", "VarHandleOp", "Assign", "AssignAdd", "AssignSub",
      "RandomUniform", "RandomStandardNormal", "RandomUniformInt",
      "QueueEnqueue", "QueueDequeue", "TPUExecute", "Send", "Recv"};
  if (kBlocklist.contains(nd.op())) return true;
  const OpDef* op_def = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(nd.op(), &op_def);
  if (s.ok() && op_def && op_def->is_stateful()) return true;
  return false;
}

}  // namespace

bool ShapeOnlyOptimizer::IsShapeLikeOp(const string& op) const {
  return IsShapeLikeOpName(op);
}

// Start from shape-like ops as seeds; keep adding nodes whose all (data) consumers
// are in seeds, excluding forbidden/stateful/resource ops. This yields nodes that
// only feed shape computations transitively.
void ShapeOnlyOptimizer::AnalyzeCandidates(const GraphDef& gdef) {
  candidates_.clear();

  // Build helper maps
  absl::flat_hash_map<string, const NodeDef*> node_map;
  node_map.reserve(gdef.node_size());
  for (const NodeDef& n : gdef.node()) {
    node_map[n.name()] = &n;
  }

  absl::flat_hash_map<string, absl::flat_hash_set<string>> consumers;
  BuildConsumers(gdef, &consumers);
  VLOG(1) << "Found " << consumers.size() << " consumer nodes.";

  // Seeds: nodes that are direct shape consumers
  absl::flat_hash_set<string> seeds;
  absl::flat_hash_set<string> blocked;
  for (const NodeDef& n : gdef.node()) {
    if (IsShapeLikeOpName(n.op())) {
      seeds.insert(n.name());
      continue;
    }
    if (IsForbiddenOpName(n.op())) {
      blocked.insert(n.name());
    }
    // Also block nodes producing resource dtype (conservative)
    auto it = n.attr().find("dtype");
    if (it != n.attr().end() && it->second.type() == DT_RESOURCE) {
      blocked.insert(n.name());
    }
  }

  // Protect producers of value-dependent inputs (partitions/indices/segment_ids/axis/shape)
  ProtectValueDependentProducers(gdef, &blocked);

  absl::flat_hash_map<string, int> remaining_nonseed_consumers;
  remaining_nonseed_consumers.reserve(consumers.size());
  std::deque<string> workq;

  for (const auto& kv : consumers) {
    const string& node = kv.first;
    // Count consumers that are not seeds.
    int cnt = 0;
    for (const auto& c : kv.second) {
      if (!seeds.contains(c)) cnt++;
    }
    remaining_nonseed_consumers[node] = cnt;

    // Preserve conservative policy: skip blocked/seed nodes and nodes with no consumers.
    if (seeds.contains(node) || blocked.contains(node)) continue;
    if (kv.second.empty()) continue;

    // If no non-seed consumers, seed and enqueue.
    if (cnt == 0) {
      seeds.insert(node);
      workq.push_back(node);
    }
  }

  // Process queue: when a node becomes seeded, decrement the remaining count
  // of its producers (nodes that feed it). If a producer's remaining count
  // drops to zero, and it satisfies the same conditions, seed it and enqueue.
  while (!workq.empty()) {
    string cur = workq.front();
    workq.pop_front();
    auto it_node = node_map.find(cur);
    if (it_node == node_map.end()) continue;
    const NodeDef* nd = it_node->second;
    for (const string& inp : nd->input()) {
      string pname = NodeNameFromInput(inp);
      if (pname.empty()) continue;
      // Skip blocked nodes and already-seeded nodes.
      if (blocked.contains(pname) || seeds.contains(pname)) continue;
      auto itrem = remaining_nonseed_consumers.find(pname);
      if (itrem == remaining_nonseed_consumers.end()) continue;
      // Decrement and check
      --(itrem->second);
      if (itrem->second == 0) {
        auto itc = consumers.find(pname);
        if (itc == consumers.end() || itc->second.empty()) continue;
        seeds.insert(pname);
        workq.push_back(pname);
      }
    }
  }

  // Finally, candidates are seeds excluding the original direct shape nodes.
  for (const string& s : seeds) {
    // Exclude original shape nodes (we treat their replacement separately)
    // and exclude blocked nodes (just in case)
    auto it = node_map.find(s);
    if (it == node_map.end()) continue;
    const NodeDef* nd = it->second;
    if (IsShapeLikeOpName(nd->op())) continue;
    if (blocked.contains(s)) continue;
    candidates_.insert(s);
  }
}

static void CreateShapeOnlyNodes(
    const absl::flat_hash_set<string>& candidates,
    const absl::flat_hash_map<string, const NodeDef*>& node_map,
    absl::flat_hash_map<string, NodeDef>* created_shapeonly) {
  for (const string& cand_name : candidates) {
    auto it_node = node_map.find(cand_name);
    if (it_node == node_map.end()) continue;
    const NodeDef* node = it_node->second;
    if (IsForbiddenOpName(node->op()) || IsStatefulOrSideEffectNode(*node) || !IsSupportedOpName(node->op()))
      continue;

    if (node->op() == "DynamicPartition") {
      int64_t num_partitions = -1;
      auto ait = node->attr().find("num_partitions");
      if (ait != node->attr().end() && ait->second.has_i()) num_partitions = ait->second.i();
      if (num_partitions <= 0) continue;
      for (int i = 0; i < num_partitions; ++i) {
        NodeDef so;
        so.set_op("ShapeOnly");
        so.set_name(cand_name + "_shapeonly_" + std::to_string(i));
        for (const string& inp : node->input()) so.add_input(inp);
        DataType dtype = DT_FLOAT;
        GetDtypeFromNodeDef(*node, &dtype);
        (*so.mutable_attr())["T"].set_type(dtype);
        (*so.mutable_attr())["orig_op"].set_s(node->op());
        (*so.mutable_attr())["output_index"].set_i(i);
        if (!node->device().empty()) so.set_device(node->device());
        created_shapeonly->insert({cand_name + ":" + std::to_string(i), so});
      }
      continue;
    }

    NodeDef so;
    so.set_op("ShapeOnly");
    so.set_name(cand_name + "_shapeonly");
    for (const string& inp : node->input()) so.add_input(inp);
    DataType dtype = DT_FLOAT;
    GetDtypeFromNodeDef(*node, &dtype);
    (*so.mutable_attr())["T"].set_type(dtype);
    (*so.mutable_attr())["orig_op"].set_s(node->op());
    if (!node->device().empty()) so.set_device(node->device());
    (*created_shapeonly)[cand_name] = so;
  }
}

static void FixupCreatedShapeOnlyInputs(absl::flat_hash_map<string, NodeDef>* created_shapeonly) {
  for (auto& kv : *created_shapeonly) {
    NodeDef& so = kv.second;
    for (int i = 0; i < so.input_size(); ++i) {
      string orig_in = so.input(i);
      string cleaned = orig_in;
      bool is_control = false;
      if (!cleaned.empty() && cleaned[0] == '^') {
        is_control = true;
        cleaned = cleaned.substr(1);
      }
      string suffix = "";
      size_t pos = cleaned.rfind(':');
      if (pos != string::npos) {
        bool digits = true;
        for (size_t j = pos + 1; j < cleaned.size(); ++j) if (!isdigit(cleaned[j])) { digits = false; break; }
        if (digits) suffix = cleaned.substr(pos);
      }
      string basename = NodeNameFromInput(cleaned);
      if (basename.empty()) continue;
      string lookup_key = basename + suffix;
      auto it_created = created_shapeonly->find(lookup_key);
      bool matched_with_suffix = true;
      if (it_created == created_shapeonly->end()) {
        it_created = created_shapeonly->find(basename);
        matched_with_suffix = false;
      }
      if (it_created == created_shapeonly->end()) continue;
      string new_in = matched_with_suffix ? it_created->second.name()
                                          : it_created->second.name() + suffix;
      if (is_control) new_in = string("^") + new_in;
      so.set_input(i, new_in);
    }
  }
}

static void RebuildResultNodes(const GraphDef& gdef,
                               const absl::flat_hash_map<string, NodeDef>& created_shapeonly,
                               const absl::flat_hash_set<string>& candidates,
                               std::vector<NodeDef>* result_nodes) {
  // Track which created nodes have already been inserted so we don't duplicate.
  absl::flat_hash_set<string> inserted_keys;

  for (const NodeDef& n : gdef.node()) {
    NodeDef newn = n;
    // Rewrite inputs of nodes that are shape consumers or candidates to point
    // to the ShapeOnly replacements where appropriate.
    for (int i = 0; i < newn.input_size(); ++i) {
      string orig_in = newn.input(i);
      string cleaned = orig_in;
      bool is_control = false;
      if (!cleaned.empty() && cleaned[0] == '^') { is_control = true; cleaned = cleaned.substr(1); }
      string suffix = "";
      size_t pos = cleaned.rfind(':');
      if (pos != string::npos) {
        bool digits = true;
        for (size_t j = pos + 1; j < cleaned.size(); ++j) if (!isdigit(cleaned[j])) { digits = false; break; }
        if (digits) suffix = cleaned.substr(pos);
      }
      string basename = NodeNameFromInput(cleaned);
      string lookup_key = basename + suffix;
      auto it_created = created_shapeonly.find(lookup_key);
      bool matched_with_suffix = true;
      if (it_created == created_shapeonly.end()) {
        it_created = created_shapeonly.find(basename);
        matched_with_suffix = false;
      }
      if (it_created == created_shapeonly.end()) continue;
      if (!IsShapeLikeOpName(n.op()) && !candidates.contains(n.name())) continue;
      string new_in = matched_with_suffix ? it_created->second.name()
                                          : it_created->second.name() + suffix;
      if (!orig_in.empty() && orig_in[0] == '^') new_in = string("^") + new_in;
      newn.set_input(i, new_in);
    }

    // Always keep the original node in the output graph.
    result_nodes->push_back(std::move(newn));

    // Immediately after the original node, insert any created ShapeOnly nodes
    // that were derived from this producer (keys equal to `name` or starting
    // with `name:` for per-output variants).
    string producer = n.name();
    string prefix = producer + ":";
    for (const auto& kv : created_shapeonly) {
      const string& key = kv.first;
      if (inserted_keys.contains(key)) continue;
      if (key == producer || absl::StartsWith(key, prefix)) {
        result_nodes->push_back(kv.second);
        inserted_keys.insert(key);
      }
    }
  }

  // Append any created ShapeOnly nodes whose producer was not present in the
  // original graph ordering (should be rare). Ensure we don't duplicate.
  for (const auto& kv : created_shapeonly) {
    if (!inserted_keys.contains(kv.first)) {
      result_nodes->push_back(kv.second);
      inserted_keys.insert(kv.first);
    }
  }
}

// For PoC: create ShapeOnly node per candidate
absl::Status ShapeOnlyOptimizer::InsertShapeOnlyNodes(const GrapplerItem& item,
                                                      GraphDef* gdef) {
  if (candidates_.empty()) return absl::OkStatus();

  absl::flat_hash_map<string, const NodeDef*> node_map;
  for (const NodeDef& n : gdef->node()) node_map[n.name()] = &n;

  std::vector<NodeDef> result_nodes;
  absl::flat_hash_map<string, NodeDef> created_shapeonly;

  CreateShapeOnlyNodes(candidates_, node_map, &created_shapeonly);
  FixupCreatedShapeOnlyInputs(&created_shapeonly);
  RebuildResultNodes(*gdef, created_shapeonly, candidates_, &result_nodes);

  // created_shapeonly nodes are inserted in-place by RebuildResultNodes above;
  // no separate append here is necessary.

  GraphDef out;
  out.mutable_library()->CopyFrom(gdef->library());
  out.mutable_versions()->CopyFrom(gdef->versions());
  for (const NodeDef& n : result_nodes) *out.add_node() = n;
  *gdef = std::move(out);

  VLOG(1) << "ShapeOnlyOptimizer: inserted " << created_shapeonly.size()
          << " ShapeOnly nodes and retargeted consumers.";
  return absl::OkStatus();
}

absl::Status ShapeOnlyOptimizer::Optimize(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* optimized_graph) {
  // Dump graph before any optimization so we can compare before/after.
  // Uses TensorFlow's standard DumpGraph helper which respects the
  // TF_DUMP_GRAPH_FMT and TF_DUMP_GRAPH_PREFIX environment variables.
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("shape_only_optimizer_before", item.graph);
  }

  // Stage 0: Analyze and mark candidate nodes.
  AnalyzeCandidates(item.graph);
  VLOG(1) << "ShapeOnlyOptimizer: Stage0 found " << candidates_.size()
          << " candidate nodes (only-used-for-shape).";

  *optimized_graph = item.graph;
  // Stage 2: Insert ShapeOnly nodes for candidates (safety checks applied).
  TF_RETURN_IF_ERROR(InsertShapeOnlyNodes(item, optimized_graph));
  VLOG(1) << "ShapeOnlyOptimizer: Stage2 ShapeOnly insert done.";

  // Dump the resulting graph after ShapeOnly insertion for diffing.
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("shape_only_optimizer_after", *optimized_graph);
  }

  return absl::OkStatus();
}

}  // end namespace grappler
}  // end namespace tensorflow