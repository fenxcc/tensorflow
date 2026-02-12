// SPDX-License-Identifier: Apache-2.0
#include "tensorflow/core/grappler/optimizers/shape_only_optimizer.h"

#include <deque>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/jit/shape_inference.h"  // for InferShapes
#include "tensorflow/core/framework/tensor_shape.pb.h"

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
    if (n.op() == "DynamicPartition") {
      if (n.input_size() >= 2) {
        string pname = NodeNameFromInput(n.input(1));
        if (!pname.empty()) blocked->insert(pname);
      }
    } else if (n.op() == "ParallelDynamicStitch") {
      if (n.input_size() >= 2 && (n.input_size() % 2 == 0)) {
        int k = n.input_size() / 2;
        for (int i = 0; i < k; ++i) {
          string pname = NodeNameFromInput(n.input(i));
          if (!pname.empty()) blocked->insert(pname);
        }
      }
    } else if (absl::StartsWith(n.op(), "SparseSegment")) {
      if (n.input_size() >= 3) {
        string pname = NodeNameFromInput(n.input(2));
        if (!pname.empty()) blocked->insert(pname);
      }
    } else if (n.op() == "Reshape") {
      if (n.input_size() >= 2) {
        string pname = NodeNameFromInput(n.input(1));
        if (!pname.empty()) blocked->insert(pname);
      }
    } else if (n.op() == "ConcatV2") {
      if (n.input_size() >= 1) {
        string pname = NodeNameFromInput(n.input(n.input_size() - 1));
        if (!pname.empty()) blocked->insert(pname);
      }
    } else if (n.op() == "GatherV2") {
      if (n.input_size() > 2) {
        string pname = NodeNameFromInput(n.input(2));
        if (!pname.empty()) blocked->insert(pname);
      }
    }
  }
}

}  // namespace

bool ShapeOnlyOptimizer::IsShapeLikeOp(const string& op) const {
  return IsShapeLikeOpName(op);
}

bool ShapeOnlyOptimizer::IsStatefulOrSideEffectOp(const NodeDef& nd) const {
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

// Start from shape-like ops as seeds; keep adding nodes whose all (data) consumers
// are in seeds, excluding forbidden/stateful/resource ops. This yields nodes that
// only feed shape computations transitively.
void ShapeOnlyOptimizer::AnalyzeCandidates(const GraphDef& gdef) {
  candidates_.clear();

  // Build helper maps
  absl::flat_hash_map<string, const NodeDef*> node_map;
  node_map.reserve(gdef.node_size());
  for (const NodeDef& n : gdef.node()) node_map[n.name()] = &n;

  absl::flat_hash_map<string, absl::flat_hash_set<string>> consumers;
  BuildConsumers(gdef, &consumers);

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

// For PoC: create ShapeOnly node per candidate
absl::Status ShapeOnlyOptimizer::InsertShapeOnlyNodes(const GrapplerItem& item,
                                                      GraphDef* gdef) {
  if (candidates_.empty()) return absl::OkStatus();

  // Prepare lookup map
  absl::flat_hash_map<string, const NodeDef*> node_map;
  for (const NodeDef& n : gdef->node()) node_map[n.name()] = &n;

  // Build consumers map (kept for possible future use)
  absl::flat_hash_map<string, absl::flat_hash_set<string>> consumers;
  BuildConsumers(*gdef, &consumers);

  // We'll produce new nodes list and created shapeonly nodes
  std::vector<NodeDef> result_nodes;
  absl::flat_hash_map<string, NodeDef> created_shapeonly;

  for (const string& cand_name : candidates_) {
    auto it_node = node_map.find(cand_name);
    if (it_node == node_map.end()) continue;
    const NodeDef* node = it_node->second;
    // Skip if forbidden / stateful
    if (IsForbiddenOpName(node->op()) || IsStatefulOrSideEffectOp(*node) || !IsSupportedOpName(node->op()))
      continue;

    // Special-case multi-output ops (DynamicPartition): create one ShapeOnly
    // node per output and register them under keys "<name>:<index>" so that
    // input rewriting can select the correct replacement based on port suffix.
    if (node->op() == "DynamicPartition") {
      // Expect attribute num_partitions
      int64_t num_partitions = -1;
      auto ait = node->attr().find("num_partitions");
      if (ait != node->attr().end() && ait->second.has_i()) num_partitions = ait->second.i();
      if (num_partitions <= 0) continue;  // cannot handle
      for (int i = 0; i < num_partitions; ++i) {
        NodeDef so;
        so.set_op("ShapeOnly");
        string so_name = cand_name + "_shapeonly_" + std::to_string(i);
        so.set_name(so_name);
        // copy inputs
        for (const string& inp : node->input()) so.add_input(inp);
        // preserve dtype if present
        DataType dtype = DT_FLOAT;
        GetDtypeFromNodeDef(*node, &dtype);
        (*so.mutable_attr())["T"].set_type(dtype);
        // mark orig op and which output index this node represents
        (*so.mutable_attr())["orig_op"].set_s(node->op());
        (*so.mutable_attr())["output_index"].set_i(i);
        if (!node->device().empty()) so.set_device(node->device());
        // register under key with suffix so rewriting can pick exact node
        string key = cand_name + ":" + std::to_string(i);
        created_shapeonly[key] = so;
      }
      continue;
    }

    // Default single-output handling
    NodeDef so;
    so.set_op("ShapeOnly");
    string so_name = cand_name + "_shapeonly";
    so.set_name(so_name);
    // Copy inputs from the original node so the ShapeOnly node can inspect
    // input tensor shapes at runtime.
    for (const string& inp : node->input()) so.add_input(inp);
    // Preserve dtype attribute if present (used by ShapeOnly implementation).
    DataType dtype = DT_FLOAT;
    GetDtypeFromNodeDef(*node, &dtype);
    (*so.mutable_attr())["T"].set_type(dtype);
    // record original op
    (*so.mutable_attr())["orig_op"].set_s(node->op());
    // Preserve device placement (optional, but keeps semantics closer to original)
    if (!node->device().empty()) so.set_device(node->device());

    // register under basename key
    created_shapeonly[cand_name] = so;
  }

  if (created_shapeonly.empty()) {
    VLOG(1) << "ShapeOnlyOptimizer: no shapeonly nodes created for candidates.";
    return absl::OkStatus();
  }

  // Rebuild nodes: rewrite inputs that reference candidate -> to shapeonly nodes,
  // but only for consumers that are in the seeds (we are replacing nodes that only
  // feed shape computations).
  // To be conservative, we only retarget consumers that are shape-like (direct) or
  // any consumer that also belongs to candidates_ (transitive).
  for (const NodeDef& n : gdef->node()) {
    // If this node itself is a candidate that we replaced, skip adding original;
    if (/* original single-output candidate names without suffix */ false) {
      // we'll append created shapeonly nodes later
      // (we keep originals removed; created nodes appended at end)
    }
    NodeDef newn = n;  // copy
    for (int i = 0; i < newn.input_size(); ++i) {
      string orig_in = newn.input(i);
      string cleaned = orig_in;
      bool is_control = false;
      if (!cleaned.empty() && cleaned[0] == '^') {
        is_control = true;
        cleaned = cleaned.substr(1);
      }

      // preserve port suffix if present
      string suffix = "";
      size_t pos = cleaned.rfind(':');
      if (pos != string::npos) {
        bool digits = true;
        for (size_t j = pos + 1; j < cleaned.size(); ++j)
          if (!isdigit(cleaned[j])) { digits = false; break; }
        if (digits) suffix = cleaned.substr(pos);
      }

      string basename = NodeNameFromInput(cleaned);

      // Prefer mapping keyed by basename+suffix (for multi-output candidates),
      // falling back to basename for single-output replacements.
      string lookup_key = basename + suffix;
      auto it_created = created_shapeonly.find(lookup_key);
      bool matched_with_suffix = true;
      if (it_created == created_shapeonly.end()) {
        it_created = created_shapeonly.find(basename);
        matched_with_suffix = false;
      }
      if (it_created == created_shapeonly.end()) continue;

      // Only retarget if this consumer is inside the "seeds"/shape-consumers:
      // If the consumer (n) is shape-like OR is itself in candidates_ (we will replace it too),
      // it's safe to retarget. Otherwise skip.
      if (!IsShapeLikeOpName(n.op()) && !candidates_.contains(n.name())) {
        continue;
      }

      // Construct replacement input.
      string new_in;
      if (matched_with_suffix) {
        // created node already represents specific output; do not append suffix
        new_in = it_created->second.name();
      } else {
        // Single output created node — preserve original port suffix when present
        new_in = it_created->second.name() + suffix;
      }
      if (is_control) new_in = string("^") + new_in;
      newn.set_input(i, new_in);
    }
    // If this node itself was replaced (single-output), skip adding original.
    // Detect by name present in created_shapeonly under basename key (no suffix)
    if (created_shapeonly.find(n.name()) != created_shapeonly.end()) {
      // skip original; will append created nodes later
      continue;
    }
    result_nodes.push_back(std::move(newn));
  }

  // Append created ShapeOnly nodes
  for (const auto& kv : created_shapeonly) {
    result_nodes.push_back(kv.second);
  }

  // Build final GraphDef
  GraphDef out;
  out.mutable_library()->CopyFrom(gdef->library());
  for (const NodeDef& n : result_nodes) *out.add_node() = n;
  *gdef = std::move(out);

  VLOG(1) << "ShapeOnlyOptimizer: inserted " << created_shapeonly.size()
          << " ShapeOnly nodes and retargeted consumers.";
  return absl::OkStatus();
}

absl::Status ShapeOnlyOptimizer::Optimize(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* optimized_graph) {
  // Stage 0: Analyze and mark candidate nodes.
  AnalyzeCandidates(item.graph);
  VLOG(1) << "ShapeOnlyOptimizer: Stage0 found " << candidates_.size()
          << " candidate nodes (only-used-for-shape).";

  // Stage 2: Insert ShapeOnly nodes for candidates (safety checks applied).
  TF_RETURN_IF_ERROR(InsertShapeOnlyNodes(item, optimized_graph));
  VLOG(1) << "ShapeOnlyOptimizer: Stage2 ShapeOnly insert done.";

  return absl::OkStatus();
}

}  // end namespace grappler
}  // end namespace tensorflow