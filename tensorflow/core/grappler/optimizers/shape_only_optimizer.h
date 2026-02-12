// SPDX-License-Identifier: Apache-2.0
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SHAPE_ONLY_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SHAPE_ONLY_OPTIMIZER_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// Optimizer that finds nodes whose outputs are only used to compute shapes,
// then replaces them (or their shape consumers) with lightweight shape-only
// constructs to avoid expensive compute. It supports:
//  - Stage0: analysis and logging (mark candidates).
//  - Stage1: replacement of "Shape" nodes with Consts when the shape vector
//            can be fully inferred.
//  - Stage2: replacement of marked upstream nodes with "ShapeOnly" lightweight
//            nodes (skeleton).
class ShapeOnlyOptimizer : public GraphOptimizer {
 public:
  ShapeOnlyOptimizer() {}
  ~ShapeOnlyOptimizer() override {}

  std::string name() const override { return "shape_only_optimizer"; }
  bool UsesFunctionLibrary() const override { return true; }

  // Optimize the provided grappler item, writing the optimized graph to
  // optimized_graph. The optimizer currently includes multiple stages which
  // can be toggled via constructor / flags (for PoC we enable them all).
  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

  // For testing / debug, expose the set of found candidates.
  const absl::flat_hash_set<string>& Candidates() const { return candidates_; }

 private:
  // Helper to populate candidates_ (stage 0).
  void AnalyzeCandidates(const GraphDef& gdef);

  // Helper to attempt Shape->Const replacements (stage 1).
  absl::Status ReplaceShapeWithConst(const GrapplerItem& item,
                                     GraphDef* gdef);

  // Helper to perform ShapeOnly insertion for candidates (stage 2).
  absl::Status InsertShapeOnlyNodes(const GrapplerItem& item, GraphDef* gdef);

  // All nodes we consider "shape-like" for the purpose of tracing.
  bool IsShapeLikeOp(const string& op) const;

  absl::flat_hash_set<string> candidates_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SHAPE_ONLY_OPTIMIZER_H_