#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
// Needed for IsRefType
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

template <typename T>
class ShapeOnlyOp : public OpKernel {
 public:
  explicit ShapeOnlyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Read attribute 'shape' if present
    TensorShapeProto proto;
    if (ctx->GetAttr("shape", &proto).ok()) {
      shape_ = TensorShape(proto);
    } else {
      // unknown shape: keep empty
      shape_ = TensorShape();
    }
    // Read orig_op if present to dispatch runtime logic
    if (!ctx->GetAttr("orig_op", &orig_op_).ok()) orig_op_.clear();

    // Read optional keep_dims attribute for reduce-like ops (if present)
    bool keep_dims = false;
    if (ctx->GetAttr("keep_dims", &keep_dims).ok()) keep_dims_ = keep_dims;
    else keep_dims_ = false;

    // Read optional output_index (used for DynamicPartition per-output ShapeOnly)
    output_index_ = -1;
    (void)ctx->GetAttr("output_index", &output_index_);
  }

  void Compute(OpKernelContext* ctx) override {
    // If optimizer provided an explicit 'shape' attr, honor it.
    if (shape_.dims() > 0 || (shape_.dims() == 0 && shape_.num_elements() > 0)) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out));
      return;
    }

    Tensor* out = nullptr;
    TensorShape out_shape;

    // Dispatch based on orig_op
    if (!orig_op_.empty()) {
      if (IsPassthroughOp(orig_op_)) {
        if (ctx->num_inputs() >= 1) {
          // If input is a ref type, forward as ref; otherwise set output to
          // reference the same Tensor (no copy).
          if (IsRefType(ctx->input_dtype(0))) {
            ctx->forward_ref_input_to_ref_output(0, 0);
            return;
          } else {
            ctx->set_output(0, ctx->input(0));
            return;
          }
        }
      } else if (IsBroadcastOp(orig_op_)) {
        Status s = ComputeBroadcastShape(ctx, &out_shape);
        if (!s.ok()) {
          // On shape mismatch, raise error to avoid silent incorrect shapes.
          OP_REQUIRES(ctx, false, s);
          return;
        }
      } else if (orig_op_ == "Reshape") {
        OP_REQUIRES_OK(ctx, ComputeReshapeShape(ctx, &out_shape));
      } else if (orig_op_ == "GatherV2") {
        OP_REQUIRES_OK(ctx, ComputeGatherV2Shape(ctx, &out_shape));
      } else if (orig_op_ == "ConcatV2") {
        OP_REQUIRES_OK(ctx, ComputeConcatV2Shape(ctx, &out_shape));
      } else if (orig_op_ == "ParallelDynamicStitch") {
        OP_REQUIRES_OK(ctx, ComputeParallelDynamicStitchShape(ctx, &out_shape));
      } else if (orig_op_ == "DynamicPartition") {
        OP_REQUIRES_OK(ctx, ComputeDynamicPartitionShape(ctx, &out_shape, output_index_));
      } else if (orig_op_ == "SparseSegmentMean" || orig_op_ == "SparseSegmentSum") {
        OP_REQUIRES_OK(ctx, ComputeSparseSegmentShape(ctx, &out_shape));
      } else if (IsReduceOp(orig_op_)) {
        // Determine reduce axes from second input (if present) or reduce all dims.
        if (ctx->num_inputs() >= 1) {
          const TensorShape in_shape = ctx->input(0).shape();
          const int rank = in_shape.dims();
          std::vector<int64_t> axes_vec;
          const int axes_input_idx = 1;  // common convention: input 1 = axes

          if (ctx->num_inputs() > axes_input_idx) {
            const Tensor& axes_t = ctx->input(axes_input_idx);
            OP_REQUIRES(ctx, axes_t.IsInitialized(), errors::InvalidArgument("Axes input is uninitialized"));
            if (axes_t.dtype() == DT_INT64) {
              auto flat = axes_t.flat<int64_t>();
              axes_vec.reserve(flat.size());
              for (int i = 0; i < flat.size(); ++i) axes_vec.push_back(flat(i));
            } else if (axes_t.dtype() == DT_INT32) {
              auto flat = axes_t.flat<int32>();
              axes_vec.reserve(flat.size());
              for (int i = 0; i < flat.size(); ++i) axes_vec.push_back(static_cast<int64_t>(flat(i)));
            } else {
              OP_REQUIRES(ctx, false, errors::InvalidArgument("Reduce axes must be int32 or int64"));
              return;
            }
          } else {
            // No axes input: reduce over all dims
            axes_vec.resize(rank);
            for (int i = 0; i < rank; ++i) axes_vec[i] = i;
          }

          // Normalize axes and validate
          std::vector<char> reduce_mask(rank, 0);
          for (int64_t raw_ax : axes_vec) {
            int64_t ax = raw_ax < 0 ? raw_ax + rank : raw_ax;
            OP_REQUIRES(ctx, ax >= 0 && ax < rank,
                        errors::InvalidArgument("Reduce axis out of range: ", raw_ax));
            reduce_mask[ax] = 1;
          }

          // Build output shape depending on keep_dims_
          if (keep_dims_) {
            TensorShape out;
            for (int i = 0; i < rank; ++i) {
              if (reduce_mask[i]) out.AddDim(1);
              else out.AddDim(in_shape.dim_size(i));
            }
            out_shape = out;
          } else {
            TensorShape out;
            for (int i = 0; i < rank; ++i) if (!reduce_mask[i]) out.AddDim(in_shape.dim_size(i));
            out_shape = out;
          }
        }
      } else {
        // Unknown orig_op: fallback to first input shape if available.
        if (ctx->num_inputs() >= 1) out_shape = ctx->input(0).shape();
      }
    } else {
      // No orig_op provided: fallback to first input shape if available.
      if (ctx->num_inputs() >= 1) out_shape = ctx->input(0).shape();
    }

    // If still empty, fallback to scalar to be safe.
    if (out_shape.dims() == 0 && out_shape.num_elements() == 0) out_shape = TensorShape({});

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    // Do not fill data; leave uninitialized for performance.
  }

 private:
  TensorShape shape_;
  string orig_op_;
  bool keep_dims_ = false;
  int output_index_ = -1;

  static bool IsPassthroughOp(const string& op) {
    static const absl::flat_hash_set<string> kPassthrough = {"Identity", "Relu", "Neg", "Abs", "Sigmoid", "Tanh"};
    return kPassthrough.contains(op);
  }

  static bool IsBroadcastOp(const string& op) {
    static const absl::flat_hash_set<string> kBroadcast = {"Add", "Mul", "Sub", "Div", "Maximum", "Minimum", "FloorMod"};
    return kBroadcast.contains(op);
  }

  static bool IsReduceOp(const string& op) {
    static const absl::flat_hash_set<string> kReduce = {"Sum", "Mean", "ReduceSum", "ReduceMean"};
    return kReduce.contains(op);
  }

  // Compute broadcasted shape across all inputs following numpy/TensorFlow rules.
  static Status ComputeBroadcastShape(OpKernelContext* ctx, TensorShape* out_shape) {
    int n = ctx->num_inputs();
    if (n == 0) {
      *out_shape = TensorShape({});
      return Status::OK();
    }
    // Start from first input
    TensorShape result = ctx->input(0).shape();
    for (int i = 1; i < n; ++i) {
      const TensorShape& s = ctx->input(i).shape();
      // Align from back
      int r = result.dims();
      int sdim = s.dims();
      int out_dims = std::max(r, sdim);
      std::vector<int64_t> dims(out_dims, 0);
      for (int d = 0; d < out_dims; ++d) {
        int idx_r = r - 1 - d;
        int idx_s = sdim - 1 - d;
        int64_t dr = (idx_r >= 0) ? result.dim_size(idx_r) : 1;
        int64_t ds = (idx_s >= 0) ? s.dim_size(idx_s) : 1;
        if (dr == ds || dr == 1 || ds == 1) {
          dims[out_dims - 1 - d] = std::max(dr, ds);
        } else {
          return errors::InvalidArgument("Incompatible shapes for broadcasting: ", result.DebugString(), " vs ", s.DebugString());
        }
      }
      // Build new result
      TensorShape newr;
      for (int64_t v : dims) newr.AddDim(v);
      result = newr;
    }
    *out_shape = result;
    return Status::OK();
  }

  // Compute shape for Reshape: input 0 = data, input 1 = shape tensor
  static Status ComputeReshapeShape(OpKernelContext* ctx, TensorShape* out_shape) {
    if (ctx->num_inputs() < 2) return errors::InvalidArgument("Reshape requires shape input");
    const TensorShape in_shape = ctx->input(0).shape();
    const Tensor& shape_t = ctx->input(1);
    if (!(shape_t.dtype() == DT_INT32 || shape_t.dtype() == DT_INT64)) {
      return errors::InvalidArgument("Reshape shape must be int32 or int64");
    }
    std::vector<int64_t> shape_vec;
    if (shape_t.dtype() == DT_INT64) {
      auto flat = shape_t.flat<int64_t>();
      shape_vec.reserve(flat.size());
      for (int i = 0; i < flat.size(); ++i) shape_vec.push_back(flat(i));
    } else {
      auto flat = shape_t.flat<int32>();
      shape_vec.reserve(flat.size());
      for (int i = 0; i < flat.size(); ++i) shape_vec.push_back(static_cast<int64_t>(flat(i)));
    }
    // Compute product of known dims of input
    int64_t known_product = 1;
    bool has_unknown = false;
    for (int i = 0; i < in_shape.dims(); ++i) {
      int64_t d = in_shape.dim_size(i);
      if (d <= 0) { has_unknown = true; break; }
      known_product *= d;
    }
    int64_t minus_one_index = -1;
    int64_t out_elems = 1;
    for (size_t i = 0; i < shape_vec.size(); ++i) {
      int64_t v = shape_vec[i];
      if (v == -1) {
        if (minus_one_index != -1) return errors::InvalidArgument("Only one -1 allowed in reshape shape");
        minus_one_index = static_cast<int64_t>(i);
      } else if (v == 0) {
        // copy from input if possible
        if (static_cast<int>(i) < in_shape.dims()) {
          int64_t d = in_shape.dim_size(i);
          if (d <= 0) return errors::InvalidArgument("Cannot copy unknown dim for 0 in reshape");
          out_elems *= d;
        } else {
          return errors::InvalidArgument("0 in reshape refers to out-of-range input dim");
        }
      } else if (v > 0) {
        out_elems *= v;
      } else {
        return errors::InvalidArgument("Invalid reshape dim: ", v);
      }
    }
    if (minus_one_index != -1) {
      if (has_unknown) return errors::InvalidArgument("Cannot infer -1 when input has unknown dim");
      if (out_elems == 0) return errors::InvalidArgument("Invalid reshape target");
      int64_t inferred = known_product / out_elems;
      shape_vec[minus_one_index] = inferred;
    }
    TensorShape out;
    for (int64_t v : shape_vec) out.AddDim(v);
    *out_shape = out;
    return Status::OK();
  }

  // Compute shape for GatherV2: params, indices, axis
  static Status ComputeGatherV2Shape(OpKernelContext* ctx, TensorShape* out_shape) {
    if (ctx->num_inputs() < 2) return errors::InvalidArgument("GatherV2 requires params and indices");
    const TensorShape params_shape = ctx->input(0).shape();
    const TensorShape indices_shape = ctx->input(1).shape();
    // axis may be input 2 or attr; try input
    int64_t axis = 0;
    if (ctx->num_inputs() > 2) {
      const Tensor& axis_t = ctx->input(2);
      if (!(axis_t.dtype() == DT_INT32 || axis_t.dtype() == DT_INT64)) return errors::InvalidArgument("axis must be int");
      if (axis_t.NumElements() != 1) return errors::InvalidArgument("axis must be scalar");
      if (axis_t.dtype() == DT_INT64) axis = axis_t.scalar<int64_t>()(); else axis = axis_t.scalar<int>()();
    } else {
      // default axis = 0
      axis = 0;
    }
    int64_t rank = params_shape.dims();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return errors::InvalidArgument("axis out of range");
    // output shape = indices.shape + params.shape[axis+1:]
    TensorShape out;
    for (int i = 0; i < indices_shape.dims(); ++i) out.AddDim(indices_shape.dim_size(i));
    for (int i = axis + 1; i < rank; ++i) out.AddDim(params_shape.dim_size(i));
    *out_shape = out;
    return Status::OK();
  }

  // Compute ConcatV2 shape: values..., axis is last input
  static Status ComputeConcatV2Shape(OpKernelContext* ctx, TensorShape* out_shape) {
    int n = ctx->num_inputs();
    if (n < 2) return errors::InvalidArgument("ConcatV2 requires at least one value and axis");
    const Tensor& axis_t = ctx->input(n - 1);
    if (!(axis_t.dtype() == DT_INT32 || axis_t.dtype() == DT_INT64)) return errors::InvalidArgument("axis must be int");
    if (axis_t.NumElements() != 1) return errors::InvalidArgument("axis must be scalar");
    int64_t axis = (axis_t.dtype() == DT_INT64) ? axis_t.scalar<int64_t>()() : axis_t.scalar<int>()();
    const TensorShape& first_shape = ctx->input(0).shape();
    int rank = first_shape.dims();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return errors::InvalidArgument("axis out of range");
    int64_t summed = 0;
    for (int i = 0; i < n - 1; ++i) {
      const TensorShape& s = ctx->input(i).shape();
      if (s.dims() != rank) return errors::InvalidArgument("All inputs to ConcatV2 must have same rank");
      int64_t dim = s.dim_size(axis);
      OP_REQUIRES(ctx, dim >= 0, errors::InvalidArgument("Concat axis dim unknown"));
      summed += dim;
    }
    TensorShape out = first_shape;
    out.set_dim(axis, summed);
    *out_shape = out;
    return Status::OK();
  }

  // Compute ParallelDynamicStitch shape: assume inputs are [indices_0..indices_{k-1}, data_0..data_{k-1}]
  static Status ComputeParallelDynamicStitchShape(OpKernelContext* ctx, TensorShape* out_shape) {
    int n = ctx->num_inputs();
    if (n == 0) return errors::InvalidArgument("ParallelDynamicStitch needs inputs");
    // need even number: k indices then k data
    if (n % 2 != 0) return errors::InvalidArgument("Expected pairs of indices and data");
    int k = n / 2;
    int64_t max_index = -1;
    TensorShape data_elem_shape;
    for (int i = 0; i < k; ++i) {
      const Tensor& idx = ctx->input(i);
      if (!(idx.dtype() == DT_INT32 || idx.dtype() == DT_INT64)) return errors::InvalidArgument("indices must be int");
      if (idx.NumElements() > 0) {
        if (idx.dtype() == DT_INT64) {
          auto flat = idx.flat<int64_t>();
          for (int j = 0; j < flat.size(); ++j) max_index = std::max(max_index, static_cast<int64_t>(flat(j)));
        } else {
          auto flat = idx.flat<int32>();
          for (int j = 0; j < flat.size(); ++j) max_index = std::max(max_index, static_cast<int64_t>(flat(j)));
        }
      }
      const TensorShape& dshape = ctx->input(k + i).shape();
      if (i == 0) {
        // element shape is dshape[1:]
        for (int t = 1; t < dshape.dims(); ++t) data_elem_shape.AddDim(dshape.dim_size(t));
      } else {
        // verify same element shape
        for (int t = 1; t < dshape.dims(); ++t) {
          if (dshape.dim_size(t) != data_elem_shape.dim_size(t - 1)) return errors::InvalidArgument("Data inputs must have same shape beyond first dim");
        }
      }
    }
    if (max_index < 0) {
      // no indices -> empty output first dim
      *out_shape = TensorShape({0});
      // append element shape
      for (int i = 0; i < data_elem_shape.dims(); ++i) out_shape->AddDim(data_elem_shape.dim_size(i));
      return Status::OK();
    }
    TensorShape out;
    out.AddDim(max_index + 1);
    for (int i = 0; i < data_elem_shape.dims(); ++i) out.AddDim(data_elem_shape.dim_size(i));
    *out_shape = out;
    return Status::OK();
  }

  // Compute DynamicPartition shape for a specific partition id provided via attr "output_index"
  static Status ComputeDynamicPartitionShape(OpKernelContext* ctx, TensorShape* out_shape, int output_index) {
    // Need provided output_index and input 1 = partitions
    if (output_index < 0) return errors::InvalidArgument("ShapeOnly for DynamicPartition requires 'output_index' attr");
    if (ctx->num_inputs() < 2) return errors::InvalidArgument("DynamicPartition requires data and partitions");
    const Tensor& partitions = ctx->input(1);
    if (!(partitions.dtype() == DT_INT32 || partitions.dtype() == DT_INT64)) return errors::InvalidArgument("partitions must be int");
    int64_t count = 0;
    if (partitions.dtype() == DT_INT64) {
      auto flat = partitions.flat<int64_t>();
      for (int i = 0; i < flat.size(); ++i) if (flat(i) == output_index) ++count;
    } else {
      auto flat = partitions.flat<int32>();
      for (int i = 0; i < flat.size(); ++i) if (flat(i) == output_index) ++count;
    }
    const TensorShape& data_shape = ctx->input(0).shape();
    TensorShape out;
    out.AddDim(count);
    for (int i = 1; i < data_shape.dims(); ++i) out.AddDim(data_shape.dim_size(i));
    *out_shape = out;
    return Status::OK();
  }

  // Compute SparseSegment* output shape: data, indices, segment_ids
  static Status ComputeSparseSegmentShape(OpKernelContext* ctx, TensorShape* out_shape) {
    if (ctx->num_inputs() < 3) return errors::InvalidArgument("SparseSegment ops require data, indices, segment_ids");
    const Tensor& seg = ctx->input(2);
    if (!(seg.dtype() == DT_INT32 || seg.dtype() == DT_INT64)) return errors::InvalidArgument("segment_ids must be int");
    int64_t max_id = -1;
    if (seg.dtype() == DT_INT64) {
      auto flat = seg.flat<int64_t>();
      for (int i = 0; i < flat.size(); ++i) max_id = std::max(max_id, static_cast<int64_t>(flat(i)));
    } else {
      auto flat = seg.flat<int32>();
      for (int i = 0; i < flat.size(); ++i) max_id = std::max(max_id, static_cast<int64_t>(flat(i)));
    }
    if (max_id < 0) {
      *out_shape = TensorShape({0});
      return Status::OK();
    }
    const TensorShape& data_shape = ctx->input(0).shape();
    TensorShape out;
    out.AddDim(max_id + 1);
    for (int i = 1; i < data_shape.dims(); ++i) out.AddDim(data_shape.dim_size(i));
    *out_shape = out;
    return Status::OK();
  }
}