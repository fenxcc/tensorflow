// SPDX-License-Identifier: Apache-2.0
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ShapeOnly")
    .Attr("Tin: list(type) >= 1")
    .Attr("orig_op: string = \"\"")
    .Attr("shape: shape = {}")
    .Attr("T: type = DT_FLOAT")
    .Input("inputs: Tin")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // If optimizer provided a 'shape' attribute, use it as the output shape.
      const AttrValue* shape_attr = c->GetAttr("shape");
      if (shape_attr != nullptr && shape_attr->has_shape()) {
        shape_inference::ShapeHandle sh;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_attr->shape(), &sh));
        c->set_output(0, sh);
        return absl::OkStatus();
      }
      // Otherwise conservatively return unknown shape. Runtime kernel computes
      // the actual shape from input tensor shapes.
      c->set_output(0, c->UnknownShape());
      return absl::OkStatus();
    });