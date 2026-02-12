// SPDX-License-Identifier: Apache-2.0
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ShapeOnly")
    .Attr("N: int >= 0")
    .Attr("Tin: list(type) = []")
    .Attr("orig_op: string = \"\"")
    .Attr("shape: shape = {}")
    .Attr("T: type = DT_FLOAT")
    .Input("inputs: N * Tin")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // If optimizer provided a 'shape' attribute, use it as the output shape.
      const TensorShapeProto* shape_proto = c->getAttrOfType<TensorShapeProto>("shape");
      if (shape_proto) {
        shape_inference::ShapeHandle sh;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(*shape_proto, &sh));
        c->set_output(0, sh);
        return Status::OK();
      }
      // Otherwise conservatively return unknown shape. Runtime kernel computes
      // the actual shape from input tensor shapes.
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });