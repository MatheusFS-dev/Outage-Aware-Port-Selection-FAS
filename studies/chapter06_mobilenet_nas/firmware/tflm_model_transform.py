
# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))
import sys
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


SOURCE_ROOT = _COMPANION_ROOT / 'src'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from studies.chapter06_mobilenet_nas.scripts.train.generate_7_port_benchmark_models import (
    replace_int8_log_with_float,
)


BATCH_ONE_IMAGE_SHAPE = [1, 32, 32]


def serialize_tflite_model(model_object: schema.ModelT) -> bytes:
    """Serialize a mutable TensorFlow Lite model object.

    Args:
        model_object (schema.ModelT): Mutable TensorFlow Lite schema object
            containing the complete model, buffers, and subgraphs to encode.

    Returns:
        bytes: Serialized TensorFlow Lite flatbuffer with the ``TFL3`` file
        identifier.

    Raises:
        TypeError: If ``model_object`` is not a mutable TFLite model object.

    Examples:
        A model parsed with ``schema.ModelT.InitFromObj`` can be passed to this
        function and parsed again from the returned bytes.
    """
    if not isinstance(model_object, schema.ModelT):
        raise TypeError("model_object must be a schema.ModelT instance.")
    builder = flatbuffers.Builder(0)
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def replace_dynamic_batch_broadcasts(tflite_model: bytes) -> bytes:
    """Replace the two dynamic image-channel broadcasts with batch-one shapes.

    Args:
        tflite_model (bytes): Strict-INT8 per-regime selector flatbuffer. It
            must contain one subgraph and exactly two ``BROADCAST_TO`` operators
            whose inputs and outputs have shape ``[1, 32, 32]`` while their
            shared shape operand is dynamically produced at runtime.

    Returns:
        bytes: Equivalent batch-one flatbuffer in which both broadcasts read
        a constant INT32 ``[1, 32, 32]`` shape tensor. Trained weights,
        quantization parameters, public model I/O, and all other operators are
        unchanged.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If the model does not have the confirmed one-subgraph,
            two-broadcast, dynamic-shape deployment structure.

    Examples:
        ``replace_dynamic_batch_broadcasts(model_bytes)`` prepares a selected
        batch-one model for TFLM tensor allocation.
    """
    if not tflite_model:
        raise ValueError("tflite_model must not be empty.")

    model = schema.Model.GetRootAsModel(tflite_model, 0)
    model_object = schema.ModelT.InitFromObj(model)
    if len(model_object.subgraphs) != 1:
        raise RuntimeError("Expected exactly one TFLite subgraph.")

    subgraph = model_object.subgraphs[0]
    broadcast_opcode_indices = {
        index
        for index, code in enumerate(model_object.operatorCodes)
        if code.builtinCode == schema.BuiltinOperator.BROADCAST_TO
    }
    broadcast_operators = [
        operator
        for operator in subgraph.operators
        if operator.opcodeIndex in broadcast_opcode_indices
    ]
    if len(broadcast_operators) != 2:
        raise RuntimeError(
            f"Expected two BROADCAST_TO operators, found {len(broadcast_operators)}."
        )

    dynamic_shape_indices = set()
    for operator in broadcast_operators:
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            raise RuntimeError("Expected two-input, one-output BROADCAST_TO operators.")
        input_tensor = subgraph.tensors[int(operator.inputs[0])]
        shape_tensor = subgraph.tensors[int(operator.inputs[1])]
        output_tensor = subgraph.tensors[int(operator.outputs[0])]
        if list(input_tensor.shape) != BATCH_ONE_IMAGE_SHAPE:
            raise RuntimeError(
                f"Unexpected broadcast input shape: {list(input_tensor.shape)}."
            )
        if list(output_tensor.shape) != BATCH_ONE_IMAGE_SHAPE:
            raise RuntimeError(
                f"Unexpected broadcast output shape: {list(output_tensor.shape)}."
            )
        if shape_tensor.type != schema.TensorType.INT32 or list(shape_tensor.shape) != [3]:
            raise RuntimeError("Expected a three-element INT32 broadcast shape tensor.")
        shape_buffer = model_object.buffers[shape_tensor.buffer]
        if shape_buffer.data is not None and len(shape_buffer.data) > 0:
            raise RuntimeError("Expected a dynamically produced broadcast shape tensor.")
        dynamic_shape_indices.add(int(operator.inputs[1]))

    if len(dynamic_shape_indices) != 1:
        raise RuntimeError("Expected both broadcasts to share one dynamic shape tensor.")

    # TFLM validates BroadcastTo during graph preparation, before a SHAPE/PACK
    # producer can run. A dedicated constant keeps the deployment semantics
    # explicit without changing or removing the original training graph nodes.
    shape_buffer = schema.BufferT()
    shape_buffer.data = np.frombuffer(
        np.asarray(BATCH_ONE_IMAGE_SHAPE, dtype="<i4").tobytes(),
        dtype=np.uint8,
    )
    model_object.buffers.append(shape_buffer)
    shape_buffer_index = len(model_object.buffers) - 1

    constant_shape_tensor = schema.TensorT()
    constant_shape_tensor.shape = [3]
    constant_shape_tensor.shapeSignature = [3]
    constant_shape_tensor.type = schema.TensorType.INT32
    constant_shape_tensor.buffer = shape_buffer_index
    constant_shape_tensor.name = "tflm_batch_one_broadcast_shape"
    constant_shape_tensor.isVariable = False
    subgraph.tensors.append(constant_shape_tensor)
    constant_shape_index = len(subgraph.tensors) - 1

    for operator in broadcast_operators:
        operator.inputs = np.asarray(
            [int(operator.inputs[0]), constant_shape_index],
            dtype=np.int32,
        )

    return serialize_tflite_model(model_object)


def replace_int8_greater_with_float(tflite_model: bytes) -> bytes:
    """Replace the INT8 GREATER inputs with equivalent float inputs.

    Args:
        tflite_model (bytes): Strict-INT8 deployment flatbuffer containing
            exactly one two-input, one-output ``GREATER`` operator. Both
            comparison inputs must be INT8 tensors and the output must be a
            BOOL tensor.

    Returns:
        bytes: Equivalent flatbuffer with one ``DEQUANTIZE`` operator before
        each GREATER input. Public model input and output tensors remain INT8,
        while GREATER compares the corresponding dequantized real values.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If the model lacks the confirmed one-subgraph,
            one-GREATER INT8 structure or the DEQUANTIZE opcode.

    Examples:
        ``replace_int8_greater_with_float(model_bytes)`` avoids the TFLM INT8
        comparison multiplier limitation without changing public model I/O.
    """
    if not tflite_model:
        raise ValueError("tflite_model must not be empty.")

    model = schema.Model.GetRootAsModel(tflite_model, 0)
    model_object = schema.ModelT.InitFromObj(model)
    if len(model_object.subgraphs) != 1:
        raise RuntimeError("Expected exactly one TFLite subgraph.")

    subgraph = model_object.subgraphs[0]
    opcode_indices = {
        code.builtinCode: index
        for index, code in enumerate(model_object.operatorCodes)
    }
    if schema.BuiltinOperator.DEQUANTIZE not in opcode_indices:
        raise RuntimeError("Missing required DEQUANTIZE opcode.")

    greater_indices = [
        index
        for index, operator in enumerate(subgraph.operators)
        if (
            model_object.operatorCodes[operator.opcodeIndex].builtinCode
            == schema.BuiltinOperator.GREATER
        )
    ]
    if len(greater_indices) != 1:
        raise RuntimeError(
            f"Expected one GREATER operator, found {len(greater_indices)}."
        )

    greater_index = greater_indices[0]
    greater_operator = subgraph.operators[greater_index]
    if len(greater_operator.inputs) != 2 or len(greater_operator.outputs) != 1:
        raise RuntimeError("Expected GREATER to have two inputs and one output.")

    input_indices = [int(index) for index in greater_operator.inputs]
    input_tensors = [subgraph.tensors[index] for index in input_indices]
    output_tensor = subgraph.tensors[int(greater_operator.outputs[0])]
    if any(tensor.type != schema.TensorType.INT8 for tensor in input_tensors):
        raise RuntimeError("Expected GREATER to have two INT8 input tensors.")
    if output_tensor.type != schema.TensorType.BOOL:
        raise RuntimeError("Expected GREATER to have a BOOL output tensor.")

    float_input_indices = []
    dequantize_operators = []
    for input_number, (input_index, input_tensor) in enumerate(
        zip(input_indices, input_tensors, strict=True)
    ):
        float_input = schema.TensorT()
        float_input.shape = list(input_tensor.shape)
        float_input.shapeSignature = list(input_tensor.shape)
        float_input.type = schema.TensorType.FLOAT32
        float_input.buffer = 0
        float_input.name = f"tflm_greater_float_input_{input_number}"
        float_input.isVariable = False
        subgraph.tensors.append(float_input)
        float_input_index = len(subgraph.tensors) - 1
        float_input_indices.append(float_input_index)

        dequantize = schema.OperatorT()
        dequantize.opcodeIndex = opcode_indices[schema.BuiltinOperator.DEQUANTIZE]
        dequantize.inputs = np.asarray([input_index], dtype=np.int32)
        dequantize.outputs = np.asarray([float_input_index], dtype=np.int32)
        dequantize_operators.append(dequantize)

    # Comparing dequantized values implements the same real-value threshold
    # decision without entering TFLM's INT8-only multiplier preparation path.
    greater_operator.inputs = np.asarray(float_input_indices, dtype=np.int32)
    subgraph.operators = (
        subgraph.operators[:greater_index]
        + dequantize_operators
        + [greater_operator]
        + subgraph.operators[greater_index + 1 :]
    )
    return serialize_tflite_model(model_object)


def validate_tflm_deployment_model(tflite_model: bytes) -> None:
    """Validate constant broadcast shapes and float LOG adaptation for TFLM.

    Args:
        tflite_model (bytes): Transformed strict-INT8 deployment flatbuffer.
            It must expose INT8 model I/O, two constant batch-one broadcasts,
            one float ``LOG``, and one float-input ``GREATER`` comparison.

    Returns:
        None: Returns only when every required deployment invariant holds.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If the model structure, broadcast constants, LOG tensor
            types, or public INT8 I/O differ from the required deployment form.

    Examples:
        Calling ``validate_tflm_deployment_model(transformed_bytes)`` verifies
        the flatbuffer before firmware generation.
    """
    if not tflite_model:
        raise ValueError("tflite_model must not be empty.")

    model = schema.Model.GetRootAsModel(tflite_model, 0)
    model_object = schema.ModelT.InitFromObj(model)
    if len(model_object.subgraphs) != 1:
        raise RuntimeError("Expected exactly one TFLite subgraph.")
    subgraph = model_object.subgraphs[0]
    if len(subgraph.inputs) != 1 or len(subgraph.outputs) != 1:
        raise RuntimeError("Expected one model input and one model output.")
    if subgraph.tensors[int(subgraph.inputs[0])].type != schema.TensorType.INT8:
        raise RuntimeError("Expected an INT8 model input.")
    if subgraph.tensors[int(subgraph.outputs[0])].type != schema.TensorType.INT8:
        raise RuntimeError("Expected an INT8 model output.")

    operators_by_code: dict[int, list[schema.OperatorT]] = {}
    for operator in subgraph.operators:
        builtin_code = model_object.operatorCodes[operator.opcodeIndex].builtinCode
        operators_by_code.setdefault(builtin_code, []).append(operator)

    broadcast_operators = operators_by_code.get(schema.BuiltinOperator.BROADCAST_TO, [])
    if len(broadcast_operators) != 2:
        raise RuntimeError(
            f"Expected two BROADCAST_TO operators, found {len(broadcast_operators)}."
        )
    for operator in broadcast_operators:
        shape_tensor = subgraph.tensors[int(operator.inputs[1])]
        shape_data = model_object.buffers[shape_tensor.buffer].data
        if shape_data is None:
            raise RuntimeError("Expected a constant broadcast shape buffer.")
        decoded_shape = np.frombuffer(bytes(shape_data), dtype="<i4").tolist()
        if decoded_shape != BATCH_ONE_IMAGE_SHAPE:
            raise RuntimeError(f"Unexpected broadcast shape data: {decoded_shape}.")

    log_operators = operators_by_code.get(schema.BuiltinOperator.LOG, [])
    if len(log_operators) != 1:
        raise RuntimeError(f"Expected one LOG operator, found {len(log_operators)}.")
    log_operator = log_operators[0]
    log_input = subgraph.tensors[int(log_operator.inputs[0])]
    log_output = subgraph.tensors[int(log_operator.outputs[0])]
    if log_input.type != schema.TensorType.FLOAT32 or log_output.type != schema.TensorType.FLOAT32:
        raise RuntimeError("Expected LOG to use FLOAT32 input and output tensors.")

    greater_operators = operators_by_code.get(schema.BuiltinOperator.GREATER, [])
    if len(greater_operators) != 1:
        raise RuntimeError(
            f"Expected one GREATER operator, found {len(greater_operators)}."
        )
    greater_operator = greater_operators[0]
    greater_inputs = [
        subgraph.tensors[int(input_index)]
        for input_index in greater_operator.inputs
    ]
    if any(tensor.type != schema.TensorType.FLOAT32 for tensor in greater_inputs):
        raise RuntimeError("Expected GREATER to use two FLOAT32 input tensors.")
    greater_output = subgraph.tensors[int(greater_operator.outputs[0])]
    if greater_output.type != schema.TensorType.BOOL:
        raise RuntimeError("Expected GREATER to retain its BOOL output tensor.")


def prepare_tflm_deployment_model(tflite_model: bytes) -> bytes:
    """Apply the required batch-one, LOG, and GREATER transformations.

    Args:
        tflite_model (bytes): Original selected strict-INT8 TFLite artifact.
            The function changes only deployment graph wiring; trained and
            quantized weights remain unchanged.

    Returns:
        bytes: Validated TFLM-compatible strict-INT8 deployment flatbuffer.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If either confirmed source transformation cannot be
            applied or the resulting deployment model fails validation.

    Examples:
        ``prepare_tflm_deployment_model(selected_model.read_bytes())`` returns
        the bytes that should be embedded in Pico firmware.
    """
    transformed_model = replace_dynamic_batch_broadcasts(tflite_model)
    transformed_model = replace_int8_log_with_float(transformed_model)
    transformed_model = replace_int8_greater_with_float(transformed_model)
    validate_tflm_deployment_model(transformed_model)
    return transformed_model
