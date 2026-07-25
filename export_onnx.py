#!/usr/bin/env python3
"""Export the SAM3 tokenizer, text/image encoders, and text-prompt decoder."""

import argparse
import gc
import gzip
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16
from torch import nn

from sam3.model.model_misc import inverse_sigmoid
from sam3.model_builder import build_sam3_image_model

IR_VERSION = 12
SEQUENCE_OPS = {
    "ConcatFromSequence",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SplitToSequence",
}


class TextEncoder(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.encoder, self.resizer = text.encoder, text.resizer

    def forward(self, input_ids):
        _, tokens = self.encoder(input_ids)
        return self.resizer(tokens).transpose(0, 1), input_ids.eq(0)


class ImageEncoder(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.trunk, self.convs = vision.trunk, vision.convs

    def forward(self, image):
        x = self.trunk(image)[-1]
        return self.convs[0](x), self.convs[1](x), self.convs[2](x)


class ExportMlp(nn.Module):
    """Provider-neutral form of SAM3's fused ViT MLP."""

    def __init__(self, mlp):
        super().__init__()
        self.fc1, self.act, self.drop1 = mlp.fc1, mlp.act, mlp.drop1
        self.norm, self.fc2, self.drop2 = mlp.norm, mlp.fc2, mlp.drop2

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.norm(x)
        return self.drop2(self.fc2(x))


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pos = model.backbone.vision_backbone.position_encoding
        self.geometry = model.geometry_encoder
        self.encoder = model.transformer.encoder
        self.decoder = model.transformer.decoder
        self.scorer = model.dot_prod_scoring
        self.segmentation = model.segmentation_head
        self.register_buffer("image_id", torch.zeros(1, dtype=torch.long))

    def forward(self, feature_0, feature_1, feature_2, text, text_padding_mask):
        image = feature_2.flatten(2).permute(2, 0, 1)
        image_pos = self.pos(feature_2).to(feature_2.dtype).flatten(2).permute(2, 0, 1)

        geo = self.geometry.cls_embed.weight.view(1, 1, -1)
        geo = self.geometry.norm(self.geometry.final_proj(geo))
        geo_mask = torch.zeros((1, 1), dtype=torch.bool, device=geo.device)
        for layer in self.geometry.encode:
            geo = layer(
                tgt=geo,
                memory=image,
                tgt_key_padding_mask=geo_mask,
                pos=image_pos,
            )
        geo = self.geometry.encode_norm(geo)
        prompt = torch.cat((text, geo))
        prompt_mask = torch.cat((text_padding_mask, geo_mask), dim=1)

        encoded = self.encoder(
            src=[image],
            src_pos=[image_pos],
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            prompt_pos=torch.zeros_like(prompt),
            feat_sizes=[feature_2.shape[-2:]],
        )
        memory = encoded["memory"]
        target = self.decoder.query_embed.weight[:, None].expand(-1, 1, -1)
        states, references, presence, _ = self.decoder(
            tgt=target,
            memory=memory,
            memory_key_padding_mask=encoded["padding_mask"],
            pos=encoded["pos_embed"],
            level_start_index=encoded["level_start_index"],
            spatial_shapes=encoded["spatial_shapes"],
            valid_ratios=encoded["valid_ratios"],
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        states = states.transpose(1, 2)
        references = references.transpose(1, 2)
        logits = self.scorer(states, prompt, prompt_mask)[-1]
        boxes = (
            inverse_sigmoid(references)
            + self.decoder.bbox_embed(states)
        ).sigmoid()[-1]
        masks = self.segmentation(
            backbone_feats=[feature_0, feature_1, feature_2],
            obj_queries=states,
            image_ids=self.image_id,
            encoder_hidden_states=memory,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )["pred_masks"]
        return masks, boxes, logits, presence.transpose(1, 2)[-1]


def export_tokenizer(path, tokenizer, bpe_path):
    vocab = dict(tokenizer.encoder)
    vocab["<|startoftext|>"] = tokenizer.sot_token_id
    vocab["<|endoftext|>"] = tokenizer.eot_token_id
    lines = gzip.open(bpe_path, "rt", encoding="utf-8").read().splitlines()
    merges = ("\n".join(lines[: 49152 - 256 - 2 + 1]) + "\n").encode()

    nodes = [
        helper.make_node(
            "StringECMARegexReplace",
            ["text", "whitespace", "space"],
            ["clean_text"],
            domain="ai.onnx.contrib",
            global_replace=1,
        ),
        helper.make_node(
            "CLIPTokenizer",
            ["clean_text"],
            ["raw_ids", "raw_mask"],
            domain="ai.onnx.contrib",
            vocab=json.dumps(vocab, ensure_ascii=False).encode(),
            merges=merges,
            padding_length=-1,
        ),
        helper.make_node("Pad", ["raw_ids", "pads", "zero"], ["padded_ids"]),
        helper.make_node("Pad", ["raw_mask", "pads", "zero"], ["padded_mask"]),
        helper.make_node(
            "Slice", ["padded_ids", "starts", "ends", "axes"], ["ids_32"]
        ),
        helper.make_node(
            "Slice", ["padded_mask", "starts", "ends", "axes"], ["mask_32"]
        ),
        helper.make_node("Shape", ["raw_ids"], ["raw_shape"]),
        helper.make_node("Gather", ["raw_shape", "one"], ["raw_length"], axis=0),
        helper.make_node("Greater", ["raw_length", "length"], ["truncated"]),
        helper.make_node(
            "Slice", ["ids_32", "starts", "prefix_ends", "axes"], ["prefix"]
        ),
        helper.make_node("Gather", ["ids_32", "last"], ["old_last"], axis=1),
        helper.make_node("Where", ["truncated", "eot", "old_last"], ["new_last"]),
        helper.make_node("Concat", ["prefix", "new_last"], ["truncated_ids"], axis=1),
        helper.make_node("Cast", ["mask_32"], ["valid"], to=TensorProto.BOOL),
        helper.make_node(
            "Where", ["valid", "truncated_ids", "zero"], ["input_ids"]
        ),
    ]
    constants = {
        "pads": ([4], [0, 0, 0, 32]),
        "zero": ([], [0]),
        "starts": ([2], [0, 0]),
        "ends": ([2], [1, 32]),
        "axes": ([2], [0, 1]),
        "one": ([1], [1]),
        "length": ([], [32]),
        "prefix_ends": ([2], [1, 31]),
        "last": ([1], [31]),
        "eot": ([], [tokenizer.eot_token_id]),
    }
    initializers = [
        helper.make_tensor(name, TensorProto.INT64, shape, values)
        for name, (shape, values) in constants.items()
    ]
    initializers += [
        helper.make_tensor("whitespace", TensorProto.STRING, [1], [rb"\s+"]),
        helper.make_tensor("space", TensorProto.STRING, [1], [b" "]),
    ]
    graph = helper.make_graph(
        nodes,
        "sam3_tokenizer",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [1])],
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 32])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="SAM3 ONNX exporter",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.contrib", 1),
        ],
    )
    model.metadata_props.add(
        key="runtime_requirement", value="onnxruntime-extensions"
    )
    model.metadata_props.add(key="precision", value="not applicable")
    model.metadata_props.add(key="target_runtime", value="TensorRT 10.13")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def prepare_vision(vision, height, width, device):
    patch_h, patch_w = height // 14, width // 14
    for block in vision.trunk.blocks:
        block.mlp = ExportMlp(block.mlp)
        attention = block.attn
        if not attention.use_rope:
            continue
        if block.window_size:
            freqs = attention.freqs_cis.to(device)
        else:
            dim = attention.head_dim
            base = 1.0 / (
                attention.rope_theta
                ** (torch.arange(0, dim, 4, device=device).float() / dim)
            )
            x = torch.arange(patch_w, device=device).repeat(patch_h)
            y = torch.arange(patch_h, device=device).repeat_interleave(patch_w)
            x = torch.outer(x * (attention.rope_pt_size[0] / patch_w), base)
            y = torch.outer(y * (attention.rope_pt_size[1] / patch_h), base)
            freqs = torch.cat(
                (
                    torch.polar(torch.ones_like(x), x),
                    torch.polar(torch.ones_like(y), y),
                ),
                dim=-1,
            )
            attention.freqs_cis = freqs
        attention.use_rope_real = True
        if hasattr(attention, "freqs_cis_real"):
            attention.freqs_cis_real = freqs.real
            attention.freqs_cis_imag = freqs.imag
        else:
            attention.register_buffer("freqs_cis_real", freqs.real)
            attention.register_buffer("freqs_cis_imag", freqs.imag)

    return patch_h, patch_w


def constant_values(model):
    values = {}
    for tensor in model.graph.initializer:
        try:
            values[tensor.name] = numpy_helper.to_array(tensor)
        except (ValueError, onnx.onnx_cpp2py_export.checker.ValidationError):
            # Large initializers may intentionally remain in external data.
            pass
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = next((attr.t for attr in node.attribute if attr.name == "value"), None)
        if value is not None:
            values[node.output[0]] = numpy_helper.to_array(value)
    return values


def replace_sequences(model):
    """Replace static SplitToSequence/SequenceAt patterns with tensor Split."""
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    values = constant_values(model)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    replacements, removed, added = {}, set(), []

    for node in model.graph.node:
        if node.op_type != "SplitToSequence":
            continue
        uses = consumers.get(node.output[0], [])
        if not uses or any(use.op_type != "SequenceAt" for use in uses):
            raise RuntimeError(f"Cannot safely lower sequence node {node.name}")
        keepdims = next(
            (
                helper.get_attribute_value(attr)
                for attr in node.attribute
                if attr.name == "keepdims"
            ),
            1,
        )
        if not keepdims:
            raise RuntimeError(f"Cannot safely lower keepdims=0 node {node.name}")

        indexed_uses = sorted(
            (int(values[use.input[1]].reshape(())), use) for use in uses
        )
        indices = [index for index, _ in indexed_uses]
        if indices != list(range(len(indices))):
            raise RuntimeError(f"Sequence outputs are not exhaustive in {node.name}")

        axis = next(
            (
                helper.get_attribute_value(attr)
                for attr in node.attribute
                if attr.name == "axis"
            ),
            0,
        )
        split = values[node.input[1]]
        split_input = node.input[1]
        if split.ndim == 0:
            source = initializers.get(node.input[0])
            if source is None:
                raise RuntimeError(f"Sequence input is not static in {node.name}")
            axis %= len(source.dims)
            dimension, chunk = source.dims[axis], int(split)
            count = math.ceil(dimension / chunk)
            if count != len(indexed_uses):
                raise RuntimeError(f"Unexpected split count in {node.name}")
            sizes = [chunk] * (count - 1) + [dimension - chunk * (count - 1)]
            split_input = f"{node.name}_sizes"
            added.append(
                numpy_helper.from_array(np.asarray(sizes, dtype=np.int64), split_input)
            )
        elif split.ndim != 1 or len(split) != len(indexed_uses):
            raise RuntimeError(f"Unexpected split sizes in {node.name}")

        outputs = [use.output[0] for _, use in indexed_uses]
        replacements[id(node)] = helper.make_node(
            "Split",
            [node.input[0], split_input],
            outputs,
            name=f"{node.name}_tensor",
            axis=axis,
        )
        removed.update(id(use) for _, use in indexed_uses)

    removed.update(replacements)
    nodes = []
    for node in model.graph.node:
        if id(node) in replacements:
            nodes.append(replacements[id(node)])
        elif id(node) not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(added)

    remaining = [
        node.op_type for node in model.graph.node if node.op_type in SEQUENCE_OPS
    ]
    if remaining:
        raise RuntimeError(f"Unsupported sequence operators remain: {remaining}")
    return len(replacements)


def assign_node_names(model):
    seen = set()
    for index, node in enumerate(model.graph.node):
        if not node.name or node.name in seen:
            node.name = f"{node.op_type}_{index}"
        seen.add(node.name)


def replace_castlike(model):
    """Lower CastLike to Cast when the target tensor type is static."""
    typed_values = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.HasField("tensor_type"):
            typed_values[value.name] = value.type.tensor_type.elem_type
    typed_values.update(
        (tensor.name, tensor.data_type) for tensor in model.graph.initializer
    )

    count = 0
    for node in model.graph.node:
        if node.op_type != "CastLike":
            continue
        # SAM3's CastLike nodes cast scalar constants to floating activations.
        # A few attention targets lack optional value_info in PyTorch's graph,
        # but the fixed model interface guarantees FLOAT for the FP32 graph.
        target_type = typed_values.get(node.input[1], TensorProto.FLOAT)
        del node.input[1:]
        del node.attribute[:]
        node.op_type = "Cast"
        node.attribute.extend([helper.make_attribute("to", target_type)])
        count += 1
    return count


def eliminate_identities(model):
    """Remove Identity nodes that do not provide a public graph output name."""
    graph_outputs = {output.name for output in model.graph.output}
    aliases = {}
    nodes = []

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        if (
            node.op_type == "Identity"
            and len(node.input) == len(node.output) == 1
            and node.output[0] not in graph_outputs
        ):
            aliases[node.output[0]] = resolve(node.input[0])
        else:
            nodes.append(node)

    for node in nodes:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return len(aliases)


def eliminate_dead_code(model):
    """Remove unreachable pure ONNX nodes, values, and initializers."""
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    required_values = {output.name for output in model.graph.output}
    required_nodes = set()
    pending = list(required_values)
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        for name in node.input:
            if name and name not in required_values:
                required_values.add(name)
                pending.append(name)

    old_node_count = len(model.graph.node)
    nodes = [node for node in model.graph.node if id(node) in required_nodes]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    initializers = [
        tensor
        for tensor in model.graph.initializer
        if tensor.name in required_values
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)

    public_values = {
        value.name for value in (*model.graph.input, *model.graph.output)
    }
    value_info = [
        value
        for value in model.graph.value_info
        if value.name in required_values and value.name not in public_values
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    return old_node_count - len(nodes)


def find_attention_nodes(model):
    """Keep complete MatMul-(Add)-Softmax-MatMul attention paths in FP32."""
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    blocked = set()
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or len(softmax.input) != 1:
            continue
        score_node = producers.get(softmax.input[0])
        first_matmul = (
            score_node if score_node and score_node.op_type == "MatMul" else None
        )
        if score_node and score_node.op_type == "Add":
            candidates = [
                producers.get(name)
                for name in score_node.input
                if producers.get(name) and producers[name].op_type == "MatMul"
            ]
            if len(candidates) == 1:
                first_matmul = candidates[0]
        second_matmuls = [
            node
            for node in consumers.get(softmax.output[0], [])
            if node.op_type == "MatMul"
        ]
        if first_matmul is None or not second_matmuls:
            continue
        blocked.update((first_matmul.name, softmax.name))
        if score_node is not first_matmul:
            blocked.add(score_node.name)
        blocked.update(node.name for node in second_matmuls)
    return sorted(blocked)


def set_metadata(model, **entries):
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update({key: str(value) for key, value in entries.items()})
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)


def save_external(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    model.ir_version = IR_VERSION
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(path))


def save_referenced_external(model, raw_path, output_path):
    """Move an existing external-data file without loading its tensor bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data_path = raw_path.with_suffix(raw_path.suffix + ".data")
    output_data_path = output_path.with_suffix(output_path.suffix + ".data")
    if output_data_path.exists():
        output_data_path.unlink()

    if raw_data_path.exists():
        shutil.move(raw_data_path, output_data_path)
        for tensor in model.graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = output_data_path.name

    model.ir_version = IR_VERSION
    onnx.save_model(model, str(output_path))
    onnx.checker.check_model(str(output_path))


def optimize_fp32(raw_path, output_path, component):
    # Keep multi-gigabyte weights external so exact graph cleanup stays bounded
    # in memory and never duplicates the full parameter payload.
    model = onnx.load(str(raw_path), load_external_data=False)
    sequence_count = replace_sequences(model)
    assign_node_names(model)
    castlike_count = replace_castlike(model)
    identity_count = eliminate_identities(model)
    dead_count = eliminate_dead_code(model)
    assign_node_names(model)
    set_metadata(
        model,
        precision="fp32",
        target_runtime="TensorRT 10.13",
        sequence_splits_lowered=sequence_count,
        castlike_lowered=castlike_count,
        identities_eliminated=identity_count,
        dead_nodes_eliminated=dead_count,
    )
    if component == "image_encoder":
        set_metadata(model, image_input="float32 NCHW, normalized to [-1,1]")
    if component == "decoder":
        set_metadata(
            model,
            boxes_format="normalized cx,cy,width,height",
            boxes_scale="x/width and y/height of the processed image",
        )
    node_count = len(model.graph.node)
    save_referenced_external(model, raw_path, output_path)
    del model
    gc.collect()
    return node_count, sequence_count


def convert_fp16(fp32_path, output_path, component):
    model = onnx.load(str(fp32_path), load_external_data=True)
    assign_node_names(model)
    blocked_nodes = find_attention_nodes(model)
    # PyTorch exports dtype-dependent casts as explicit Cast(to=FLOAT). Make
    # those internal casts follow the converted graph precision; the converter
    # adds separate FP32 casts at public I/O and blocked attention boundaries.
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to = next((attr for attr in node.attribute if attr.name == "to"), None)
        if to is not None and to.i == TensorProto.FLOAT:
            to.i = TensorProto.FLOAT16
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=[],
        node_block_list=blocked_nodes,
        min_positive_val=5.960464477539063e-08,
        max_finite_val=65504.0,
        check_fp16_ready=False,
    )
    # Intermediate annotations are optional; discarding them avoids stale types
    # while runtimes infer the final graph directly from its nodes.
    del model.graph.value_info[:]
    set_metadata(
        model,
        precision="mixed fp16 with attention in fp32",
        fp32_attention_nodes=len(blocked_nodes),
        target_runtime="TensorRT 10.13",
    )
    save_external(model, output_path)
    del model
    gc.collect()
    return len(blocked_nodes)


def export(module, inputs, path, input_names, output_names, opset):
    print(f"Exporting {path.name} ...")
    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=True,
        external_data=True,
        optimize=False,
    )


def export_component(
    module,
    inputs,
    component,
    input_names,
    output_names,
    opset,
    raw_dir,
    fp32_dir,
    fp16_dir,
):
    raw_path = raw_dir / f"{component}.onnx"
    export(module, inputs, raw_path, input_names, output_names, opset)
    gc.collect()
    nodes, sequences = optimize_fp32(
        raw_path, fp32_dir / f"{component}.onnx", component
    )
    blocked = convert_fp16(
        fp32_dir / f"{component}.onnx",
        fp16_dir / f"{component}.onnx",
        component,
    )
    print(
        f"  {component}: optimized graph has {nodes} nodes, "
        f"{sequences} sequence splits lowered, "
        f"{blocked} attention nodes kept in FP32"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export TensorRT 10.13-compatible FP32 and mixed-FP16 SAM3 ONNX "
            "components. The image input is float32 NCHW (N=1), normalized "
            "to [-1, 1]."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="sam3.pt; downloads from HF if omitted"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"))
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(1008, 1008),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    height, width = args.image_size
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if height <= 0 or width <= 0 or height % 14 or width % 14:
        parser.error(
            "--image-size dimensions must be positive multiples of the "
            "14-pixel patch size"
        )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but PyTorch cannot access an NVIDIA driver")

    fp32_dir = args.output_dir / "fp32"
    fp16_dir = args.output_dir / "fp16"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
        device=device,
        enable_inst_interactivity=False,
        compile=False,
    ).eval()
    model.requires_grad_(False)
    vision = model.backbone.vision_backbone
    patch_h, patch_w = prepare_vision(vision, height, width, device)
    model.transformer.decoder.compilable_cord_cache = (
        model.transformer.decoder._get_coords(patch_h, patch_w, device)
    )
    model.transformer.decoder.compilable_stored_size = (patch_h, patch_w)
    vision.position_encoding.cache.clear()
    vision.position_encoding(torch.zeros(1, 1, patch_h, patch_w, device=device))

    bpe_path = Path(__file__).parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ids = torch.zeros(1, 32, dtype=torch.long, device=device)
    text = torch.zeros(32, 1, 256, device=device)
    text_mask = torch.zeros(1, 32, dtype=torch.bool, device=device)
    features = (
        torch.zeros(1, 256, patch_h * 4, patch_w * 4, device=device),
        torch.zeros(1, 256, patch_h * 2, patch_w * 2, device=device),
        torch.zeros(1, 256, patch_h, patch_w, device=device),
    )
    with tempfile.TemporaryDirectory(dir=args.output_dir) as raw:
        raw_dir = Path(raw)
        tokenizer_path = fp32_dir / "tokenizer.onnx"
        export_tokenizer(
            tokenizer_path,
            model.backbone.language_backbone.tokenizer,
            bpe_path,
        )
        shutil.copy2(tokenizer_path, fp16_dir / "tokenizer.onnx")
        with torch.inference_mode():
            export_component(
                TextEncoder(model.backbone.language_backbone).eval(),
                (ids,),
                "text_encoder",
                ["input_ids"],
                ["text_features", "text_padding_mask"],
                args.opset,
                raw_dir,
                fp32_dir,
                fp16_dir,
            )
            export_component(
                ImageEncoder(vision).eval(),
                (torch.zeros(1, 3, height, width, device=device),),
                "image_encoder",
                ["image"],
                ["feature_0", "feature_1", "feature_2"],
                args.opset,
                raw_dir,
                fp32_dir,
                fp16_dir,
            )
            export_component(
                Decoder(model).eval(),
                (*features, text, text_mask),
                "decoder",
                [
                    "feature_0",
                    "feature_1",
                    "feature_2",
                    "text_features",
                    "text_padding_mask",
                ],
                ["mask_logits", "boxes_cxcywh", "scores", "presence_score"],
                args.opset,
                raw_dir,
                fp32_dir,
                fp16_dir,
            )
    print(f"Done. FP32 models: {fp32_dir.resolve()}")
    print(f"Done. FP16 models: {fp16_dir.resolve()}")


if __name__ == "__main__":
    main()
