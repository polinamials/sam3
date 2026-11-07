#!/usr/bin/env python3
"""Export fixed-shape SAM3 image models for ONNX Runtime and TensorRT."""

import argparse
import gc
import gzip
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from torch import nn

IR_VERSION = 12
OPSET_VERSION = 23
TEXT_LENGTH = 32
PATCH_SIZE = 14
SEQUENCE_OPS = {
    "ConcatFromSequence", "SequenceAt", "SequenceConstruct", "SequenceEmpty",
    "SequenceErase", "SequenceInsert", "SequenceLength", "SplitToSequence",
}
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
ONNX_DTYPES = {
    "fp32": TensorProto.FLOAT,
    "fp16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
}
FEATURE_ALIASES = {
    "text": "text",
    "box": "box",
    "boxes": "box",
    "bounding-box": "box",
    "bounding_box": "box",
    "bounding box": "box",
    "point": "point",
    "points": "point",
}


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class ExportMlp(nn.Module):
    """Equivalent exportable form of the vision trunk's eval-time MLP."""

    def __init__(self, mlp):
        super().__init__()
        self.fc1, self.act, self.drop1 = mlp.fc1, mlp.act, mlp.drop1
        self.norm, self.fc2, self.drop2 = mlp.norm, mlp.fc2, mlp.drop2

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(self.norm(x)))


class PrecisionModule(nn.Module):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision
        self.output_dtype = PRECISION_DTYPES[precision]

    def precision_context(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.output_dtype)

    def output(self, tensor):
        return tensor.to(self.output_dtype) if tensor.is_floating_point() else tensor


class TextEncoder(PrecisionModule):
    def __init__(self, text, precision):
        super().__init__(precision)
        self.encoder, self.resizer = text.encoder, text.resizer

    def forward(self, input_ids):
        with self.precision_context():
            _, tokens = self.encoder(input_ids)
            features = self.resizer(tokens).transpose(0, 1)
        return self.output(features), input_ids.eq(0)


class ImageEncoder(PrecisionModule):
    def __init__(self, vision, tracker, concept_enabled, point_enabled, precision):
        super().__init__(precision)
        self.trunk = vision.trunk
        self.convs = vision.convs if concept_enabled else None
        self.point_convs = vision.sam2_convs if point_enabled else None
        if point_enabled:
            self.point_conv_s0 = tracker.sam_mask_decoder.conv_s0
            self.point_conv_s1 = tracker.sam_mask_decoder.conv_s1
            self.register_buffer(
                "no_mem_embed", tracker.no_mem_embed.reshape(1, -1, 1, 1)
            )

    def forward(self, image):
        with self.precision_context():
            trunk = self.trunk(image)[-1]
            outputs = []
            if self.convs is not None:
                outputs.extend(self.convs[i](trunk) for i in range(3))
            if self.point_convs is not None:
                high_res_0 = self.point_conv_s0(self.point_convs[0](trunk))
                high_res_1 = self.point_conv_s1(self.point_convs[1](trunk))
                image_embedding = self.point_convs[2](trunk)
                image_embedding = image_embedding + self.no_mem_embed.to(
                    image_embedding.dtype
                )
                outputs.extend((high_res_0, high_res_1, image_embedding))
        return tuple(self.output(value) for value in outputs)


class ConceptDecoder(PrecisionModule):
    def __init__(
        self, model, batch, use_text, use_boxes, precision,
        fixed_text=None, fixed_text_mask=None,
    ):
        super().__init__(precision)
        self.pos = model.backbone.vision_backbone.position_encoding
        self.geometry = model.geometry_encoder
        self.encoder = model.transformer.encoder
        self.decoder = model.transformer.decoder
        self.scorer = model.dot_prod_scoring
        self.segmentation = model.segmentation_head
        self.batch = batch
        self.use_text = use_text
        self.use_boxes = use_boxes
        self.register_buffer("image_ids", torch.arange(batch, dtype=torch.long))
        if not use_text:
            self.register_buffer("fixed_text", fixed_text)
            self.register_buffer("fixed_text_mask", fixed_text_mask)

    def _geometry_prompt(self, image, image_pos, feature_size, inputs):
        if self.use_boxes:
            from sam3.model.geometry_encoders import Prompt

            boxes, box_labels, box_padding_mask = inputs
            geometry_prompt = Prompt(
                box_embeddings=boxes.transpose(0, 1),
                box_labels=box_labels.transpose(0, 1),
                box_mask=box_padding_mask,
            )
            return self.geometry(
                geometry_prompt,
                img_feats=[image],
                img_sizes=[feature_size],
                img_pos_embeds=[image_pos],
            )

        geometry = self.geometry.cls_embed.weight.view(1, 1, -1)
        geometry = geometry.expand(-1, self.batch, -1)
        geometry = self.geometry.norm(self.geometry.final_proj(geometry))
        geometry_mask = torch.zeros(
            (self.batch, 1), dtype=torch.bool, device=geometry.device
        )
        for layer in self.geometry.encode:
            geometry = layer(
                tgt=geometry,
                memory=image,
                tgt_key_padding_mask=geometry_mask,
                pos=image_pos,
            )
        return self.geometry.encode_norm(geometry), geometry_mask

    def forward(self, feature_0, feature_1, feature_2, *inputs):
        if self.use_text:
            text, text_padding_mask = inputs[:2]
            box_inputs = inputs[2:]
        else:
            text, text_padding_mask = self.fixed_text, self.fixed_text_mask
            box_inputs = inputs

        with self.precision_context():
            image = feature_2.flatten(2).permute(2, 0, 1)
            image_pos = (
                self.pos(feature_2).to(feature_2.dtype).flatten(2).permute(2, 0, 1)
            )
            geometry, geometry_mask = self._geometry_prompt(
                image, image_pos, feature_2.shape[-2:], box_inputs
            )
            prompt = torch.cat((text, geometry))
            prompt_mask = torch.cat((text_padding_mask, geometry_mask), dim=1)
            encoded = self.encoder(
                src=[image],
                src_pos=[image_pos],
                prompt=prompt,
                prompt_key_padding_mask=prompt_mask,
                prompt_pos=torch.zeros_like(prompt),
                feat_sizes=[feature_2.shape[-2:]],
            )
            memory = encoded["memory"]
            target = self.decoder.query_embed.weight[:, None].expand(
                -1, self.batch, -1
            )
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
            scores = self.scorer(states, prompt, prompt_mask)[-1]
            boxes = (
                inverse_sigmoid(references) + self.decoder.bbox_embed(states)
            ).sigmoid()[-1]
            masks = self.segmentation(
                backbone_feats=[feature_0, feature_1, feature_2],
                obj_queries=states,
                image_ids=self.image_ids,
                encoder_hidden_states=memory,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )["pred_masks"]
            presence = presence.transpose(1, 2)[-1]
        return tuple(self.output(value) for value in (masks, boxes, scores, presence))


class PointDecoder(PrecisionModule):
    def __init__(self, tracker, precision):
        super().__init__(precision)
        self.prompt_encoder = tracker.sam_prompt_encoder
        self.mask_decoder = tracker.sam_mask_decoder

    def forward(
        self, point_high_res_0, point_high_res_1, point_image_embedding,
        point_coords, point_labels,
    ):
        with self.precision_context():
            sparse, dense = self.prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None
            )
            masks, iou_scores, _, _ = self.mask_decoder.predict_masks(
                image_embeddings=point_image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                repeat_image=False,
                high_res_features=[point_high_res_0, point_high_res_1],
            )
        return self.output(masks), self.output(iou_scores)


def export_tokenizer(path, tokenizer, bpe_path, batch):
    vocab = dict(tokenizer.encoder)
    vocab["<|startoftext|>"] = tokenizer.sot_token_id
    vocab["<|endoftext|>"] = tokenizer.eot_token_id
    lines = gzip.open(bpe_path, "rt", encoding="utf-8").read().splitlines()
    merges = ("\n".join(lines[: 49152 - 256 - 2 + 1]) + "\n").encode()
    nodes = [
        helper.make_node(
            "StringECMARegexReplace", ["text", "whitespace", "space"],
            ["clean_text"], domain="ai.onnx.contrib", global_replace=1,
        ),
        helper.make_node(
            "CLIPTokenizer", ["clean_text"], ["raw_ids", "raw_mask"],
            domain="ai.onnx.contrib", vocab=json.dumps(vocab, ensure_ascii=False).encode(),
            merges=merges, padding_length=-1,
        ),
        helper.make_node("Pad", ["raw_ids", "pads", "zero"], ["padded_ids"]),
        helper.make_node("Pad", ["raw_mask", "pads", "zero"], ["padded_mask"]),
        helper.make_node("Slice", ["padded_ids", "starts", "ends", "axes"], ["ids"]),
        helper.make_node("Slice", ["padded_mask", "starts", "ends", "axes"], ["mask"]),
        helper.make_node("Shape", ["raw_ids"], ["raw_shape"]),
        helper.make_node("Gather", ["raw_shape", "one"], ["raw_length"], axis=0),
        helper.make_node("Greater", ["raw_length", "length"], ["truncated"]),
        helper.make_node("Slice", ["ids", "starts", "prefix_ends", "axes"], ["prefix"]),
        helper.make_node("Gather", ["ids", "last"], ["old_last"], axis=1),
        helper.make_node("Where", ["truncated", "eot", "old_last"], ["new_last"]),
        helper.make_node("Concat", ["prefix", "new_last"], ["truncated_ids"], axis=1),
        helper.make_node("Cast", ["mask"], ["valid"], to=TensorProto.BOOL),
        helper.make_node("Where", ["valid", "truncated_ids", "zero"], ["input_ids"]),
    ]
    values = {
        "pads": ([4], [0, 0, 0, TEXT_LENGTH]),
        "zero": ([], [0]),
        "starts": ([2], [0, 0]),
        "ends": ([2], [batch, TEXT_LENGTH]),
        "axes": ([2], [0, 1]),
        "one": ([1], [1]),
        "length": ([], [TEXT_LENGTH]),
        "prefix_ends": ([2], [batch, TEXT_LENGTH - 1]),
        "last": ([1], [TEXT_LENGTH - 1]),
        "eot": ([], [tokenizer.eot_token_id]),
    }
    initializers = [
        helper.make_tensor(name, TensorProto.INT64, shape, data)
        for name, (shape, data) in values.items()
    ]
    initializers += [
        helper.make_tensor("whitespace", TensorProto.STRING, [1], [rb"\s+"]),
        helper.make_tensor("space", TensorProto.STRING, [1], [b" "]),
    ]
    graph = helper.make_graph(
        nodes, "sam3_tokenizer",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [batch])],
        [helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, [batch, TEXT_LENGTH]
        )],
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
    model.ir_version = IR_VERSION
    model.metadata_props.add(
        key="runtime_requirement", value="onnxruntime-extensions"
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def prepare_vision(vision, height, width, device):
    """Freeze fixed-resolution RoPE constants in an equivalent real form."""
    patch_h, patch_w = height // PATCH_SIZE, width // PATCH_SIZE
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
                (torch.polar(torch.ones_like(x), x),
                 torch.polar(torch.ones_like(y), y)),
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


def prepare_fixed_shapes(
    model, height, width, patch_h, patch_w, concept_enabled, point_enabled
):
    if concept_enabled:
        decoder = model.transformer.decoder
        decoder.compilable_cord_cache = decoder._get_coords(
            patch_h, patch_w, model.device
        )
        decoder.compilable_stored_size = (patch_h, patch_w)

        position = model.backbone.vision_backbone.position_encoding
        position.cache.clear()
        position(torch.zeros(1, 1, patch_h, patch_w, device=model.device))

    if point_enabled:
        prompt_encoder = model.inst_interactive_predictor.model.sam_prompt_encoder
        prompt_encoder.image_embedding_size = (patch_h, patch_w)
        prompt_encoder.input_image_size = (height, width)
        prompt_encoder.mask_input_size = (4 * patch_h, 4 * patch_w)


def encode_fixed_text(text_model, texts, precision):
    input_ids = text_model.tokenizer(texts, context_length=TEXT_LENGTH).to(
        next(text_model.parameters()).device
    )
    context = (
        nullcontext() if precision == "fp32"
        else torch.autocast("cuda", dtype=PRECISION_DTYPES[precision])
    )
    with torch.inference_mode(), context:
        _, tokens = text_model.encoder(input_ids)
        features = text_model.resizer(tokens).transpose(0, 1)
    return features.to(PRECISION_DTYPES[precision]), input_ids.eq(0)


def export(module, inputs, path, input_names, output_names):
    print(f"Exporting {path.name} ...", flush=True)
    torch.onnx.export(
        module.eval(), inputs, path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=True,
        optimize=True,
    )


def install_export(raw_path, output_path):
    """Move an exported model and its external weights out of the temp directory."""
    model = onnx.load(raw_path, load_external_data=False)
    external_files = {
        item.value
        for tensor in model.graph.initializer
        if tensor.data_location == TensorProto.EXTERNAL
        for item in tensor.external_data
        if item.key == "location"
    }

    output_path.unlink(missing_ok=True)
    raw_path.replace(output_path)
    for location in external_files:
        source = raw_path.parent / location
        target = output_path.parent / location
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        source.replace(target)


def tensor_types(model):
    types = {}
    values = (*model.graph.input, *model.graph.output, *model.graph.value_info)
    for value in values:
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    for initializer in model.graph.initializer:
        types[initializer.name] = initializer.data_type
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor = next(
            (attr.t for attr in node.attribute
             if attr.type == onnx.AttributeProto.TENSOR),
            None,
        )
        if tensor is not None:
            types[node.output[0]] = tensor.data_type
    return types


def validate_graph(model, path, component, precision, batch, height, width):
    nodes = list(model.graph.node)
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    duplicate_names = sorted(input_names & output_names)
    if duplicate_names:
        raise RuntimeError(
            f"{path.name} reuses graph input names as outputs: {duplicate_names}"
        )
    unsupported = sorted(
        {node.op_type for node in nodes if node.domain in ("", "ai.onnx")}
        & SEQUENCE_OPS
    )
    custom_domains = sorted(
        {node.domain for node in nodes if node.domain not in ("", "ai.onnx")}
    )
    if unsupported or (component != "tokenizer" and custom_domains):
        raise RuntimeError(
            f"{path.name} has unsupported ops/domains: {unsupported + custom_domains}"
        )
    for value in (*model.graph.input, *model.graph.output):
        dimensions = value.type.tensor_type.shape.dim
        if any(not dimension.HasField("dim_value") for dimension in dimensions):
            raise RuntimeError(f"{path.name} has a dynamic dimension on {value.name}")
    if component == "tokenizer":
        return

    types = tensor_types(model)
    expected_type = ONNX_DTYPES[precision]
    floating_types = {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.DOUBLE,
    }
    for value in model.graph.output:
        output_type = value.type.tensor_type.elem_type
        if output_type in floating_types and output_type != expected_type:
            raise RuntimeError(
                f"{path.name} output {value.name} has unexpected type {output_type}"
            )
    attentions = [node for node in nodes if node.op_type == "Attention"]
    if not attentions:
        raise RuntimeError(f"{path.name} has no native ONNX Attention nodes")
    for node in attentions:
        qkv_types = [types.get(name) for name in node.input[:3]]
        if qkv_types != [expected_type] * 3:
            raise RuntimeError(
                f"{path.name} has unexpected Attention input types: {qkv_types}"
            )

    producer = {output: node for node in nodes for output in node.output if output}
    consumers = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    for node in nodes:
        if node.op_type != "Softmax" or not node.input or not node.output:
            continue
        before = producer.get(node.input[0])
        after = consumers.get(node.output[0], ())
        if before and before.op_type == "MatMul" and any(
            consumer.op_type == "MatMul" for consumer in after
        ):
            raise RuntimeError(
                f"{path.name} still contains primitive MatMul/Softmax attention"
            )

    for node in nodes:
        if node.op_type == "Range" and any(
            types.get(name) in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
            for name in node.input
        ):
            raise RuntimeError(f"{path.name} contains a low-precision Range")

    if component == "image_encoder":
        image = next(value for value in model.graph.input if value.name == "image")
        tensor_type = image.type.tensor_type
        dimensions = [dimension.dim_value for dimension in tensor_type.shape.dim]
        expected_shape = [batch, 3, height, width]
        if tensor_type.elem_type != TensorProto.FLOAT or dimensions != expected_shape:
            raise RuntimeError(f"image input must be float32 {expected_shape}")


def finalize(path, component, precision, batch, height, width, features):
    model = onnx.load(path, load_external_data=False)
    model.ir_version = IR_VERSION
    metadata = {
        "features": ",".join(features),
        "precision": precision if component != "tokenizer" else "not applicable",
        "target_runtime": "ONNX Runtime 1.27.1 + TensorRT 11.1.0",
    }
    if component != "tokenizer":
        metadata["precision_mode"] = "strongly typed mixed precision"
    if component == "image_encoder":
        metadata["image_input"] = (
            f"float32 NCHW [{batch},3,{height},{width}], normalized to [-1,1]"
        )
    if component == "concept_decoder":
        metadata["concept_boxes"] = (
            "normalized cx,cy,width,height; labels: 1 positive, 0 negative"
        )
    if component == "point_decoder":
        metadata["points"] = (
            "absolute resized-image (x,y); labels: 1 foreground, "
            "0 background, -1 padding"
        )
        metadata["point_mask_channels"] = (
            "0 is the single-mask token; 1..3 are multimask candidates"
        )
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(metadata)
    del model.metadata_props[:]
    for key, value in existing.items():
        model.metadata_props.add(key=key, value=value)

    validate_graph(model, path, component, precision, batch, height, width)
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    print(f"  {path.name}: {len(model.graph.node)} nodes", flush=True)


def parse_features(parser, values):
    features = []
    for value in values:
        feature = FEATURE_ALIASES.get(value.lower())
        if feature is None:
            parser.error(
                f"unknown feature {value!r}; choose text, bounding-box, or point"
            )
        if feature not in features:
            features.append(feature)
    return features


def parse_image_size(parser, values):
    if len(values) == 1:
        height = width = values[0]
    elif len(values) == 2:
        height, width = values
    else:
        parser.error("--image-size accepts SIZE or HEIGHT WIDTH")
    if min(height, width) <= 0 or height % PATCH_SIZE or width % PATCH_SIZE:
        parser.error("image dimensions must be positive multiples of 14")
    return height, width


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed-shape SAM3 image models for ONNX Runtime 1.27.1 "
            "and TensorRT 11.1.0."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="sam3.pt; download from Hugging Face if omitted",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--image-size", type=int, nargs="+", default=[1008], metavar="SIZE",
        help="fixed SIZE or HEIGHT WIDTH; each dimension must be divisible by 14",
    )
    parser.add_argument(
        "--precision", choices=tuple(PRECISION_DTYPES), default="fp32"
    )
    parser.add_argument(
        "--features", nargs="+", default=["text"], metavar="FEATURE",
        help="any combination of: text, bounding-box, point",
    )
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1)
    args = parser.parse_args()

    features = parse_features(parser, args.features)
    height, width = parse_image_size(parser, args.image_size)
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if "box" in features and args.max_boxes <= 0:
        parser.error("--max-boxes must be positive when bounding-box is enabled")
    if "point" in features and args.max_points <= 0:
        parser.error("--max-points must be positive when point is enabled")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not torch.cuda.is_available():
        parser.error("CUDA is required to export SAM3")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("the available CUDA device does not support BF16")

    from sam3.model_builder import build_sam3_image_model

    device = "cuda"
    output_dir = (args.output_dir / args.precision).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_enabled = "point" in features
    concept_enabled = "text" in features or "box" in features
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
        device=device,
        enable_segmentation=concept_enabled,
        enable_inst_interactivity=point_enabled,
        compile=False,
    ).eval()
    model.requires_grad_(False)

    vision = model.backbone.vision_backbone
    patch_h, patch_w = prepare_vision(vision, height, width, device)
    prepare_fixed_shapes(
        model, height, width, patch_h, patch_w, concept_enabled, point_enabled
    )
    dtype = PRECISION_DTYPES[args.precision]
    if "text" in features:
        ids = torch.zeros(args.batch, TEXT_LENGTH, dtype=torch.long, device=device)
        text = torch.zeros(
            TEXT_LENGTH, args.batch, 256, dtype=dtype, device=device
        )
        text_mask = torch.zeros(
            args.batch, TEXT_LENGTH, dtype=torch.bool, device=device
        )
    if concept_enabled:
        concept_features = tuple(
            torch.zeros(
                args.batch, 256, patch_h * scale, patch_w * scale,
                dtype=dtype, device=device,
            )
            for scale in (4, 2, 1)
        )

    fixed_text = fixed_text_mask = None
    if concept_enabled and "text" not in features:
        fixed_text, fixed_text_mask = encode_fixed_text(
            model.backbone.language_backbone,
            ["visual"] * args.batch,
            args.precision,
        )

    tracker = model.inst_interactive_predictor.model if point_enabled else None
    jobs = []
    if "text" in features:
        bpe = Path(__file__).parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        export_tokenizer(
            output_dir / "tokenizer.onnx",
            model.backbone.language_backbone.tokenizer,
            bpe,
            args.batch,
        )
        jobs.append((
            TextEncoder(model.backbone.language_backbone, args.precision),
            (ids,), "text_encoder", ["input_ids"],
            ["text_features", "text_padding_mask"],
        ))

    image_output_names = []
    if concept_enabled:
        image_output_names.extend(["feature_0", "feature_1", "feature_2"])
    if point_enabled:
        image_output_names.extend(
            ["point_high_res_0", "point_high_res_1", "point_image_embedding"]
        )
    jobs.append((
        ImageEncoder(vision, tracker, concept_enabled, point_enabled, args.precision),
        (torch.zeros(
            args.batch, 3, height, width, dtype=torch.float32, device=device
        ),),
        "image_encoder", ["image"], image_output_names,
    ))

    if concept_enabled:
        concept_inputs = list(concept_features)
        concept_input_names = ["feature_0", "feature_1", "feature_2"]
        if "text" in features:
            concept_inputs.extend((text, text_mask))
            concept_input_names.extend(("text_features", "text_padding_mask"))
        if "box" in features:
            concept_inputs.extend((
                torch.zeros(
                    args.batch, args.max_boxes, 4,
                    dtype=torch.float32, device=device,
                ),
                torch.ones(
                    args.batch, args.max_boxes, dtype=torch.long, device=device,
                ),
                torch.zeros(
                    args.batch, args.max_boxes, dtype=torch.bool, device=device,
                ),
            ))
            concept_input_names.extend(
                ("prompt_boxes_cxcywh", "box_labels", "box_padding_mask")
            )
        jobs.append((
            ConceptDecoder(
                model, args.batch, "text" in features, "box" in features,
                args.precision, fixed_text, fixed_text_mask,
            ),
            tuple(concept_inputs),
            "concept_decoder",
            concept_input_names,
            ["mask_logits", "boxes_cxcywh", "scores", "presence_score"],
        ))

    if point_enabled:
        point_features = (
            torch.zeros(
                args.batch, 32, patch_h * 4, patch_w * 4,
                dtype=dtype, device=device,
            ),
            torch.zeros(
                args.batch, 64, patch_h * 2, patch_w * 2,
                dtype=dtype, device=device,
            ),
            torch.zeros(
                args.batch, 256, patch_h, patch_w,
                dtype=dtype, device=device,
            ),
        )
        jobs.append((
            PointDecoder(tracker, args.precision),
            (
                *point_features,
                torch.zeros(
                    args.batch, args.max_points, 2,
                    dtype=torch.float32, device=device,
                ),
                torch.ones(
                    args.batch, args.max_points, dtype=torch.int32, device=device,
                ),
            ),
            "point_decoder",
            [
                "point_high_res_0", "point_high_res_1", "point_image_embedding",
                "point_coords", "point_labels",
            ],
            ["point_mask_logits", "point_iou_scores"],
        ))

    with tempfile.TemporaryDirectory(dir=output_dir) as temp:
        raw = Path(temp)
        with torch.inference_mode():
            for module, inputs, name, input_names, output_names in jobs:
                export(
                    module, inputs, raw / f"{name}.onnx",
                    input_names, output_names,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model_names = [job[2] for job in jobs]
        del jobs, model, vision
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for name in model_names:
            install_export(raw / f"{name}.onnx", output_dir / f"{name}.onnx")
            finalize(
                output_dir / f"{name}.onnx", name, args.precision,
                args.batch, height, width, features,
            )

    if "text" in features:
        finalize(
            output_dir / "tokenizer.onnx", "tokenizer", args.precision,
            args.batch, height, width, features,
        )
    print(f"Done: {output_dir}", flush=True)
    print("TensorRT EP: use the strongly typed ONNX tensor types.", flush=True)


if __name__ == "__main__":
    main()
