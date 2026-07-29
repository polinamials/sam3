#!/usr/bin/env python3
"""Export fixed-shape, mixed-FP16 SAM3 ONNX models for TensorRT 10.16."""

import argparse
import gc
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from torch import nn

IR_VERSION = 12
OPSET_VERSION = 23
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


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class ExportMlp(nn.Module):
    """Export SAM3's eval-time Linear/GELU/Linear MLP without its private op."""

    def __init__(self, mlp):
        super().__init__()
        self.fc1, self.act, self.drop1 = mlp.fc1, mlp.act, mlp.drop1
        self.norm, self.fc2, self.drop2 = mlp.norm, mlp.fc2, mlp.drop2

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(self.norm(x)))


class TextEncoder(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.encoder, self.resizer = text.encoder, text.resizer

    def forward(self, input_ids):
        with torch.autocast("cuda", dtype=torch.float16):
            _, tokens = self.encoder(input_ids)
            return self.resizer(tokens).transpose(0, 1), input_ids.eq(0)


class ImageEncoder(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.trunk, self.convs = vision.trunk, vision.convs

    def forward(self, image):
        with torch.autocast("cuda", dtype=torch.float16):
            x = self.trunk(image)[-1]
            return self.convs[0](x), self.convs[1](x), self.convs[2](x)


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
        with torch.autocast("cuda", dtype=torch.float16):
            image = feature_2.flatten(2).permute(2, 0, 1)
            image_pos = (
                self.pos(feature_2)
                .to(feature_2.dtype)
                .flatten(2)
                .permute(2, 0, 1)
            )

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
                inverse_sigmoid(references) + self.decoder.bbox_embed(states)
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
        helper.make_tensor(name, TensorProto.INT64, shape, data)
        for name, (shape, data) in values.items()
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
    model.ir_version = IR_VERSION
    model.metadata_props.add(
        key="runtime_requirement", value="onnxruntime-extensions"
    )
    model.metadata_props.add(key="target_runtime", value="ONNX Runtime 1.27.1")
    onnx.checker.check_model(model)
    onnx.save(model, path)


def prepare_vision(vision, height, width, device):
    """Freeze fixed-shape RoPE constants using its equivalent real formulation."""
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


def export(module, inputs, path, input_names, output_names):
    print(f"Exporting {path.name} ...", flush=True)
    torch.onnx.export(
        module.eval(),
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=True,
        optimize=True,
    )


def simplify(raw_path, output_path):
    output_path.unlink(missing_ok=True)
    output_path.with_suffix(output_path.suffix + ".data").unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxsim",
            str(raw_path.resolve()),
            output_path.name,
            "0",
            "--save-as-external-data",
            "--no-large-tensor",
            "1MB",
        ],
        cwd=output_path.parent,
        check=True,
    )


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
            (attr.t for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR),
            None,
        )
        if tensor is not None:
            types[node.output[0]] = tensor.data_type
    return types


def validate_graph(model, path, component):
    nodes = list(model.graph.node)
    unsupported = sorted(
        {node.op_type for node in nodes if node.domain in ("", "ai.onnx")}
        & SEQUENCE_OPS
    )
    custom_domains = sorted(
        {node.domain for node in nodes if node.domain not in ("", "ai.onnx")}
    )
    if unsupported or (component != "tokenizer" and custom_domains):
        raise RuntimeError(
            f"{path.name} has unsupported ops/domains: "
            f"{unsupported + custom_domains}"
        )
    if component == "tokenizer":
        return

    types = tensor_types(model)
    attentions = [node for node in nodes if node.op_type == "Attention"]
    if not attentions:
        raise RuntimeError(
            f"{path.name} has no native ONNX Attention nodes; use opset 23"
        )
    for node in attentions:
        qkv_types = [types.get(name) for name in node.input[:3]]
        if qkv_types != [TensorProto.FLOAT16] * 3:
            raise RuntimeError(
                f"{path.name} has non-FP16 Attention inputs: {qkv_types}"
            )

    producer = {
        output: node for node in nodes for output in node.output if output
    }
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
            types.get(input_name) == TensorProto.FLOAT16 for input_name in node.input
        ):
            raise RuntimeError(f"{path.name} contains an illegal FP16 Range")

    if component == "image_encoder":
        image = next(value for value in model.graph.input if value.name == "image")
        tensor_type = image.type.tensor_type
        dimensions = tensor_type.shape.dim
        if (
            tensor_type.elem_type != TensorProto.FLOAT
            or len(dimensions) != 4
            or dimensions[0].dim_value != 1
            or dimensions[1].dim_value != 3
        ):
            raise RuntimeError(
                "image_encoder.onnx input must be float32 [1,3,height,width]"
            )


def finalize(path, component):
    model = onnx.load(path, load_external_data=False)
    model.ir_version = IR_VERSION
    metadata = {
        "precision": (
            "not applicable"
            if component == "tokenizer"
            else "mixed FP16 exported by PyTorch CUDA autocast"
        ),
        "target_runtime": "ONNX Runtime 1.27.1 + TensorRT 10.16.1",
    }
    if component != "tokenizer":
        metadata["tensorrt_mode"] = "strongly typed; set trt_fp16_enable=false"
    if component == "image_encoder":
        metadata["image_input"] = "float32 NCHW [1,3,H,W], normalized to [-1,1]"
    if component == "decoder":
        metadata["boxes"] = "normalized cx,cy,width,height"
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(metadata)
    del model.metadata_props[:]
    for key, value in existing.items():
        model.metadata_props.add(key=key, value=value)

    validate_graph(model, path, component)
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    print(f"  {path.name}: {len(model.graph.node)} nodes", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed-shape SAM3 mixed-FP16 ONNX models for ONNX Runtime "
            "1.27.1 and TensorRT 10.16.1."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="sam3.pt; download from Hugging Face if omitted"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"))
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(1008, 1008),
        metavar=("HEIGHT", "WIDTH"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required because FP16 placement follows CUDA autocast")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    height, width = args.image_size
    if min(height, width) <= 0 or height % 14 or width % 14:
        parser.error("image dimensions must be positive multiples of 14")

    from sam3.model_builder import build_sam3_image_model

    output_dir = (args.output_dir / "fp16").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
        device="cuda",
        enable_inst_interactivity=False,
        compile=False,
    ).eval()
    model.requires_grad_(False)

    vision = model.backbone.vision_backbone
    patch_h, patch_w = prepare_vision(vision, height, width, "cuda")
    model.transformer.decoder.compilable_cord_cache = (
        model.transformer.decoder._get_coords(patch_h, patch_w, "cuda")
    )
    model.transformer.decoder.compilable_stored_size = (patch_h, patch_w)
    vision.position_encoding.cache.clear()
    vision.position_encoding(
        torch.zeros(1, 1, patch_h, patch_w, device="cuda")
    )

    ids = torch.zeros(1, 32, dtype=torch.long, device="cuda")
    text = torch.zeros(32, 1, 256, dtype=torch.float16, device="cuda")
    text_mask = torch.zeros(1, 32, dtype=torch.bool, device="cuda")
    features = tuple(
        torch.zeros(
            1,
            256,
            patch_h * scale,
            patch_w * scale,
            dtype=torch.float16,
            device="cuda",
        )
        for scale in (4, 2, 1)
    )

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
        raw = Path(temp)
        bpe = Path(__file__).parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        export_tokenizer(
            output_dir / "tokenizer.onnx",
            model.backbone.language_backbone.tokenizer,
            bpe,
        )
        jobs = [
            (
                TextEncoder(model.backbone.language_backbone),
                (ids,),
                "text_encoder",
                ["input_ids"],
                ["text_features", "text_padding_mask"],
            ),
            (
                ImageEncoder(vision),
                (
                    torch.zeros(
                        1,
                        3,
                        height,
                        width,
                        dtype=torch.float32,
                        device="cuda",
                    ),
                ),
                "image_encoder",
                ["image"],
                ["feature_0", "feature_1", "feature_2"],
            ),
            (
                Decoder(model),
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
            ),
        ]
        with torch.inference_mode():
            for module, inputs, name, input_names, output_names in jobs:
                export(
                    module,
                    inputs,
                    raw / f"{name}.onnx",
                    input_names,
                    output_names,
                )
                gc.collect()
                torch.cuda.empty_cache()

        del jobs, features, text, text_mask, ids, model, vision
        gc.collect()
        torch.cuda.empty_cache()

        for name in ("text_encoder", "image_encoder", "decoder"):
            simplify(raw / f"{name}.onnx", output_dir / f"{name}.onnx")
            finalize(output_dir / f"{name}.onnx", name)

    finalize(output_dir / "tokenizer.onnx", "tokenizer")
    print(f"Done: {output_dir}", flush=True)
    print("ORT TensorRT EP: use trt_fp16_enable=False (strong typing).", flush=True)


if __name__ == "__main__":
    main()
