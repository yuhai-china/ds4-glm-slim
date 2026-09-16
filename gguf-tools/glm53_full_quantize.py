#!/usr/bin/env python3
"""Build DwarfStar's asymmetric Q2 GGUF for the full GLM-5.3 model."""

from __future__ import annotations

import argparse
import json
import os
import sys

import functools

import glm53_quantize as q
from glm53_manifest import glm53_full_spec, validate_glm53_full_index


# Checkpoint layout (block count, MTP block, routed expert count).  Filled in
# from config.json by main() so that expert-pruned derivatives of GLM-5.3 such
# as GLM-5.3-SLIM-E192 (192 routed experts, no MTP block) can be converted too.
EXPERT_COUNT = 256
BLOCK_COUNT = 79
MTP_BLOCK = 78
SPEC = glm53_full_spec()


def configure_layout(config):
    global EXPERT_COUNT, BLOCK_COUNT, MTP_BLOCK, SPEC
    SPEC = glm53_full_spec(config)
    EXPERT_COUNT = SPEC["expert_count"]
    BLOCK_COUNT = SPEC["block_count"]
    MTP_BLOCK = SPEC["mtp_block"]
    if SPEC["leading_dense"] != 3:
        q.fail(f"unsupported first_k_dense_replace: {SPEC['leading_dense']}")
    return SPEC


def source_prefix(layer):
    return f"model.layers.{layer}"


def add_experts(plan, db, layer, part, qtype):
    source = f"{source_prefix(layer)}.mlp.experts.{{expert}}.{part}_proj.weight"
    shape = db.info(source.format(expert=0))["shape"]
    if len(shape) != 2:
        q.fail(f"expert source is not a matrix: {source.format(expert=0)}")
    for expert in range(EXPERT_COUNT):
        name = source.format(expert=expert)
        if db.info(name)["shape"] != shape:
            q.fail(f"expert shape mismatch: {name}")
    item = q.TensorPlan(
        f"blk.{layer}.ffn_{part}_exps.weight",
        (shape[1], shape[0], EXPERT_COUNT),
        qtype,
        f"routed_{part}",
        source=source,
        expert_layer=layer,
        expert_part=part,
        expert_count=EXPERT_COUNT,
    )
    item.nbytes = q.qtype_nbytes(qtype, item.shape)
    plan.append(item)


def indexer_owner(layer, indexer_types):
    if layer == MTP_BLOCK:
        return layer
    owner = None
    for current in range(layer + 1):
        if indexer_types[current] == "full":
            owner = current
    if owner is None:
        q.fail(f"layer {layer} has no indexer owner")
    return owner


def add_attention(plan, db, layer, indexer_types):
    prefix = f"{source_prefix(layer)}.self_attn"
    owner_prefix = f"{source_prefix(indexer_owner(layer, indexer_types))}.self_attn"
    mapping = (
        ("attn_q_a", prefix, "q_a_proj.weight", q.QTYPE_Q8_0),
        ("attn_q_a_norm", prefix, "q_a_layernorm.weight", q.QTYPE_F32),
        ("attn_q_b", prefix, "q_b_proj.weight", q.QTYPE_Q8_0),
        ("attn_kv_a_mqa", prefix, "kv_a_proj_with_mqa.weight", q.QTYPE_Q8_0),
        ("attn_kv_a_norm", prefix, "kv_a_layernorm.weight", q.QTYPE_F32),
        ("attn_output", prefix, "o_proj.weight", q.QTYPE_Q8_0),
        ("indexer.attn_q_b", owner_prefix, "indexer.wq_b.weight", q.QTYPE_Q8_0),
        ("indexer.attn_k", owner_prefix, "indexer.wk.weight", q.QTYPE_Q8_0),
        ("indexer.k_norm", owner_prefix, "indexer.k_norm.weight", q.QTYPE_F32),
        ("indexer.k_norm", owner_prefix, "indexer.k_norm.bias", q.QTYPE_F32),
        ("indexer.proj", owner_prefix, "indexer.weights_proj.weight", q.QTYPE_F32),
    )
    for target, source_base, source_tail, qtype in mapping:
        suffix = ".bias" if source_tail.endswith(".bias") else ".weight"
        target_name = target + (".bias" if suffix == ".bias" else ".weight")
        q.add_regular(
            plan,
            db,
            f"blk.{layer}.{target_name}",
            f"{source_base}.{source_tail}",
            qtype,
            "dsa",
        )

    combined = f"{prefix}.kv_b_proj.weight"
    expected = [64 * (192 + 256), 512]
    if db.info(combined)["shape"] != expected:
        q.fail(f"unexpected kv_b shape for layer {layer}: {db.info(combined)['shape']}")
    q.add_regular(
        plan,
        db,
        f"blk.{layer}.attn_k_b.weight",
        combined,
        q.QTYPE_Q8_0,
        "dsa",
        shape=(192, 512, 64),
        transform="kv_b_k",
    )
    q.add_regular(
        plan,
        db,
        f"blk.{layer}.attn_v_b.weight",
        combined,
        q.QTYPE_Q8_0,
        "dsa",
        shape=(512, 256, 64),
        transform="kv_b_v",
    )


def add_ffn(plan, db, layer, provisional_q2k=False):
    prefix = f"{source_prefix(layer)}.mlp"
    if layer < 3:
        for part in ("gate", "up", "down"):
            q.add_regular(
                plan,
                db,
                f"blk.{layer}.ffn_{part}.weight",
                f"{prefix}.{part}_proj.weight",
                q.QTYPE_Q8_0,
                "dense_ffn",
            )
        return

    q.add_regular(
        plan,
        db,
        f"blk.{layer}.ffn_gate_inp.weight",
        f"{prefix}.gate.weight",
        q.QTYPE_F32,
        "router",
    )
    q.add_regular(
        plan,
        db,
        f"blk.{layer}.exp_probs_b.bias",
        f"{prefix}.gate.e_score_correction_bias",
        q.QTYPE_F32,
        "router",
    )
    expert_qtype = (
        q.QTYPE_Q2_K
        if provisional_q2k or layer == MTP_BLOCK
        else q.QTYPE_IQ2_XXS
    )
    for part in ("gate", "up", "down"):
        add_experts(plan, db, layer, part, expert_qtype)
    for part in ("gate", "up", "down"):
        q.add_regular(
            plan,
            db,
            f"blk.{layer}.ffn_{part}_shexp.weight",
            f"{prefix}.shared_experts.{part}_proj.weight",
            q.QTYPE_Q8_0,
            "shared_expert",
        )


def build_plan(db, config, provisional_q2k=False):
    plan = []
    q.add_regular(plan, db, "token_embd.weight", "model.embed_tokens.weight", q.QTYPE_Q8_0, "embedding")
    indexer_types = config["indexer_types"]
    if len(indexer_types) != SPEC["trunk_layers"]:
        q.fail(f"expected {SPEC['trunk_layers']} indexer types, got {len(indexer_types)}")

    for layer in range(BLOCK_COUNT):
        prefix = source_prefix(layer)
        q.add_regular(
            plan,
            db,
            f"blk.{layer}.attn_norm.weight",
            f"{prefix}.input_layernorm.weight",
            q.QTYPE_F32,
            "norm",
        )
        add_attention(plan, db, layer, indexer_types)
        q.add_regular(
            plan,
            db,
            f"blk.{layer}.ffn_norm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            q.QTYPE_F32,
            "norm",
        )
        add_ffn(plan, db, layer, provisional_q2k)
        if layer == MTP_BLOCK:
            q.add_regular(
                plan,
                db,
                f"blk.{layer}.nextn.eh_proj.weight",
                f"{prefix}.eh_proj.weight",
                q.QTYPE_Q8_0,
                "mtp",
            )
            for target, source in (
                ("enorm", "enorm.weight"),
                ("hnorm", "hnorm.weight"),
                ("shared_head_norm", "shared_head.norm.weight"),
            ):
                q.add_regular(
                    plan,
                    db,
                    f"blk.{layer}.nextn.{target}.weight",
                    f"{prefix}.{source}",
                    q.QTYPE_F32,
                    "mtp",
                )

    q.add_regular(plan, db, "output_norm.weight", "model.norm.weight", q.QTYPE_F32, "output")
    q.add_regular(plan, db, "output.weight", "lm_head.weight", q.QTYPE_Q8_0, "output")

    names = [item.name for item in plan]
    # 3 dense layers x 18 tensors, MoE layers x 23 tensors, 4 MTP-only tensors
    # (eh_proj, enorm, hnorm, shared_head_norm), embedding + output norm +
    # output.  1809 for the official checkpoint, 1782 without the MTP block.
    expected = 3 * 18 + (BLOCK_COUNT - 3) * 23 + (4 if MTP_BLOCK is not None else 0) + 3
    if len(names) != expected:
        q.fail(f"full GLM-5.3 plan has {len(names)} tensors, expected {expected}")
    if len(names) != len(set(names)):
        q.fail("duplicate GGUF tensor names in full GLM-5.3 plan")
    offset = 0
    for item in plan:
        item.offset = offset
        offset += q.align(item.nbytes)
    return plan


def model_metadata(hf_dir, source_revision, model_name="GLM-5.3", repo_url=None):
    with open(os.path.join(hf_dir, "chat_template.jinja"), "rb") as fp:
        chat_template = fp.read()
    if repo_url is None:
        repo_url = "https://huggingface.co/zai-org/GLM-5.3"
    return [
        q.kv_string("general.architecture", "glm-dsa"),
        q.kv_string("general.name", model_name),
        q.kv_string("general.basename", model_name),
        q.kv_string("general.version", "5.3"),
        q.kv_string("general.license", "glm-5.3"),
        q.kv_string("general.source.repo_url", repo_url),
        q.kv_string("general.source.revision", source_revision),
        q.kv_u32("general.alignment", q.GGUF_ALIGNMENT),
        q.kv_u32("general.quantization_version", 2),
        q.kv_u32("glm-dsa.block_count", BLOCK_COUNT),
        q.kv_u32("glm-dsa.nextn_predict_layers", SPEC["nextn_predict_layers"]),
        q.kv_u64("glm-dsa.context_length", 1048576),
        q.kv_u32("glm-dsa.embedding_length", 6144),
        q.kv_u32("glm-dsa.vocab_size", 154880),
        q.kv_u32("glm-dsa.feed_forward_length", 12288),
        q.kv_u32("glm-dsa.expert_feed_forward_length", 2048),
        q.kv_u32("glm-dsa.expert_count", EXPERT_COUNT),
        q.kv_u32("glm-dsa.expert_used_count", 8),
        q.kv_u32("glm-dsa.expert_shared_count", 1),
        q.kv_u32("glm-dsa.leading_dense_block_count", 3),
        q.kv_u32("glm-dsa.expert_group_count", 1),
        q.kv_u32("glm-dsa.expert_group_used_count", 1),
        q.kv_u32("glm-dsa.expert_gating_func", 2),
        q.kv_f32("glm-dsa.expert_weights_scale", 2.5),
        q.kv_bool("glm-dsa.expert_weights_norm", True),
        q.kv_f32("glm-dsa.attention.layer_norm_rms_epsilon", 1.0e-5),
        q.kv_u32("glm-dsa.attention.head_count", 64),
        q.kv_u32("glm-dsa.attention.head_count_kv", 1),
        q.kv_u32("glm-dsa.attention.key_length", 576),
        q.kv_u32("glm-dsa.attention.value_length", 512),
        q.kv_u32("glm-dsa.attention.key_length_mla", 256),
        q.kv_u32("glm-dsa.attention.value_length_mla", 256),
        q.kv_u32("glm-dsa.attention.q_lora_rank", 2048),
        q.kv_u32("glm-dsa.attention.kv_lora_rank", 512),
        q.kv_u32("glm-dsa.attention.indexer.head_count", 32),
        q.kv_u32("glm-dsa.attention.indexer.key_length", 128),
        q.kv_u32("glm-dsa.attention.indexer.top_k", 2048),
        q.kv_u32("glm-dsa.rope.dimension_count", 64),
        q.kv_f32("glm-dsa.rope.freq_base", 8000000.0),
        q.kv_string("tokenizer.chat_template", chat_template),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official zai-org/GLM-5.3 FP8 snapshot")
    parser.add_argument("--tokenizer-template", required=True, help="compatible GLM GGUF tokenizer source")
    parser.add_argument("--out", help="output GGUF")
    parser.add_argument("--imatrix", help="DwarfStar/llama.cpp importance matrix")
    parser.add_argument(
        "--provisional-q2k",
        action="store_true",
        help="use Q2_K for every routed expert to build a fast calibration model",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", default="e0b07fd2751b42d5efa199cc02c2b271deadc516")
    parser.add_argument("--model-name", default="GLM-5.3", help="general.name / general.basename")
    parser.add_argument("--repo-url", help="general.source.repo_url (defaults to zai-org/GLM-5.3)")
    parser.add_argument("--quants-library", help="path to libds4quants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 64:
        parser.error("--threads must be between 1 and 64")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is used")
    return args


def main():
    args = parse_args()
    with open(os.path.join(args.hf, "config.json"), "rb") as fp:
        config = json.load(fp)
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        q.fail(f"unexpected architecture: {config.get('architectures')!r}")
    spec = configure_layout(config)
    print(
        f"glm53-full-quantize: layout blocks={BLOCK_COUNT} mtp_block={MTP_BLOCK} "
        f"routed_experts={EXPERT_COUNT}",
        file=sys.stderr,
    )
    db = q.SourceDB(args.hf, functools.partial(validate_glm53_full_index, spec=spec))
    try:
        plan = build_plan(db, config, args.provisional_q2k)
        tokenizer_records, template_tokens = q.load_tokenizer_records(args.tokenizer_template)
        q.validate_tokenizer_template(args.hf, template_tokens, 154880)
        kv_records = model_metadata(args.hf, args.source_revision, args.model_name, args.repo_url)
        if args.dry_run:
            q.print_plan(plan, kv_records, tokenizer_records)
        else:
            q.write_gguf(args, plan, kv_records, tokenizer_records, db)
            print(f"glm53-full-quantize: wrote {args.out}", file=sys.stderr)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"glm53-full-quantize: error: {error}", file=sys.stderr)
        sys.exit(1)
