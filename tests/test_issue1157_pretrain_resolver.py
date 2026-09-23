"""#1157: pin the pretrain trainer's LoRA target resolver call.

The pretrain path already calls the shared resolver, but the existing tests
mock the MoE target helper and therefore do not fail if the resolver call is
deleted. This test drives a real SoupConfig through PretrainTrainerWrapper's
own setup order and spies on the shared resolver call.
"""

from unittest.mock import MagicMock, patch

from soup_cli.config.schema import SoupConfig


def test_pretrain_setup_pins_lora_target_resolver_call():
    """The pretrain trainer must resolve target_modules before building PEFT config."""
    from soup_cli.trainer.pretrain import PretrainTrainerWrapper
    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    cfg = SoupConfig(
        base="some-model",
        task="pretrain",
        data={"train": "./data.jsonl"},
        training={
            "quantization": "none",
            "lora": {
                "r": 4,
                "alpha": 8,
                "target_modules": "auto",
            },
        },
    )

    model = MagicMock()
    model.config.model_type = "llama"
    model.get_nb_trainable_parameters.return_value = (100, 1000)

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
        patch("soup_cli.utils.moe.detect_moe_model", return_value=False),
        patch(
            "soup_cli.utils.moe.resolve_moe_lora_targets",
            side_effect=lambda _model, _tcfg, targets, _console: targets,
        ),
        patch(
            "soup_cli.utils.peft_wiring.resolve_lora_target_modules",
            wraps=resolve_lora_target_modules,
        ) as resolve_targets,
        patch(
            "soup_cli.utils.peft_wiring.build_lora_config",
            return_value=MagicMock(),
        ),
        patch("peft.get_peft_model", return_value=model),
        patch("peft.prepare_model_for_kbit_training"),
        patch("soup_cli.utils.peft_wiring.apply_lisa_setup", return_value=False),
        patch("soup_cli.utils.block_expansion.apply_block_expansion_if_configured"),
        patch("soup_cli.utils.peft_wiring.apply_pre_lora_patches"),
        patch("soup_cli.utils.peft_wiring.apply_post_lora_patches"),
    ):
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
        wrapper._setup_transformers(cfg, cfg.training)

    resolve_targets.assert_called_once_with(model, "auto")
