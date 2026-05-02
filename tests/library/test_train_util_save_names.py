import argparse

import pytest

from library import train_util


def make_args(output_name=None, intermediate_save_name_format=None):
    return argparse.Namespace(
        output_name=output_name,
        intermediate_save_name_format=intermediate_save_name_format,
    )


def test_default_intermediate_checkpoint_names_match_existing_formats():
    args = make_args(output_name="model")

    assert train_util.get_epoch_ckpt_name(args, ".safetensors", 1) == "model-000001.safetensors"
    assert train_util.get_step_ckpt_name(args, ".ckpt", 1) == "model-step00000001.ckpt"


def test_custom_intermediate_name_format_for_epoch_and_step_dirs():
    args = make_args(output_name="run", intermediate_save_name_format="{model_name}-{save_type}-{number:04d}")

    assert train_util.get_epoch_intermediate_save_name(args, 3, step_no=120) == "run-epoch-0003"
    assert train_util.get_step_intermediate_save_name(args, 120, epoch_no=3) == "run-step-0120"


def test_intermediate_name_format_rejects_path_separators():
    args = make_args(output_name="run", intermediate_save_name_format="{model_name}/{save_type}-{number}")

    with pytest.raises(ValueError, match="not a path"):
        train_util.get_epoch_intermediate_save_name(args, 1)


@pytest.mark.parametrize("name_format", [".", "..", "C:run"])
def test_intermediate_name_format_rejects_non_plain_names(name_format):
    args = make_args(output_name="run", intermediate_save_name_format=name_format)

    with pytest.raises(ValueError, match="plain file or directory name"):
        train_util.get_epoch_intermediate_save_name(args, 1)


def test_intermediate_name_format_rejects_counter_from_other_save_type():
    epoch_args = make_args(output_name="run", intermediate_save_name_format="{model_name}-{step:08d}")
    step_args = make_args(output_name="run", intermediate_save_name_format="{model_name}-{epoch:06d}")
    nested_step_args = make_args(output_name="run", intermediate_save_name_format="{model_name}-{number:{epoch}d}")

    with pytest.raises(ValueError, match="field \\{step\\} for epoch save"):
        train_util.get_epoch_intermediate_save_name(epoch_args, 3, step_no=120)

    with pytest.raises(ValueError, match="field \\{epoch\\} for step save"):
        train_util.get_step_intermediate_save_name(step_args, 120, epoch_no=3)

    with pytest.raises(ValueError, match="field \\{epoch\\} for step save"):
        train_util.get_step_intermediate_save_name(nested_step_args, 120, epoch_no=3)


def test_checkpoint_extension_is_added_separately_from_custom_format():
    args = make_args(output_name="run", intermediate_save_name_format="{model_name}-{save_type}-{number:02d}")

    assert train_util.get_step_ckpt_name(args, ".safetensors", 12, epoch_no=2) == "run-step-12.safetensors"
