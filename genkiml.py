from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import onnx
import tensorflow as tf
import tf2onnx
import torch


def handle_keras_model(f_path: Path, opset: int):
    # Hack because I'm too lazy to figure out how to read directly from IOstream
    model = tf.keras.models.load_model(f_path)

    # Use from_function for tf functions
    model_onnx, _ = tf2onnx.convert.from_keras(
        model, input_signature=[tf.TensorSpec(shape=model.input.shape, dtype=model.input.dtype, name="x")], opset=opset
    )

    return model_onnx


def handle_onnx_model(f_path, opset):
    model_onnx = onnx.load_model(f_path)
    return model_onnx


def handle_torch_model(f_path, opset, input_shape):
    model = torch.load(f_path)
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError(
            "Please compile your model e.g. using `torch.jit.trace`. Only jit compiled PyTorch models are supported"
        )

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        print(f"Input shape: {input_shape}")
        torch.onnx.export(model, torch.randn(input_shape), f.name, opset_version=opset)
        model_onnx = onnx.load(f.name)

    return model_onnx


def handle_model_using_extension(f_path: Path, opset: int, input_shape=None):
    ext = f_path.suffix

    if ext in (".keras", ".h5") or f_path.is_dir():
        model_onnx = handle_keras_model(f_path, opset)
    elif ext == ".onnx":
        model_onnx = handle_onnx_model(f_path, opset)
    elif ext in (".pt", ".pth"):
        if input_shape is None:
            raise ValueError(
                "For a PyTorch model, please provide input shape for an example using the `--input-shape` argument"
            )
        model_onnx = handle_torch_model(f_path, opset, input_shape)
    else:
        raise ValueError(f"Unsupported model type {ext}")
    return model_onnx


def package_into_zip(path_to_copy: Path, model_onnx, output_path, out_name: str = "genkiml_cpp.zip") -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # Copy all the necessary files
        tmp_dir_path = Path(tmp_dir_name)
        dst_path = tmp_dir_path / path_to_copy.name
        shutil.copytree(path_to_copy, dst_path)

        model_onnx_path = tmp_dir_path / "model.onnx"
        onnx.save(model_onnx, model_onnx_path)
        # Copy model
        models_path = dst_path / "model"
        models_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_onnx_path, models_path)

        file_path = output_path / out_name
        with ZipFile(file_path, "w") as zip_obj:
            for f_name in tmp_dir_path.rglob("*"):
                zip_obj.write(f_name, arcname=f_name.relative_to(tmp_dir_path))
    return file_path


def main(model_path: str, opset: int, input_shape: tuple[int], output_path: str, model_only: bool) -> None:
    print("Converting model")
    model_onnx = handle_model_using_extension(Path(model_path), opset, input_shape)
    print("Packaging c++ library")

    output_path = Path(output_path if output_path is not None else ".")
    if model_only:
        file_path = output_path / "model.onnx"
        onnx.save(model_onnx, file_path)
    else:
        file_path = package_into_zip(Path(__file__).parent / "genki_ml", model_onnx, output_path)
    print(f"Exported to {file_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=None,
        help="List of integers representing the shape of the input tensor in the model",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Where to output the resulting C++ runtime")
    parser.add_argument("--model-only", action="store_true", help="Export the model without the C++ runtime")
    parser.add_argument("--onnx-opset", type=int, default=13, help="Which ONNX opset to use")
    args = parser.parse_args()

    main(args.model_path, args.onnx_opset, args.input_shape, args.output_path, args.model_only)
