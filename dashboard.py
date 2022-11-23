from __future__ import annotations

import tempfile
from pathlib import Path

import netron
import streamlit as st
import tensorflow as tf

from exporting_functionality import handle_model_using_extension, save_model, package_into_zip


# Also possible to use the command line to do this:
#
# `python -m tf2onnx.convert --saved-model path/to/savedmodel --output dst/path/model.onnx --opset 13`
# `python -m tf2onnx.convert --tflite path/to/model.tflite --output dst/path/model.onnx --opset 13`


def model_summary_str(model) -> str:
    """model.summary prints out by default. This fetches the info into a string and returns it"""
    vals = []
    model.summary(print_fn=lambda x: vals.append(x))
    return "\n".join(vals)


def get_model_tf() -> tf.keras.Model:
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=(100,)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(2)
    ])


def get_model_tf_large() -> tf.keras.Model:
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=(100,)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(2)
    ])


def main():
    opset = st.selectbox("opset", [9, 11, 13], index=2)
    model_uploaded = st.file_uploader("Add a model")

    if model_uploaded is None:
        return

    with tempfile.NamedTemporaryFile(suffix=model_uploaded.name) as f:
        f.write(model_uploaded.getvalue())
        try:
            model_onnx = handle_model_using_extension(f.name, opset)
        except ValueError:
            st.write(f"Unsupported model type, {f.name}")
            return

    save_model(model_onnx)

    if st.button("View model graph"):
        netron.start("model.onnx")

    path_to_copy = Path("genki_ml")
    out_file = package_into_zip(path_to_copy)

    st.write("Model has been compiled, press button to download")
    with open(out_file, "rb") as f:
        st.download_button("Download code", f, file_name=out_file, mime="application/octet-stream")


if __name__ == '__main__':
    main()
