"""
Minimal client for the Psi0 inference server.

Usage:
    1. Start the server:
       bash scripts/deploy/serve_psi0_simple.sh \
           .runs/finetune/stackcups.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2604151753 30000

    2. Run this client:
       python examples/simple/client_example.py
"""

import json
import requests
import numpy as np
from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype

SERVER_URL = "http://localhost:22085"


# --- numpy serialization (matches server's helpers.py) ---

def numpy_serialize(arr: np.ndarray) -> dict:
    data = arr.data if arr.flags["C_CONTIGUOUS"] else arr.tobytes()
    return {
        "__numpy__": b64encode(data).decode(),
        "dtype": dtype_to_descr(arr.dtype),
        "shape": list(arr.shape),
    }


def numpy_deserialize(dct: dict):
    if "__numpy__" in dct:
        buf = b64decode(dct["__numpy__"])
        arr = np.frombuffer(buf, descr_to_dtype(dct["dtype"]))
        return arr.reshape(dct["shape"]) if dct["shape"] else arr[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: convert_numpy_in_dict(v, func) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


# --- API helpers ---

def health_check() -> bool:
    resp = requests.get(f"{SERVER_URL}/health")
    return resp.json().get("status") == "ok"


def predict_action(
    image: np.ndarray,
    state: np.ndarray,
    instruction: str = "stack cups",
    reset: bool = False,
) -> np.ndarray:
    """
    Send one observation to the server, get back predicted actions.

    Args:
        image: RGB image from head_camera, shape (H, W, 3), uint8
        state: proprioceptive state, shape (26,) float32
        instruction: task instruction string
        reset: True on the first step of an episode (resets RTC state)

    Returns:
        actions: shape (action_exec_horizon, 26), denormalized joint targets
    """
    payload = {
        "image": {"head_camera": image},
        "instruction": instruction,
        "history": {"reset": True} if reset else {},
        "state": {"states": state.reshape(1, -1)},  # (1, D) — observation_horizon=1
        "condition": {},
        "gt_action": np.zeros(1, dtype=np.float32),
        "dataset_name": "real",
        "timestamp": "0",
    }

    payload = convert_numpy_in_dict(payload, numpy_serialize)
    resp = requests.post(f"{SERVER_URL}/act", json=payload)
    result = convert_numpy_in_dict(resp.json(), numpy_deserialize)
    return result["action"]


# --- Example usage ---

if __name__ == "__main__":
    # 1. Health check
    print(f"Server healthy: {health_check()}")

    # 2. Send a dummy observation (replace with real camera + state data)
    dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_state = np.zeros(26, dtype=np.float32)

    # First step — pass reset=True to clear RTC history
    actions = predict_action(dummy_image, dummy_state, reset=True)
    print(f"Predicted actions shape: {actions.shape}")
    print(f"First action step: {actions[0]}")

    # Subsequent steps — reset=False (default)
    actions = predict_action(dummy_image, dummy_state)
    print(f"Next actions shape: {actions.shape}")
