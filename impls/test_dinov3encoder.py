import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

# OGBench imports
from utils.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset, HGCDataset

# Import your encoder
from utils.encoders import DinoV3AdapterEncoder  # adjust path if needed

# Import agent config
from agents import gcbc  # THIS loads agents/gciql.py as a Python module


def main():

    # --- Load the agent config exactly like main.py ---
    config = gcbc.get_config()  # This function exists in all agent config files
    config = config.to_dict()    # Convert ml_collections.ConfigDict to plain Python dict

    env_name = "visual-antmaze-medium-navigate-v0"
    frame_stack = config['frame_stack']

    # --- Load dataset exactly like main.py ---
    env, train_dataset_raw, _ = make_env_and_datasets(env_name, frame_stack=frame_stack)

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]

    train_dataset = dataset_class(Dataset.create(**train_dataset_raw), config)

    # --- Sample real data ---
    batch = train_dataset.sample(8)
    obs = batch["observations"]

    # --- Init encoder ---
    encoder = DinoV3AdapterEncoder()
    variables = encoder.init(jax.random.PRNGKey(0), obs)

    # --- Compute features ---
    feats = encoder.apply(variables, obs)
    print("Feature mean =", float(feats.mean()))
    print("Feature std  =", float(feats.std()))

    # --- Pixel shift test ---
    img0 = obs[0]
    img_shift = jnp.roll(img0, 1, axis=1)
    f0 = encoder.apply(variables, img0[None])
    f1 = encoder.apply(variables, img_shift[None])
    print("Pixel shift difference =", float(jnp.linalg.norm(f0 - f1)))

    # --- Temporal test ---
    f_prev = encoder.apply(variables, obs[0:1])
    f_next = encoder.apply(variables, obs[1:2])
    print("Temporal difference =", float(jnp.linalg.norm(f_prev - f_next)))


if __name__ == "__main__":
    main()
