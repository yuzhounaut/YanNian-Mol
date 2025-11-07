"""
Example demonstrating configuration usage.

This example shows how to:
1. Create a default configuration
2. Load configuration from YAML
3. Create configuration from dictionary
4. Save configuration to file
5. Access configuration values
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import lifespan_predictor
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: This example requires pyyaml and pydantic to be installed
# Install with: pip install pyyaml pydantic

try:
    from lifespan_predictor.config import Config

    print("=" * 60)
    print("Configuration Example")
    print("=" * 60)

    # 1. Create default configuration
    print("\n1. Creating default configuration...")
    config = Config()
    print(f"   Default batch size: {config.training.batch_size}")
    print(f"   Default learning rate: {config.training.learning_rate}")
    print(f"   Default random seed: {config.random_seed}")

    # 2. Create configuration from dictionary
    print("\n2. Creating configuration from dictionary...")
    custom_config = Config.from_dict(
        {
            "data": {"train_csv": "custom_train.csv", "output_dir": "custom_results"},
            "training": {"batch_size": 64, "max_epochs": 50, "learning_rate": 0.001},
            "random_seed": 123,
        }
    )
    print(f"   Custom batch size: {custom_config.training.batch_size}")
    print(f"   Custom learning rate: {custom_config.training.learning_rate}")
    print(f"   Custom random seed: {custom_config.random_seed}")

    # 3. Access derived values
    print("\n3. Accessing derived values...")
    total_fp_dim = config.get_total_fp_dim()
    print(f"   Total fingerprint dimension: {total_fp_dim}")
    print(
        f"   (Morgan: {config.featurization.morgan_nbits} + "
        f"RDKit: {config.featurization.rdkit_fp_nbits} + "
        f"MACCS: {config.featurization.maccs_nbits})"
    )

    device = config.get_device()
    print(f"   Device: {device}")

    # 4. Save configuration to file
    print("\n4. Saving configuration to file...")
    output_path = "example_config.yaml"
    custom_config.save(output_path)
    print(f"   Configuration saved to: {output_path}")

    # 5. Load configuration from file
    print("\n5. Loading configuration from file...")
    loaded_config = Config.from_yaml(output_path)
    print(f"   Loaded batch size: {loaded_config.training.batch_size}")
    print(f"   Loaded learning rate: {loaded_config.training.learning_rate}")

    # 6. Convert to dictionary
    print("\n6. Converting configuration to dictionary...")
    config_dict = loaded_config.to_dict()
    print(f"   Dictionary keys: {list(config_dict.keys())}")

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"\n   Cleaned up: {output_path}")

    print("\n" + "=" * 60)
    print("Configuration example completed successfully!")
    print("=" * 60)

except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install pyyaml pydantic")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
