# Machine Unlearning with NCF

This project explores machine unlearning techniques with a recommendation system baseline using the Neutral Collaborative Filtering (NCF) model on the KuaiRec dataset.

The trained weight of an example NCF model can be found at [huggingface.co/seoyuny/social_media_ncf](https://huggingface.co/seoyuny/social_media_ncf).
The KuaiRec dataset can be found at [kuairec.com](https://kuairec.com/).

# Getting Started

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Visual Studio Code](https://code.visualstudio.com/)
*   [Dev Containers extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Setup

This project is configured to run inside a Docker container using VS Code's Dev Containers feature. This provides a consistent and reproducible development environment.

1.  **Clone the repository.**
2.  **Open the cloned folder in VS Code.**
3.  **Reopen in Container:** Click the notification in the bottom-right corner to reopen the project in the dev container. This will build the environment and install all dependencies.

For manual setup instructions, see the "Manual Setup" section below.

## End-to-End Workflow

Follow these steps to replicate the baseline model training.

### 1. Data Setup

1.  Download the **KuaiRec 2.0** dataset. You will need the `big_matrix.csv` file.
2.  Create a directory named `data` in the root of this project.
3.  Place the `big_matrix.csv` file inside a nested directory structure: `/app/data/KuaiRec 2.0/data/big_matrix.csv`.

The `/data/` directory is ignored by Git, so your dataset will not be committed.

### 2. Prepare Datasets

Run the data preparation script. This script processes the raw `big_matrix.csv` into training and test sets (`.pt` files) that are saved in the `/artifacts` directory.

```bash
python src/data/prepare_dataset.py
```

### 3. Train the Baseline Model

Run the baseline training script. This will train the NCF model using the processed datasets.

```bash
python src/training/train_baseline.py
```

### 4. Check the Outputs

*   The best performing model will be saved to `models/ncf_best.pth`.
*   Detailed training progress is logged to `training.log`.

## Project Structure

*   `.devcontainer/`: Configuration for the VS Code development container.
*   `artifacts/`: Stores all generated files (processed datasets, etc.). This is ignored by Git.
*   `data/`: Stores the raw KuaiRec dataset. This is ignored by Git.
*   `models/`: Stores trained model weights (`.pth` files). This is ignored by Git.
*   `src/`: Contains all Python source code:
    *   `data/`: Scripts for data processing and preparation.
    *   `models/`: The PyTorch `NCF` model definition.
    *   `training/`: Scripts for model training. Includes the main `train_baseline.py` and a `train_template.py` for experiments.
    *   `utils/`: Common utility functions, such as evaluation metrics.
    *   `test_environment.py`: A simple script to verify the container setup.
*   `Dockerfile`, `docker-compose.yml`, `requirements.txt`: Files that define the Docker container environment.
*   `training.log`: Log file for the baseline training script. Ignored by Git.

## Manual Setup (Alternative)

If you prefer not to use the VS Code Dev Containers feature, you can build and run the container manually.

1.  **Build and Run the Container:**
    ```bash
    docker-compose up -d --build
    ```
2.  **Access the Container:**
    ```bash
    docker exec -it machine_unlearning bash
    ```
3.  **Verify the Environment:**
    To ensure all libraries are installed correctly, run the verification script:
    ```bash
    python3 src/test_environment.py
    ```

## Further Reading
Once the setup is complete, see the README files in `src/data/README.md` and `src/training/README.md` for more detailed information about the data preparation and training scripts.

