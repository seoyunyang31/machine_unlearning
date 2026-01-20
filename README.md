# Machine Unlearning Project

This project explores machine unlearning techniques with a recommendation system baseline using the NeuMF model on the KuaiRec dataset.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Visual Studio Code](https://code.visualstudio.com/)
*   [Dev Containers extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Setup

This project is configured to run inside a Docker container using VS Code's Dev Containers feature. This provides a consistent and reproducible development environment for all team members.

### Quick Start with VS Code

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Open in VS Code:**
    Open the cloned repository folder in Visual Studio Code.

3.  **Reopen in Container:**
    You will see a notification in the bottom-right corner asking if you want to "Reopen in Container". Click it.

    This will:
    *   Build the Docker image using the `Dockerfile`.
    *   Run the container using the `docker-compose.yml` configuration.
    *   Mount your local project directory into the `/app` directory in the container.
    *   Automatically install the required Python extensions (`ms-python.python`) inside the container as specified in `devcontainer.json`.
    *   Run `pip install -r requirements.txt` after the container is created to install all necessary Python packages.

You are now ready to work inside the development container. All commands, scripts, and applications will run within this isolated environment.

### Manual Setup (Alternative)

If you prefer not to use the VS Code Dev Containers feature directly, you can build and run the container manually.

1.  **Build and Run the Container:**
    From the root of the project directory, run:
    ```bash
    docker-compose up -d --build
    ```
    This will build the image and start the `app` service in the background.

2.  **Access the Container:**
    You can open a shell inside the running container:
    ```bash
    docker exec -it machine_unlearning bash
    ```
    You will be in the `/app` directory and can run the project scripts from there.

### Verify the Environment

Once the container is running, open a new terminal in VS Code (`Terminal > New Terminal`). To ensure all libraries are installed correctly and the environment is ready, run the simple verification script:

```bash
`python3 src/test_environment.py`
```

## Project Structure

*   `src/`: Contains all Python source code for data processing, model implementation, and training.
*   `data/`: Intended for datasets. This directory is in `.gitignore` and should not be committed.
*   `.devcontainer/`: Contains the configuration for the VS Code development container.
*   `Dockerfile`: Defines the Docker image for the development environment.
*   `docker-compose.yml`: Defines the Docker services and their configuration.
*   `requirements.txt`: Lists the Python dependencies for the project.

*   `artifacts/`: Stores all generated files (processed datasets, model weights, unlearning indices). Ignored by git.
*   `data/`: Contains the raw KuaiRec dataset. Ignored by git.
*   `src/`: Contains all Python source code, organized into packages:
    *   `data/`: Scripts for data processing and generation (``prepare_dataset.py``, ``generate_unlearning_indices.py``).
    *   `models/`: The PyTorch `NCF` model definition.
    *   `training/`: Scripts for training and evaluation (``train_baseline.py``).
    *   `utils/`: Common utility functions (e.g., ``metrics.py``).
    *   ``test_environment.py``: A simple script to verify the container setup.
*   `.devcontainer/`: Configuration for the VS Code development container.
*   `Dockerfile`, ``docker-compose.yml``, ``requirements.txt``: Define the container environment.

## Running the code
Once the setup is complete, you can execute the scripts within the container. For example, to run the training script:
```bash
python3 src/train_baseline.py --data_path data/"KuaiRec 2.0"/data/big_matrix.csv
```

