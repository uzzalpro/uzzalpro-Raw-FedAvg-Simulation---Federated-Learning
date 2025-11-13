# Raw FedAvg Simulation: Distributed Linear Regression

This project demonstrates the **Federated Averaging (FedAvg) algorithm** from scratch, without external machine learning frameworks (like **PyTorch or TensorFlow**). It simulates a decentralized machine learning process by coordinating one central **Server** and two independent **Clients** to collaboratively train a simple Linear Regression model without ever sharing their raw data.

The core objective is to visualize the weighted average calculation, the fundamental mathematical operation of FedAvg.

This project uses **Python**, **Flask**, and **NumPy** to handle the network communication and the core mathematical aggregation process.

## 🎯 Project Goal

To visually and mathematically demonstrate the FedAvg algorithm, showing how a global model is trained by averaging weight updates from decentralized, locally-trained models (Clients) that hold private data.

## 📁 Project Structure

The project is organized into three separate directories, one for each independent application:

```bash
    ├── client_1/
    │   ├── client_1_app.py     # Client 1: Flask app with private data (y=2x+1) running on port 5001
    │   └── pyproject.toml      # Client 1 dependencies (Flask, NumPy)
    ├── client_2/
    │   ├── client_2_app.py     # Client 2: Flask app with private data (y=x+1) running on port 5002
    │   └── pyproject.toml
    └── server/
        ├── server_app.py         # Server: Flask app for orchestration and aggregation running on port 5000
        └── pyproject.toml        # Server dependencies (Flask, NumPy, Requests)
```

## ⚙️ Project Architecture and Components

The system consists of three independent Python applications, communicating via Flask APIs:

| Component | Port | Role | Core Logic |
| :--- | :--- | :--- | :--- |
| **Client 1** (`client_1_app.py`) | `5001` | Local Trainer | Holds private data. Performs **Gradient Descent** locally when requested by the Server. |
| **Client 2** (`client_2_app.py`) | `5002` | Local Trainer | Holds different private data. Performs **Gradient Descent** locally. |
| **Server** (`server_app.py`) | `5000` | Orchestrator & Aggregator | Coordinates global rounds. Executes the **FedAvg weighted average** math to update the global model. |

## 🛠️ Prerequisites

To run this simulation locally, you need the following:

1.  **Python (3.10+):** Installed and accessible on your system.
2.  **Poetry:** Used for managing dependencies and creating isolated virtual environments.
    ```bash
    pip install poetry
    ```

## 🚀 Step-by-Step Run Instructions

The simulation requires three separate terminals to run the applications and a fourth terminal to send the command that starts the process.

### Step 1: Organize and Install Dependencies

Ensure your project files are organized into three folders (`client_1`, `client_2`, `server`) and run the setup command in each directory.

| App Directory | Terminal | Command to Run |
| :--- | :--- | :--- |
| `client_1` | **Terminal 1** | `cd client_1 && poetry install` |
| `client_2` | **Terminal 2** | `cd client_2 && poetry install` |
| `server` | **Terminal 3** | `cd server && poetry install` |

### Step 2: Start All Application Servers

Once dependencies are installed, start each application using Poetry's environment runner.

| App Directory | Terminal | Command to Run |
| :--- | :--- | :--- |
| `client_1` | **Terminal 1** | `cd client_1 && poetry run python client_1_app.py` |
| `client_2` | **Terminal 2** | `cd client_2 && poetry run python client_2_app.py` |
| `server` | **Terminal 3** | `cd server && poetry run python server_app.py` |

### Step 3: Trigger the Federated Learning Process

Open a **new, fourth terminal** and use `curl` to send a POST request to the server's `/start` endpoint. This command begins the sequence of global rounds.

```bash
curl -X POST [http://127.0.0.1:5000/start](http://127.0.0.1:5000/start)
```

## 📈 Monitoring and Validation

After running the curl command, monitor Terminal 3 (Server). You will see the complete, round-by-round process log.

The server log will display the FedAvg Aggregation Math, showing the calculation of the new global weights using the weighted average formula:

* $W_{t+1}$ = $\sum_{k=1}^{K} \frac{n_k}{N} W_k$
* $W_{t+1}$ is the New Global Model.
* $n_k$ is the data size of the Client (always 2 in this example).
* $N$ is the total data size of all participating Clients (always 4 in this example).
* $W_k$ is the trained model from the Client.

Observe how the final global weights $w$ and $b$ converge over 5 rounds to a compromise model.
