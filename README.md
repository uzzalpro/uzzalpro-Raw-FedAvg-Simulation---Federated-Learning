# Raw FedAvg Simulation: Distributed Linear Regression

This project demonstrates the **Federated Averaging (FedAvg) algorithm** from scratch. It simulates a decentralized machine learning process by coordinating one central **Server** and two independent **Clients** to collaboratively train a simple Linear Regression model without ever sharing their raw data.

The core objective is to visualize the weighted average calculation, the fundamental mathematical operation of FedAvg.

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