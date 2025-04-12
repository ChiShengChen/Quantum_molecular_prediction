import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # Keep numpy if needed by qml functions
import functools # Import functools
import torch.nn.init as init # Import init

print("Initializing Quantum Layers Module...")

# Helper function (needed here if applying init directly in this file)
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

# --- Enhanced Quantum Layer (for original QNN) ---
class EnhancedQuantumLayer(nn.Module):
    """
    Improved quantum layer with data re-uploading and trainable rotations
    """
    def __init__(self, n_qubits=8, n_layers=3, input_dim=128):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim # Store input_dim
        
        # Create a quantum device
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights_rx, weights_ry, weights_rz, weights_cz, final_rotations):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights_rx[l, i], wires=i)
                    qml.RY(weights_ry[l, i], wires=i)
                    qml.RZ(weights_rz[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, (i + 1) % n_qubits])
                if l % 2 == 1:
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * weights_cz[l, i], wires=i)
            for i in range(n_qubits):
                qml.RX(final_rotations[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Define weight shapes
        weight_shapes = {
            "weights_rx": (n_layers, n_qubits),
            "weights_ry": (n_layers, n_qubits),
            "weights_rz": (n_layers, n_qubits),
            "weights_cz": (n_layers, n_qubits),
            "final_rotations": n_qubits
        }
        
        # Create the TorchLayer
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Input projection (maps input_dim to n_qubits)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, n_qubits),
            nn.LayerNorm(n_qubits)
        )
        
        # Output projection (maps n_qubits back to input_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        # Apply input projection
        x_proj = self.input_proj(x)
        
        # Process through the quantum circuit
        results = []
        for sample in x_proj:
            # Ensure sample is on CPU for Pennylane default.qubit
            sample_cpu = sample.detach().cpu()
            q_result = self.qlayer(sample_cpu)
            results.append(q_result)
        
        # Stack results and move back to original device
        quantum_output = torch.stack(results).to(x.device)
        
        # Apply output projection
        return self.output_proj(quantum_output)

# --- QASA Model Quantum Components (Based on qasa_damped.py) ---

class DampedStyleQuantumLayer(nn.Module):
    """ 
    Quantum Layer structured similarly to qasa_damped.py's QuantumLayer.
    Adapted for tabular data (no explicit timestep input).
    Uses LayerNorm instead of BatchNorm.
    Includes projection for skip connection mismatch.
    """
    def __init__(self, input_dim, output_dim, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits # Number of primary qubits
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.num_wires = n_qubits + 1 # Use n_qubits + 1 wires like in qasa_damped.py

        # Define the device (try lightning first)
        try:
            qdevice_name = "lightning.qubit"
            self.qdevice = qml.device(qdevice_name, wires=self.num_wires)
            print(f"Using {qdevice_name} with {self.num_wires} wires for DampedStyleQuantumLayer.")
        except (qml.DeviceError, ImportError):
            qdevice_name = "default.qubit"
            print(f"Warning: Failed to init lightning.qubit. Using {qdevice_name} with {self.num_wires} wires for DampedStyleQuantumLayer.")
            self.qdevice = qml.device(qdevice_name, wires=self.num_wires)

        # Define QNode inside __init__ based on qasa_damped.py circuit
        @qml.qnode(self.qdevice, interface="torch", diff_method="parameter-shift")
        def _damped_style_circuit(inputs, weights):
            # inputs shape: (n_qubits,)
            # weights shape: (n_layers, n_qubits + 1)
            n_q = self.n_qubits # Primary qubits
            n_l = self.n_layers

            # Encoding (RX, RZ on first n_qubits)
            for i in range(n_q):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)

            # Layer 0 Rotations (RX, RZ on first n_qubits)
            for i in range(n_q):
                qml.RX(weights[0, i], wires=i)
                qml.RZ(weights[0, i], wires=i)

            # Subsequent Layers (Entanglement + Rotations)
            for l in range(1, n_l):
                # Standard CNOT ladder on primary qubits
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
                # Rotations (RY, RZ on primary qubits)
                for i in range(n_q):
                    qml.RY(weights[l, i], wires=i)
                    qml.RZ(weights[l, i], wires=i)
                
                # Interaction with auxiliary qubit
                qml.CNOT(wires=[n_q - 1, n_q]) # CNOT from last primary to auxiliary
                qml.RY(weights[l, n_q], wires=n_q) # Use the last weight element for the aux qubit
                qml.RZ(weights[l, n_q], wires=n_q) # Use the last weight element for the aux qubit

            # Measurement (only on the first n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

        # Define weight shape and TorchLayer
        self.weight_shape = (self.n_layers, self.n_qubits + 1)
        self.qlayer = qml.qnn.TorchLayer(_damped_style_circuit, {'weights': self.weight_shape})
        
        # Classical processing
        self.input_proj = nn.Linear(input_dim, self.n_qubits)
        self.norm = nn.LayerNorm(self.n_qubits) # Use LayerNorm
        self.output_proj = nn.Linear(self.n_qubits, output_dim)

        # Add projection layer for skip connection if dimensions might mismatch
        self.skip_proj = None
        if input_dim != output_dim:
            print(f"DampedStyleQuantumLayer: Input dim {input_dim} != Output dim {output_dim}. Adding skip projection.")
            self.skip_proj = nn.Linear(input_dim, output_dim)
            self.skip_proj.apply(init_weights) # Initialize the skip projection layer

        # Apply initialization to other linear layers
        self.input_proj.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(self, x):
        # x shape: (batch, input_dim)
        batch_size = x.size(0)
        
        # Input projection and normalization
        x_proj_tanh = torch.tanh(self.input_proj(x)) 
        # Use LayerNorm
        x_proj_norm = self.norm(x_proj_tanh)

        # Quantum processing (process samples individually)
        outputs = []
        for i in range(batch_size):
            q_in = x_proj_norm[i].detach().cpu() # Ensure input is on CPU for default.qubit
            # Call qlayer with only the input tensor
            q_out = self.qlayer(q_in).to(x.device)
            outputs.append(q_out)
        quantum_output = torch.stack(outputs)
        
        # Output projection
        out_proj = self.output_proj(quantum_output)
        
        # Residual connection (handle mismatch with projection)
        if x.shape[-1] == self.output_dim:
            return x + out_proj
        elif self.skip_proj is not None:
             # Apply projection to x before adding
             return self.skip_proj(x) + out_proj
        else:
            # Fallback if dimensions mismatch but no projection was created
            print(f"Warning: Skip connection failed in DampedStyleQuantumLayer. Input dim {x.shape[-1]} != Output dim {self.output_dim}. No projection layer. Returning quantum output only.")
            return out_proj

# --- Quantum LSTM Components ---

class QuantumLSTMCell(nn.Module):
    """
    Quantum LSTM Cell where parts of the gate calculations are replaced by a PQC.
    """
    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical linear layers for combined input+hidden -> PQC input projection
        self.input_hidden_proj = nn.Linear(input_size + hidden_size, n_qubits)
        
        # Define the PQC device
        try:
            qdevice_name = "lightning.qubit"
            self.qdevice = qml.device(qdevice_name, wires=self.n_qubits)
            print(f"Using {qdevice_name} with {self.n_qubits} wires for QuantumLSTMCell.")
        except (qml.DeviceError, ImportError):
            qdevice_name = "default.qubit"
            print(f"Warning: Failed to init lightning.qubit. Using {qdevice_name} with {self.n_qubits} wires for QuantumLSTMCell.")
            self.qdevice = qml.device(qdevice_name, wires=self.n_qubits)

        # Define the QNode inside __init__
        @qml.qnode(self.qdevice, interface="torch", diff_method="parameter-shift")
        def _quantum_lstm_gates_pqc(inputs, weights):
            # inputs shape: (n_qubits,)
            # weights shape: (n_layers, n_qubits, 3) # For RX, RY, RZ rotations
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')

            # Replace BasicEntanglerLayers with explicit loops and gates
            for l in range(self.n_layers):
                # Entanglement Layer (e.g., CNOT ladder)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 1: # Add CNOT between last and first for ring
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                    
                # Rotation Layer
                for i in range(self.n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)

            # Measure expectation values - output will have size n_qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Define weight shape and TorchLayer for the PQC
        # Back to shape 3 for RX, RY, RZ parameters
        self.weight_shape = (self.n_layers, self.n_qubits, 3)
        print(f"Using PQC weight shape: {self.weight_shape}") # Log the shape
        self.qlayer_gates = qml.qnn.TorchLayer(_quantum_lstm_gates_pqc, {'weights': self.weight_shape})

        # Classical layers to map PQC output and potentially input/hidden to LSTM gates
        # Output n_qubits -> 4 * hidden_size (for f_t, i_t, o_t, g_t)
        self.pqc_to_gates = nn.Linear(n_qubits + input_size + hidden_size, 4 * hidden_size)
        
        # Apply Kaiming init to classical layers
        self.input_hidden_proj.apply(init_weights)
        self.pqc_to_gates.apply(init_weights)

    def forward(self, x, states):
        # x: input tensor of shape (batch, input_size)
        # states: tuple (h_prev, c_prev) each of shape (batch, hidden_size)
        h_prev, c_prev = states
        batch_size = x.size(0)

        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # 1. Project combined vector for PQC input
        pqc_input_proj = torch.tanh(self.input_hidden_proj(combined))

        # 2. Run the PQC (process samples individually)
        pqc_outputs = []
        for i in range(batch_size):
            q_in = pqc_input_proj[i].detach().cpu() # Ensure input is on CPU if needed
            q_out = self.qlayer_gates(q_in).to(x.device)
            pqc_outputs.append(q_out)
        pqc_output = torch.stack(pqc_outputs) # Shape: (batch, n_qubits)

        # 3. Classical post-processing to get gates
        # Combine PQC output with original combined input/hidden features
        combined_for_gates = torch.cat((pqc_output, combined), dim=1)
        gates = self.pqc_to_gates(combined_for_gates)

        # Split the result into four gates/candidate state components
        f_t, i_t, o_t, g_t = torch.chunk(gates, 4, dim=1)

        # Apply activations
        f_t = torch.sigmoid(f_t) # Forget gate
        i_t = torch.sigmoid(i_t) # Input gate
        o_t = torch.sigmoid(o_t) # Output gate
        g_t = torch.tanh(g_t)    # Candidate cell state

        # 4. Calculate new cell state and hidden state
        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t