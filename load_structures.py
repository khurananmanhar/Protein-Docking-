import numpy as np
from Bio.PDB import PDBParser
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# ML Model
# -------------------------
class DockingMLP(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Load protein + ligand
# -------------------------
def load_protein_and_ligand(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    protein_coords = []
    ligand_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "HOH":
                    continue
                if residue.id[0] != " ":
                    for atom in residue:
                        ligand_coords.append(atom.get_coord())
                else:
                    for atom in residue:
                        protein_coords.append(atom.get_coord())

    protein_coords = np.array(protein_coords)
    ligand_coords = np.array(ligand_coords)

    print("Protein atoms:", len(protein_coords))
    print("Ligand atoms:", len(ligand_coords))

    return protein_coords, ligand_coords

# -------------------------
# Random pose generation
# -------------------------
def random_rotation_matrix():
    rand = np.random.rand(3)
    theta = rand[0] * 2 * np.pi
    phi = rand[1] * 2 * np.pi
    z = rand[2]
    r = np.sqrt(z)
    V = np.array([
        np.cos(phi) * r,
        np.sin(phi) * r,
        np.sqrt(1 - z)
    ])
    H = np.eye(3) - 2 * np.outer(V, V)
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return -H @ R

def generate_random_pose(ligand_coords, box_size=10.0):
    center = ligand_coords.mean(axis=0)
    centered = ligand_coords - center
    R = random_rotation_matrix()
    rotated = centered @ R.T
    translation = np.random.uniform(-box_size, box_size, size=3)
    translated = rotated + translation
    return translated

def generate_multiple_poses(ligand_coords, n_poses=100, box_size=10.0):
    return [generate_random_pose(ligand_coords, box_size) for _ in range(n_poses)]

# -------------------------
# Feature extraction
# -------------------------
def extract_pose_features(protein_coords, ligand_coords):
    distances = np.linalg.norm(
        protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
        axis=2
    )
    features = [
        distances.mean(),
        distances.min(),
        distances.max(),
        np.sum(distances < 4),
        np.sum(distances < 3)
    ]
    return np.array(features, dtype=np.float32)

# -------------------------
# Contact score (labels)
# -------------------------
def simple_contact_score(protein_coords, ligand_coords, cutoff=4.0):
    score = 0
    for p in protein_coords:
        distances = np.linalg.norm(ligand_coords - p, axis=1)
        score += np.sum(distances < cutoff)
    return score


# -------------------------
# Experimental binding data loader
# -------------------------
def load_binding_data(csv_path, id_column="id", affinity_column="affinity"):
    """
    Load experimental binding affinities from a CSV file.

    Expects a header with at least two columns: an identifier column (default
    'id') and an affinity/label column (default 'affinity'). The identifier is
    used to match structure filenames (base name without extension) to their
    experimental value.

    Returns a dict mapping id -> float(affinity). Non-parsable values become
    `np.nan`.
    """
    data = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if id_column not in row or affinity_column not in row:
                raise ValueError(f"CSV must contain columns '{id_column}' and '{affinity_column}'")
            key = row[id_column].strip()
            val = row[affinity_column].strip()
            try:
                valf = float(val)
            except Exception:
                valf = np.nan
            data[key] = valf

    print(f"Loaded binding data entries: {len(data)} from {csv_path}")
    return data

# -------------------------
# ML scoring
# -------------------------
def score_poses_ml(protein_coords, poses, model):
    scores = []
    for pose in poses:
        features = extract_pose_features(protein_coords, pose)
        features_tensor = torch.tensor(features).unsqueeze(0)
        score = model(features_tensor).item()
        scores.append(score)
    return np.array(scores)

def get_top_poses(poses, scores, top_n=5):
    idx = np.argsort(scores)[::-1]
    top_poses = [poses[i] for i in idx[:top_n]]
    top_scores = scores[idx[:top_n]]
    return top_poses, top_scores

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Paths (users should replace these with their real files)
    protein_path = "protein.pdb"

    # Load structure from PDB that contains both protein and ligand
    protein_coords, ligand_coords = load_protein_and_ligand(protein_path)

    # Try to load experimental binding data (CSV). The CSV should contain
    # an `id` column matching the base filename (without extension) of the
    # protein/complex and an `affinity` column with the experimental value.
    binding_csv = "binding.csv"
    binding_data = {}
    try:
        binding_data = load_binding_data(binding_csv, id_column="id", affinity_column="affinity")
    except FileNotFoundError:
        print(f"Warning: binding CSV not found at {binding_csv}. Falling back to synthetic labels.")
    except Exception as e:
        print(f"Warning: could not load binding CSV: {e}. Falling back to synthetic labels.")

    # Determine sample id from protein filename (e.g., 'protein' from 'protein.pdb')
    sample_id = os.path.splitext(os.path.basename(protein_path))[0]
    experimental_affinity = binding_data.get(sample_id, None)
    if experimental_affinity is None or (isinstance(experimental_affinity, float) and np.isnan(experimental_affinity)):
        print(f"No experimental affinity found for id='{sample_id}'. Using synthetic contact-score labels.")
        use_experimental = False
    else:
        print(f"Using experimental affinity for id='{sample_id}': {experimental_affinity}")
        use_experimental = True

    # Generate random poses to train the model
    n_train_poses = 200
    poses = generate_multiple_poses(ligand_coords, n_train_poses)

    # Extract features and labels. If experimental data is available we use the
    # same experimental affinity for all generated poses of this complex. If
    # not, we fall back to the previous synthetic contact score per-pose.
    features_list = []
    labels_list = []
    for pose in poses:
        features = extract_pose_features(protein_coords, pose)
        if use_experimental:
            label = float(experimental_affinity)
        else:
            label = simple_contact_score(protein_coords, pose)  # synthetic label
        features_list.append(features)
        labels_list.append(label)

    X = torch.tensor(np.array(features_list), dtype=torch.float32)
    y = torch.tensor(np.array(labels_list), dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create model
    model = DockingMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # -------------------------
    # Training loop
    # -------------------------
    n_epochs = 50
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("Training complete!")

    # -------------------------
    # Generate test poses and score
    # -------------------------
    test_poses = generate_multiple_poses(ligand_coords, n_poses=100)
    ml_scores = score_poses_ml(protein_coords, test_poses, model)
    top_poses, top_scores = get_top_poses(test_poses, ml_scores, top_n=5)

    print("Top 5 ML predicted scores after training:", top_scores)
