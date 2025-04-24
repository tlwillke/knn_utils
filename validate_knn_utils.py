import os
import subprocess
import tempfile
import unittest
import numpy as np

# Helper to write fvec files.
def write_fvecs(fname, arr):
    """
    Write a numpy array (shape: n x d) to an fvec file.
    Each vector is stored as: [d, float, float, ..., float]
    The dimension is written as an int32.
    """
    n, d = arr.shape
    with open(fname, "wb") as f:
        for i in range(n):
            # Write the dimension as int32
            np.array([d], dtype=np.int32).tofile(f)
            # Write the vector data as float32
            arr[i].astype(np.float32).tofile(f)

# Helper to read ivec files.
def read_ivecs(fname):
    """
    Read an ivec file into a numpy array.
    Each record is stored as: [k, int, int, ..., int]
    """
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    ivecs = []
    i = 0
    while i < len(data):
        k = data[i]
        i += 1
        vec = data[i : i + k]
        ivecs.append(vec)
        i += k
    return np.array(ivecs)

# Brute force nearest neighbor computation.
def brute_force_nn(base, query, k, metric="l2"):
    results = []
    if metric in ("l2", "euclidean"):
        for q in query:
            dists = np.linalg.norm(base - q, axis=1)
            idx = np.argsort(dists)[:k]
            results.append(idx)
    elif metric == "cosine":
        # Assume vectors are normalized.
        for q in query:
            sims = np.dot(base, q)
            idx = np.argsort(-sims)[:k]
            results.append(idx)
    else:
        raise ValueError("Unsupported metric: " + metric)
    return np.array(results)

class TestComputeGroundTruthScript(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to hold test files.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_file = os.path.join(self.tmpdir.name, "base.fvecs")
        self.query_file = os.path.join(self.tmpdir.name, "query.fvecs")
        self.output_file = os.path.join(self.tmpdir.name, "output.ivecs")
        # Path to the script (assumed to be in the same directory as this test file).
        self.script_path = os.path.join(os.path.dirname(__file__), "knn_utils.py")
        # Create a small synthetic dataset that avoids ties.
        dim = 1000
        num_base = 1000
        num_query = 10
        np.random.seed(12345)
        self.base = (np.random.rand(num_base, dim) * 10).astype(np.float32)  # base vectors
        self.query = (np.random.rand(num_query, dim) * 10).astype(np.float32)  # query vectors
        self.k = 4

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_script(self, metric):
        """
        Write out the fvec files, run the compute_ground_truth.py script with
        the given metric (l2 or cosine), and return the nearest neighbor indices
        read from the output ivec file.
        """
        # Write the base and query datasets in fvec format.
        write_fvecs(self.base_file, self.base)
        write_fvecs(self.query_file, self.query)
        cmd = [
            "python",
            self.script_path,
            "--base", self.base_file,
            "--query", self.query_file,
            "--output", self.output_file,
            "--k", str(self.k),
            "--gpus", "-1",  # Use CPU mode for testing.
            "--metric", metric
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.fail(f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        # Read and return the output ivec file.
        nn_indices = read_ivecs(self.output_file)
        return nn_indices

    def test_l2_metric(self):
        # Test the L2 (Euclidean) metric.
        nn_indices = self.run_script("l2")
        expected = brute_force_nn(self.base, self.query, self.k, metric="l2")
        for i in range(self.query.shape[0]):
            self.assertEqual(
                set(nn_indices[i].tolist()),
                set(expected[i].tolist()),
                f"L2 metric test failed for query {i}: expected {expected[i]}, got {nn_indices[i]}"
            )

    def test_cosine_metric(self):
        # For cosine similarity, the vectors must be normalized.
        def normalize(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return arr / norms
        self.base = normalize(self.base)
        self.query = normalize(self.query)
        nn_indices = self.run_script("cosine")
        expected = brute_force_nn(self.base, self.query, self.k, metric="cosine")
        for i in range(self.query.shape[0]):
            self.assertEqual(
                set(nn_indices[i].tolist()),
                set(expected[i].tolist()),
                f"Cosine metric test failed for query {i}: expected {expected[i]}, got {nn_indices[i]}"
            )


if __name__ == '__main__':
    unittest.main()
