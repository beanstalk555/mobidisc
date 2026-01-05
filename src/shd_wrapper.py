import subprocess
import tempfile
import os
from typing import List, Optional
import platform
SHD_DIR = "./bin/shd"

if platform.system() == "Windows":
    SHD_EXECUTABLE = SHD_DIR + "/windows_x86_64/shd.exe"
else:
    SHD_EXECUTABLE = SHD_DIR + "/linux_x86_64/shd"


class SHDWrapper:
    """Wrapper for the SHD (Sparsity-based Hypergraph Dualization) C program."""
    
    def __init__(self, shd_binary_path: str = SHD_EXECUTABLE):
        """
        Initialize the SHD wrapper.
        
        Args:
            shd_binary_path: Path to the compiled shd binary
        """
        self.shd_binary_path = shd_binary_path
        if not os.path.exists(shd_binary_path):
            raise FileNotFoundError(f"SHD binary not found at {shd_binary_path}")
    
    def write_input_file(self, cnf_clauses: List[List[int]], filename: str) -> None:
        """
        Write CNF clauses to input file in the format expected by SHD.
        
        The input format is:
        - Each line represents a transaction (clause)
        - Numbers on each line are the items (literals) in that transaction
        
        Args:
            cnf_clauses: List of clauses, where each clause is a list of integers
            filename: Output filename
        """
        with open(filename, 'w') as f:
            for clause in cnf_clauses:
                f.write(' '.join(map(str, clause)) + '\n')
    
    def parse_output_file(self, filename: str) -> List[List[int]]:
        """
        Parse the output file from SHD.
        
        Args:
            filename: Input filename
            
        Returns:
            List of minimal hitting sets
        """
        hitting_sets = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    hitting_set = [int(x) for x in line.split()]
                    hitting_sets.append(sorted(hitting_set))
        return sorted(hitting_sets, key=lambda x: (len(x), x))
    
    def find_minimal_hitting_sets(
        self,
        cnf_clauses: List[List[int]],
        mode: str = "0",
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
        verbose: bool = False
    ) -> List[List[int]]:
        """
        Find minimal hitting sets for the given CNF clauses.
        
        Args:
            cnf_clauses: List of clauses (each clause is a list of integers)
            mode: SHD mode string (e.g., "0", "9D", "Dc")
            min_size: Minimum size of hitting sets (-l option)
            max_size: Maximum size of hitting sets (-u option)
            max_solutions: Stop after finding this many solutions (-S option)
            verbose: Show progress (V option)
            
        Returns:
            List of minimal hitting sets
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as input_file:
            input_filename = input_file.name
            self.write_input_file(cnf_clauses, input_filename)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as output_file:
            output_filename = output_file.name
        
        try:
            # Build command
            cmd = [self.shd_binary_path, mode, input_filename, output_filename]
            
            # Add optional parameters
            if min_size is not None:
                cmd.extend(['-l', str(min_size)])
            if max_size is not None:
                cmd.extend(['-u', str(max_size)])
            if max_solutions is not None:
                cmd.extend(['-S', str(max_solutions)])
            if verbose:
                cmd[1] = cmd[1] + 'V'
            
            # Execute SHD
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if verbose:
                print(result.stdout)
            
            # Parse output
            hitting_sets = self.parse_output_file(output_filename)
            return hitting_sets
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SHD execution failed: {e.stderr}")
        finally:
            # Cleanup temporary files
            if os.path.exists(input_filename):
                os.remove(input_filename)
            if os.path.exists(output_filename):
                os.remove(output_filename)