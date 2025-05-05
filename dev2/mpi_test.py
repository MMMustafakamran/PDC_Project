from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get hostname
import socket
hostname = socket.gethostname()

# Print rank and hostname
print(f"Rank {rank} running on {hostname}")

# Barrier to synchronize
comm.Barrier()

if rank == 0:
    print("\nAll processes completed!") 