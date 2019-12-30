source ./venv/bin/activate

mpiexec -np 2 python -u train_mpi.py
