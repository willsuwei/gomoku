setlocal
cd /d %~dp0

call activate tensorflow-gpu_1-15

mpiexec -np 8 python -u train_mpi.py
REM mpiexec -np 6 python -u train_mpi.py --no_trainer