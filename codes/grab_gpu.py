import os
import subprocess
import time
import argparse
from multiprocessing import Process, Value, Lock, Array

def get_gpu_mem(gpu_id):
    gpu_query = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in gpu_query.decode('utf-8').split('\n')[:-1]]
    return gpu_memory[gpu_id]

def get_free_gpus()->list:
    gpu_query = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in gpu_query.decode('utf-8').split('\n')[:-1]]
    free_gpus = [i for i, mem in enumerate(gpu_memory) if mem < 100]
    return free_gpus

def occupy_gpu(gpu_id:int, n, occupy_num, ocpy_gpus, lock, a_dim=100000):
    with lock:
        if get_gpu_mem(gpu_id) < 100 and occupy_num.value < n:
            import torch
            a = torch.ones((a_dim,a_dim)).cuda(gpu_id)
            ocpy_gpus[occupy_num.value]= gpu_id
            occupy_num.value += 1
            print(f"Occupying GPU {gpu_id}, Total Occupied: {occupy_num.value}")
    while True:
        time.sleep(10)

def occupy_all_gpus(n:int, occupy_num, ocpy_gpus, interval=10):
    print("Launching process to occupy GPU ...")
    lock = Lock()
    processes = [] #List to store the processes
    
    while occupy_num.value < n:
        free_gpus = get_free_gpus()
        will_occupy_num = min(n, max(0,len(free_gpus)))
        for i in range(will_occupy_num):
            if occupy_num.value < n:
                p = Process(target=occupy_gpu, args=(free_gpus[i], n, occupy_num, ocpy_gpus, lock))
                p.start()
                processes.append(p)
            
        time.sleep(interval) # enough time to occupy gpus and update nvidia-smi
    
    return processes, ocpy_gpus

def run_my_program(n, desired_script, processes, ocpy_gpus, occupy_num):
    for p in processes:
        p.terminate()
    ocpy_gpus_list = list(ocpy_gpus[:occupy_num.value])
    cuda_visible_devices = ",".join(map(str, ocpy_gpus_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    subprocess.run([desired_script, str(n)])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Occupy GPUs")
    parser.add_argument(
        "--n", type=int, default=2, help="Number of GPUs to occupy"
    )
    parser.add_argument(
        "--otime", type=int, default=10, help="Time of occupying gpu" 
    )
    parser.add_argument(
        "--spath", type=str, default='./train_lora.sh', help="the execute script path"
    )
    args = parser.parse_args()
    n = args.n
    occupy_time = args.otime
    desired_script = args.spath
    occupy_num = Value('i', 0) # Shared variable to count occupied GPUs
    ocpy_gpus =  Array('i', [-1 for _ in range(8)])# Shared array to store occupied gpu

    processes,ocpy_gpus = occupy_all_gpus(n, occupy_num, ocpy_gpus)
    time.sleep(occupy_time)
    run_my_program(n, desired_script, processes, ocpy_gpus, occupy_num)


    
