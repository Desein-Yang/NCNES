import torch.multiprocessing as mp
import time

def say(ms1,ms2):
    start = time.time()
    print("pop"+str(ms1)+"ind"+str(ms2))
    time.sleep(ms1)
    print("Task run time %s" % str(time.time()-start))
    return ms1,ms2


if __name__ == "__main__":
    jobs = None
    results = []
    pool = mp.Pool(3)
    ms = 1
    for i in range(3):
        jobs = [
            pool.apply_async(say,(i,j,)) for j in range(3)
        ]
    for j in jobs:
        results.append(j.get())
    print(results)
    
