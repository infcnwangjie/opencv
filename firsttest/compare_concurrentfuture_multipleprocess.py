# 1.同步执行--------------
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os, time, random


def task(n):
    # print('[%s] is running'%os.getpid())
    # time.sleep(random.randint(1,3))  #I/O密集型的，，一般用线程，用了进程耗时长
    return n ** 2


def use_concurrent_futures():
    '''使用concurrent.futures的进程池'''
    start = time.time()
    p = ProcessPoolExecutor()
    for i in range(1000):  # 现在是开了10个任务， 那么如果是上百个任务呢，就不能无线的开进程，那么就得考虑控制
        # 线程数了，那么就得考虑到池了
        obj = p.submit(task, i).result()  # 相当于apply同步方法
    p.shutdown()  # 相当于close和join方法
    print('=' * 30)
    print(time.time() - start)  # 17.36499309539795


def use_multipleprocess_pool():
    '''使用原生进程池'''
    start = time.time()
    from multiprocessing import Process, Pool
    p = Pool()
    for i in range(1000):  # 现在是开了10个任务， 那么如果是上百个任务呢，就不能无线的开进程，那么就得考虑控制
        # 线程数了，那么就得考虑到池了
        # obj = p.apply_async(task, (i,)).get()  # 相当于apply同步方法
        obj = p.apply(task, (i,))  # 相当于apply同步方法
    # p.shutdown()  # 相当于close和join方法
    print('=' * 30)
    print(time.time() - start)  # 17.36499309539795


if __name__ == '__main__':
    # use_concurrent_futures()  # cost 0.7176012992858887
    use_multipleprocess_pool()  # cost 0.32760047912597656
