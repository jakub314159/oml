import threading
import multiprocessing
import time


# Task: Compute squares of numbers
def compute_squares(numbers):
    results = [n**2 for n in numbers]
    return results


def thread_task(numbers):
    print(f"Thread starting: {threading.current_thread().name}")
    compute_squares(numbers)
    print(f"Thread ending: {threading.current_thread().name}")


def process_task(numbers):
    print(f"Process starting: {multiprocessing.current_process().name}")
    compute_squares(numbers)
    print(f"Process ending: {multiprocessing.current_process().name}")


# Main function
if __name__ == "__main__":
    numbers = range(1, 10**6)  # Large dataset for demonstration

    # MULTITHREADING
    print("\n--- Multithreading Example ---")
    start_time = time.time()

    threads = []
    for i in range(4):  # 4 threads
        thread = threading.Thread(target=thread_task, args=(numbers,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Multithreading took {time.time() - start_time:.2f} seconds")

    # MULTIPROCESSING
    print("\n--- Multiprocessing Example ---")
    start_time = time.time()

    processes = []
    for i in range(4):  # 4 processes
        process = multiprocessing.Process(target=process_task, args=(numbers,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(f"Multiprocessing took {time.time() - start_time:.2f} seconds")
