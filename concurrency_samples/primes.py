import threading
import multiprocessing
import time


# Target function: Find prime numbers
def find_primes(n):
    """Find all prime numbers up to n."""
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for divisor in range(2, int(num ** 0.5) + 1):
            if num % divisor == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes


# Wrapper function for threading
def thread_task(end):
    print(f"Thread starting: {threading.current_thread().name}")
    find_primes(end)
    print(f"Thread ending: {threading.current_thread().name}")


# Wrapper function for multiprocessing
def process_task(end):
    print(f"Process starting: {multiprocessing.current_process().name}")
    find_primes(end)
    print(f"Process ending: {multiprocessing.current_process().name}")


# Main function
if __name__ == "__main__":
    n = 500000  # Define the upper limit for prime number calculation

    # MULTITHREADING
    print("\n--- Multithreading Example ---")
    start_time = time.time()

    threads = []
    chunk_size = n // 4  # Divide range into 4 chunks
    for i in range(4):  # 4 threads
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size
        thread = threading.Thread(target=thread_task, args=(end, ))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Multithreading took {time.time() - start_time:.2f} seconds")

    # MULTIPROCESSING
    print("\n--- Multiprocessing Example ---")
    start_time = time.time()

    processes = []
    chunk_size = n // 4  # Divide range into 4 chunks
    for i in range(4):  # 4 processes
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size
        process = multiprocessing.Process(target=process_task, args=(end, ))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(f"Multiprocessing took {time.time() - start_time:.2f} seconds")
