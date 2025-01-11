from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time


# Target function: Find prime numbers
def find_primes_in_range(start, end):
    """Find all prime numbers in the range [start, end]."""
    primes = []
    for num in range(start, end + 1):
        is_prime = True
        for divisor in range(2, int(num**0.5) + 1):
            if num % divisor == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes


# Wrapper function for multiprocessing
def process_task(kwargs):
    print(f"Process starting: {multiprocessing.current_process().name}")
    primes = find_primes_in_range(kwargs.get("start"), kwargs.get("end"))
    print(f"Process ending: {multiprocessing.current_process().name}")
    return primes


# Main function
if __name__ == "__main__":
    n = 1_000_000  # Upper limit for prime number calculation
    num_processes = 4  # Number of processes to use
    chunk_size = n // num_processes  # Divide the range into chunks

    # Create ranges for each process
    ranges = [(i * chunk_size + 1, (i + 1) * chunk_size) for i in range(num_processes)]

    # MULTIPROCESSING
    print("\n--- Multiprocessing Example ---")
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_task, [{"start": t[0], "end": t[1]} for t in ranges]))

    print(f"Multiprocessing took {time.time() - start_time:.2f} seconds")
    # print(f"Primes: {results}")

    start_time = time.time()
    find_primes_in_range(1, n)
    print(f"Single process: {time.time() - start_time:.2f}")
