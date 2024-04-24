import numpy as np
import csv
from collections import deque

# RNG for exponential distribution
class CustomRandomGenerator:
    def __init__(self, seed=None):
        self.seed = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32
        self.current = seed

    def rand(self):
        self.current = (self.a * self.current + self.c) % self.m
        return self.current / self.m

def erlang(rng, k, lambd):
    result = 0
    for _ in range(k):
        result += -np.log(1 - rng.rand()) / lambd
    return result

def normal(mean, std_dev, rng):
    u1 = rng.rand()
    u2 = rng.rand()
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + std_dev * z

# Define model parameters
lambda1 = 3
lambda2 = 2
mu1 = 2
mu2 = 1.5
service_time1 = 0
service_time2 = 0
server_status = 0  # 0 - free, 1 - busy
queue = deque()  # Task queue (stack)
current_time = 0  # Current time
simulation_time = 500  # Simulation time
rng = CustomRandomGenerator(seed=78727)  # Random number generator

# Statistics for calculating MOE1 and MOE2
downtime_times = []  # Server downtime times
queue_lengths = []  # Queue lengths at different time points

# Function to print current simulation status
def print_simulation_status(event_type, current_time, j1, j2, server_status, queue):
    print("Time:", current_time, "Event:", event_type, "J1:", j1 if j1 != float('inf') else "-", "J2:",
          j2 if j2 != float('inf') else "-", "Server status:", "Busy" if server_status == 1 else "Free",
          "Queue length:", len(queue))

# Start simulation
print("Simulation started.")
print_simulation_status("Start", current_time, "-", "-", server_status, queue)

# Simulation events
simulation_data = []  # For saving simulation data
while current_time < simulation_time:
    # Generate time until arrival of next task for each type
    interarrival_time1 = erlang(rng, 2, lambda1)
    interarrival_time2 = np.random.normal(2, 0.2)
    j1 = current_time + interarrival_time1 if current_time + interarrival_time1 < simulation_time else float('inf')
    j2 = current_time + interarrival_time2 if current_time + interarrival_time2 < simulation_time else float('inf')

    # Determine service time for the current task
    if server_status == 0:
        service_time1 = np.random.normal(2, 0.15)
        service_time2 = np.random.normal(1.5, 0.5)

    # Choose the event that occurs first
    min_time = min(j1, j2, service_time1, service_time2)
    event_type = None
    if j1 == min_time:
        event_type = "J1"
    elif j2 == min_time:
        event_type = "J2"
    elif service_time1 == min_time:
        event_type = "E1"
    elif service_time2 == min_time:
        event_type = "E2"

    # Update current time
    current_time += min_time

    # Recalculate interarrival and service times based on the updated current time
    if j1 != float('inf'):
        j1 -= min_time
    if j2 != float('inf'):
        j2 -= min_time
    if server_status == 1:
        service_time1 -= min_time
        service_time2 -= min_time

    # Update server status and queue
    if event_type == "J1":
        queue.appendleft("J1")  # Add to the front of the queue
    elif event_type == "J2":
        queue.appendleft("J2")  # Add to the front of the queue

    # Service tasks in the queue if the server is free
    if server_status == 0:
        if queue:
            server_status = 1
            job_type = queue.pop()  # Remove the last element from the queue
            if job_type == "J1":
                service_time1 = np.random.normal(2, 1.5)  # Service time for Type J1 tasks
            elif job_type == "J2":
                service_time2 = np.random.normal(1.5, 0.5)  # Service time for Type J2 tasks
    elif server_status == 1:
        if not queue:
            server_status = 0

    # Update queue if server is busy and tasks are still arriving
    elif server_status == 1:
        if event_type == "J1" or event_type == "J2":
            queue.appendleft(event_type)  # Add to the front of the queue
            queue_lengths.append(len(queue))  # Update queue length
            # Update interarrival times for tasks in queue
            for i in range(len(queue)):
                if queue[i] == "J1":
                    j1 -= min_time
                elif queue[i] == "J2":
                    j2 -= min_time

    # Save data for statistics
    if server_status == 0:
        downtime_times.append(current_time)
    queue_lengths.append(len(queue))

    # Print current simulation status
    print_simulation_status(event_type, current_time, j1, j2, server_status, queue)

    # Save simulation data
    simulation_data.append(
        [current_time, event_type, j1, j2, "Busy" if server_status == 1 else "Free", len(queue), str(queue)])

# Print table with simulation results
print("\nSimulation process table template")
print("{:<2} {:<2} {:<2} {:<2} {:<2} {:<2} {:<2}".format("Num", "Time", "Event", "J1", "J2", "St", "S", "n", "Q"))

# Print first 10 rows
for i, row in enumerate(simulation_data[:10]):
    print("{:<2} {:<2} {:<2} {:<2} {:<2} {:<2} {:<2}".format(i + 1, *row))

# Print last 10 rows
print("...")
for i, row in enumerate(simulation_data[-10:], start=len(simulation_data) - 9):
    print("{:<2} {:<2} {:<2} {:<2} {:<2} {:<2} {:<2}".format(i, *row))

# Calculate MOE1 and MOE2 metrics
downtime_factor = len(downtime_times) / simulation_time
average_jobs_in_queue = sum(queue_lengths) / len(queue_lengths)

# Print MOE1 and MOE2 values
print("\nMOE1 Downtime factor:", downtime_factor)
print("MOE2 Average of jobs in queue:", average_jobs_in_queue)

# Save simulation data to a CSV file
with open('simulation_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Time", "Event", "J1", "J2", "Server status", "Queue length", "Queue"])
    writer.writerows(simulation_data)

print("\nSimulation results have been saved to simulation_results.csv")
