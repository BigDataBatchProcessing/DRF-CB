from drf_scheduler import Simulation
from data_model import Node, Application
import numpy as np

# PRZYKŁAD 2

# --- Definicja zasobów ---
# Załóżmy 2 zasoby: [CPU, Pamięć GB]
RES_CPU = 0
RES_MEM = 1

# --- Definicja Klastra (Węzły) ---
# 2 węzły, każdy [4 CPU, 8 GB RAM]
node1 = Node(id=1, R_k=np.array([4.0, 8.0]))
node2 = Node(id=2, R_k=np.array([4.0, 8.0]))
cluster_nodes = [node1, node2]

# --- Definicja Aplikacji PRZYKŁAD 2 ---
# Aplikacja A - 1: (1 CPU, 8GB, 16s)
app1 = Application(
    id=1,
    task_prototype={'requirements': np.array([1.0, 4.0]), 'duration': 5.0}
)
# Aplikacja B - 2: (4 CPU, 2GB, 20s)
app2 = Application(
    id=2,
    task_prototype={'requirements': np.array([2.0, 2.0]), 'duration': 10.0}
)
# Aplikacja C - 3: (2 CPU, 4GB, 60s)
app3 = Application(
    id=3,
    task_prototype={'requirements': np.array([1.0, 2.0]), 'duration': 3.0}
)
cluster_apps = [app1, app2, app3]

# --- Definicja Kolejki Zdarzeń (Żądania) ---
# (czas, app_id, liczba_zadań)
submission_queue = [
    (0.0, 1, 3), # W czasie 0, app 1 chce 10 małych zadań
    (0.0, 2, 2),  # W czasie 5, app 2 chce 4 duże zadania
    (2.0, 3, 2)  # W czasie 10, app 1 chce kolejne 5 małych zadań
]

# --- Uruchomienie Symulacji ---
print("--- Rozpoczynam symulację ---")

# Ustawiamy wagi wywłaszczania (np. koszt jest 2x ważniejszy niż zysk)
sim = Simulation(
    nodes=cluster_nodes, 
    apps=cluster_apps, 
    submission_queue=submission_queue,
    preemption_alpha=50.0,
    preemption_beta=10.0 
)

sim.run()

print("\n--- Symulacja zakończona ---")
print(f"Całkowity czas: {sim.current_time:.2f}")
print("Stan końcowy aplikacji:")
for app in sim.apps.values():
    print(f"  Aplikacja {app.id}: s_i = {app.s_i:.3f}, Uruchomione: {len(app.running_tasks)}, Oczekujące: {len(app.pending_tasks)}")
print("Stan końcowy węzłów:")
for node in sim.nodes.values():
    print(f"  Węzeł {node.id}: Użycie C_k = {node.C_k} / {node.R_k}")