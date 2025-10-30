import numpy as np
from enum import Enum
from dataclasses import dataclass, field

# Używamy numpy do łatwych operacji na wektorach zasobów
ResourceVector = np.ndarray

# --- Typy i klasy pomocnicze ---

class TaskStatus(Enum):
    PENDING = 1
    RUNNING = 2
    FINISHED = 3

@dataclass
class Task:
    """Reprezentuje pojedyncze zadanie (instancję prototypu)"""
    id: int
    app_id: int
    requirements: ResourceVector # Wektor zasobów (np. [cpu, mem, gpu])
    duration: float
    status: TaskStatus = TaskStatus.PENDING
    start_time: float = -1.0
    node_id: int = -1
    
    def elapsed_time(self, current_time: float) -> float: 
        if self.status == TaskStatus.RUNNING and self.start_time >= 0:
            return current_time - self.start_time
        return 0.0

@dataclass
class Application:
    """Reprezentuje aplikację/użytkownika"""
    id: int
    task_prototype: dict # {'requirements': ResourceVector, 'duration': float}
    pending_tasks: list[Task] = field(default_factory=list)
    running_tasks: dict[int, Task] = field(default_factory=dict) # {task_id: Task}
    
    # U_i: Całkowite zasoby przydzielone aplikacji
    U_i: ResourceVector = field(init=False)
    # s_i: Poziom dominacji
    s_i: float = 0.0

    def __post_init__(self):
        # Inicjalizujemy U_i na wektor zerowy o wymiarze zasobów
        self.U_i = np.zeros_like(self.task_prototype['requirements'])

    def update_dominant_share(self, total_cluster_resources: ResourceVector):
        """Aktualizuje s_i na podstawie U_i i zasobów klastra"""
        # Unikamy dzielenia przez zero, jeśli zasób nie istnieje
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = self.U_i / total_cluster_resources
            # Zastępujemy NaN (wynik 0/0) i inf (wynik x/0) zerami
            shares[~np.isfinite(shares)] = 0  
        
        self.s_i = np.max(shares) if shares.size > 0 else 0.0

@dataclass
class Node:
    """Reprezentuje węzeł w klastrze"""
    id: int
    R_k: ResourceVector # Pojemność węzła
    C_k: ResourceVector = field(init=False) # Aktualne użycie
    running_tasks: dict[int, Task] = field(default_factory=dict) # {task_id: Task}

    def __post_init__(self):
        self.C_k = np.zeros_like(self.R_k)

    def can_fit(self, task_req: ResourceVector) -> bool:
        """Sprawdza, czy zadanie zmieści się na tym węźle"""
        return np.all(self.C_k + task_req <= self.R_k)

    def add_task(self, task: Task):
        """Dodaje zadanie do węzła"""
        if not self.can_fit(task.requirements):
            raise ValueError(f"Zadanie {task.id} nie mieści się na węźle {self.id}")
        self.C_k += task.requirements
        self.running_tasks[task.id] = task

    def remove_task(self, task: Task):
        """Usuwa zadanie z węzła"""
        if task.id not in self.running_tasks:
            raise ValueError(f"Zadania {task.id} nie ma na węźle {self.id}")
        self.C_k -= task.requirements
        del self.running_tasks[task.id]