import itertools
from dataclasses import dataclass, field

event_counter = itertools.count()

# --- Typy zdarzeń symulacji ---

@dataclass
class SimEvent:
    """Bazowa klasa zdarzenia w symulacji"""
    time: float
    # priorytet, aby zdarzenia o tym samym czasie były przetwarzane poprawnie
    priority: int = 0
    counter: int = field(init=False)
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        # Przypisz unikalne, rosnące ID każdemu zdarzeniu
        self.counter = next(event_counter)
    
    def __lt__(self, other):
        """Definiuje kolejność sortowania dla heapq: (time, priority, counter)"""
        # 1. Porównaj czas
        if self.time != other.time:
            return self.time < other.time
        
        # 2. Porównaj priorytet (niższa liczba to wyższy priorytet, więc `<` to większy priorytet)
        if self.priority != other.priority:
            return self.priority < other.priority
            
        # 3. Rozstrzygnij remisy za pomocą unikalnego licznika (najważniejsze!)
        # Nigdy nie dotrze do payload!
        return self.counter < other.counter

@dataclass
class SubmitEvent(SimEvent):
    """Zdarzenie: nadejście nowych zadań"""
    priority: int = 1 # Wyższy priorytet niż uruchomienie schedulera
    # UWAGA: Usunięto zbędną re-definicję pola 'payload'

@dataclass
class TaskFinishEvent(SimEvent):
    """Zdarzenie: zakończenie się zadania"""
    priority: int = 1 # Taki sam jak Submit
    # UWAGA: Usunięto zbędną re-definicję pola 'payload'

@dataclass
class SchedulerRunEvent(SimEvent):
    """Zdarzenie: uruchomienie pętli szeregowania"""
    priority: int = 2 # Niższy priorytet, aby wykonać się po Submit/Finish
    # (Tu 'payload' poprawnie nie był re-definiowany)