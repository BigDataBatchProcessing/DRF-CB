from data_model import Task, Application, Node, TaskStatus, ResourceVector
from event_types import SubmitEvent, TaskFinishEvent, SchedulerRunEvent, SimEvent
import heapq
import itertools
import numpy as np

# --- Główna klasa Symulatora ---

class Simulation:
    def __init__(self, nodes: list[Node], apps: list[Application], submission_queue: list[tuple], 
                 preemption_alpha: float = 1.0, preemption_beta: float = 1.0, preemption_epsilon: float = 0.001):
        
        self.current_time = 0.0
        self.event_queue = [] # heapq (min-heap)
        
        self.nodes = {n.id: n for n in nodes}
        self.apps = {a.id: a for a in apps}
        
        # Całkowite zasoby klastra R_total
        self.R_total = np.sum([n.R_k for n in nodes], axis=0)
        
        self.task_id_counter = itertools.count()
        self.all_tasks = {} # Globalna mapa {task_id: Task}

        # Parametry wywłaszczania
        self.ALPHA = preemption_alpha
        self.BETA = preemption_beta
        self.EPSILON = preemption_epsilon

        # Załaduj początkowe zdarzenia
        for (time, app_id, num_tasks) in submission_queue:
            # Używamy float(time) na wszelki wypadek
            evt = SubmitEvent(time=float(time), payload={'app_id': app_id, 'num_tasks': num_tasks})
            heapq.heappush(self.event_queue, evt)

    def _calculate_s_i(self, U_vector: ResourceVector) -> float:
        """Oblicza hipotetyczny udział dominujący dla danego wektora użycia U_i"""
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = U_vector / self.R_total
            shares[~np.isfinite(shares)] = 0  # 0/0 -> 0, x/0 -> 0
        return np.max(shares) if shares.size > 0 else 0.0

    def _calculate_task_cost(self, task: Task) -> float:
        """Oblicza koszt wywłaszczenia dla pojedynczego zadania"""
        # t_p - czas, który zadanie już działało
        t_p = task.elapsed_time(self.current_time)
        if t_p <= 0:
            return 0.0 # Zadanie jeszcze nie ruszyło / właśnie ruszyło
            
        D_P = task.requirements
        
        # Znajdź dominujący zasób dla ZADANIA (d_i,dom / R_total,dom)
        with np.errstate(divide='ignore', invalid='ignore'):
            task_shares = D_P / self.R_total
            task_shares[~np.isfinite(task_shares)] = 0
        
        max_share = np.max(task_shares)
        if max_share == 0:
            return 0.0 # Zadanie nic nie zużywa

        # Twoja formuła: koszt = t_p * (d_i,dom / R_total,dom)
        # (d_i,dom / R_total,dom) to po prostu max_share
        koszt = t_p * max_share
        return koszt
    
    def run(self):
        """Główna pętla symulacji"""
        
        # Rozpocznij od próby szeregowania o czasie 0
        self.trigger_scheduler_run()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            print(f"\n--- Czas: {self.current_time:.2f}, Zdarzenie: {type(event).__name__} {event.payload} ---")

            if isinstance(event, SubmitEvent):
                self.handle_submit_event(event)
            elif isinstance(event, TaskFinishEvent):
                self.handle_task_finish_event(event)
            elif isinstance(event, SchedulerRunEvent):
                self.run_scheduler_cycle()
            
            # Zapobiegaj pętlom, jeśli scheduler nic nie robi
            if isinstance(event, SchedulerRunEvent) and not event.payload.get('allocation_made', False):
                 # Jeśli scheduler się uruchomił i nic nie zaalokował, 
                 # nie ma sensu uruchamiać go ponownie, dopóki nie przyjdzie nowe zdarzenie
                 pass
            elif not any(isinstance(e, SchedulerRunEvent) for e in self.event_queue):
                 # Jeśli po obsłudze zdarzenia (Submit/Finish) 
                 # nie ma zaplanowanego uruchomienia schedulera, uruchom go teraz.
                 self.trigger_scheduler_run()

    def trigger_scheduler_run(self, allocation_made=False):
        """Dodaje zdarzenie uruchomienia schedulera do kolejki"""
        # Sprawdź, czy już nie ma takiego zdarzenia o tym samym czasie
        if not any(isinstance(e, SchedulerRunEvent) and e.time == self.current_time for e in self.event_queue):
            # Użycie float() jako zabezpieczenie, choć current_time powinno być float
            evt = SchedulerRunEvent(time=float(self.current_time), payload={'allocation_made': allocation_made})
            heapq.heappush(self.event_queue, evt)

    def handle_submit_event(self, event: SubmitEvent):
        """Obsługa zdarzenia dodania nowych zadań"""
        app_id = event.payload['app_id']
        num_tasks = event.payload['num_tasks']
        app = self.apps[app_id]
        
        print(f"  Dodaję {num_tasks} zadań dla aplikacji {app_id}")
        for _ in range(num_tasks):
            task_id = next(self.task_id_counter)
            proto = app.task_prototype
            new_task = Task(id=task_id, app_id=app_id, 
                            requirements=proto['requirements'], 
                            duration=proto['duration'])
            app.pending_tasks.append(new_task)
            self.all_tasks[task_id] = new_task

    def handle_task_finish_event(self, event: TaskFinishEvent):
        """Obsługa zdarzenia zakończenia zadania (zwolnienie zasobów)"""
        task_id = event.payload['task_id']
        app_id = event.payload['app_id']
        node_id = event.payload['node_id']

        if task_id not in self.all_tasks:
            print(f"  OSTRZEŻENIE: Zakończone zadanie {task_id} nie znalezione (prawdopodobnie wywłaszczone).")
            return

        task = self.all_tasks[task_id]
        node = self.nodes[node_id]
        app = self.apps[app_id]

        print(f"  Kończy się zadanie {task_id} (Aplikacja {app_id}) na węźle {node_id}")
        
        try:
            # Zwolnij zasoby na węźle
            node.remove_task(task)
            
            # Zaktualizuj zasoby aplikacji
            app.U_i -= task.requirements
            del app.running_tasks[task_id]
            
            # Zaktualizuj s_i aplikacji
            app.update_dominant_share(self.R_total)
            
            # Usuń zadanie z globalnej listy
            task.status = TaskStatus.FINISHED
            del self.all_tasks[task_id]
            
        except ValueError as e:
            print(f"  BŁĄD przy zwalnianiu zasobów: {e}")


    def run_scheduler_cycle(self):
        """
        Implementacja Twojej głównej pętli 'while ALOKACJA_W_RUNDZIE'
        """
        print("  Uruchamiam cykl szeregowania...")
        
        while True: # Pętla 'while ALOKACJA_W_RUNDZIE'
            alokacja_w_rundzie = False
            
            # ... (logika sortowania aplikacji, taka jak poprzednio) ...
            
            pending_apps = [app for app in self.apps.values() if app.pending_tasks]
            if not pending_apps:
                print("    Brak oczekujących zadań. Koniec cyklu.")
                break 
                
            posortowane_aplikacje = sorted(pending_apps, key=lambda a: a.s_i)
            print(f"    Posortowane aplikacje (wg s_i): {[ (a.id, f'{a.s_i:.3f}') for a in posortowane_aplikacje ]}")

            for app in posortowane_aplikacje:
                if not app.pending_tasks:
                    continue
                
                zadanie_do_uruchomienia = app.pending_tasks[0]
                D_i = zadanie_do_uruchomienia.requirements

                # 1. ITERACJA PRZEZ WĘZŁY
                wezel_cel = self.find_best_node(D_i)

                # 2. JEŚLI ZNALEZIONO WĘZEŁ
                if wezel_cel is not None:
                    # ... (ta logika pozostaje BEZ ZMIAN) ...
                    
                    print(f"    ALOKACJA: Zadanie {zadanie_do_uruchomienia.id} (Aplikacja {app.id}) na węźle {wezel_cel.id}")
                    task = app.pending_tasks.pop(0)
                    task.status = TaskStatus.RUNNING
                    task.start_time = self.current_time
                    task.node_id = wezel_cel.id
                    
                    wezel_cel.add_task(task)
                    app.U_i += task.requirements
                    app.running_tasks[task.id] = task
                    app.update_dominant_share(self.R_total)
                    
                    finish_event = TaskFinishEvent(
                        time=self.current_time + task.duration,
                        payload={'task_id': task.id, 'app_id': app.id, 'node_id': wezel_cel.id}
                    )
                    heapq.heappush(self.event_queue, finish_event)
                    
                    alokacja_w_rundzie = True
                    break # Przerwij pętlę 'for' i zacznij nową rundę 'while'

                # 3. JEŚLI NIE ZNALEZIONO WĘZŁA (WĘZEŁ_CEL == null)
                else:
                    # BLOK ANALIZY WYWŁASZCZANIA
                    print(f"    Brak miejsca dla zadania {zadanie_do_uruchomienia.id} (Aplikacja {app.id}). Szukam kandydata do wywłaszczenia...")
                    
                    # *** POCZĄTEK ZMIAN ***
                    
                    # Przekazujemy całe zadanie, nie tylko jego wymagania
                    zestaw_ofiar, wezel_ofiary = self.find_preemption_candidate(app, zadanie_do_uruchomienia)
                    
                    if zestaw_ofiar: # Jeśli lista nie jest pusta
                        
                        victim_app_id = zestaw_ofiar[0].app_id
                        aplikacja_ofiary = self.apps[victim_app_id]
                        zasoby_zwolnione = np.zeros_like(self.R_total)

                        print(f"    WYWŁASZCZANIE: {len(zestaw_ofiar)} zadań (App {victim_app_id}) z węzła {wezel_ofiary.id} na rzecz zadania {zadanie_do_uruchomienia.id} (App {app.id})")

                        # 1. Dokonaj wywłaszczenia (aktualizacja C_k, U_p)
                        for ofiara_task in zestaw_ofiar:
                            print(f"      - Wywłaszczam zadanie {ofiara_task.id}")
                            wezel_ofiary.remove_task(ofiara_task)
                            zasoby_zwolnione += ofiara_task.requirements
                            del aplikacja_ofiary.running_tasks[ofiara_task.id]

                            # 2. Zwróć zadanie ofiary do kolejki oczekujących
                            ofiara_task.status = TaskStatus.PENDING
                            ofiara_task.start_time = -1.0
                            ofiara_task.node_id = -1
                            # Utracony czas jest zachowany w 'elapsed_time'
                            # Można by to śledzić, ale dla uproszczenia resetujemy
                            aplikacja_ofiary.pending_tasks.insert(0, ofiara_task)
                        
                        # Zaktualizuj U_p (poza pętlą, raz)
                        aplikacja_ofiary.U_i -= zasoby_zwolnione
                        
                        # 3. Zaktualizuj poziom dominacji ofiary P
                        aplikacja_ofiary.update_dominant_share(self.R_total)
                        
                        # 4. Uruchom zadanie i (zwycięzcy) w zwolnionym miejscu
                        task = app.pending_tasks.pop(0) # zadanie_do_uruchomienia
                        
                        task.status = TaskStatus.RUNNING
                        task.start_time = self.current_time
                        task.node_id = wezel_ofiary.id
                        
                        # Sprawdzenie bezpieczeństwa (powinno być OK dzięki logice find_preemption)
                        if not wezel_ofiary.can_fit(task.requirements):
                             print(f"    KRYTYCZNY BŁĄD: Wywłaszczenie nie zwolniło miejsca!")
                             # Wycofaj zmiany (skomplikowane) lub przerwij
                             # Na razie zakładamy, że logika jest poprawna
                        
                        wezel_ofiary.add_task(task)
                        app.U_i += task.requirements
                        app.running_tasks[task.id] = task
                        
                        # 5. Zaktualizuj poziom dominacji zwycięzcy i
                        app.update_dominant_share(self.R_total)
                        
                        # 6. Stwórz zdarzenie zakończenia dla nowego zadania
                        finish_event = TaskFinishEvent(
                            time=self.current_time + task.duration,
                            payload={'task_id': task.id, 'app_id': app.id, 'node_id': wezel_ofiary.id}
                        )
                        heapq.heappush(self.event_queue, finish_event)

                        alokacja_w_rundzie = True
                        break # Przerwij pętlę 'for' i zacznij nową rundę 'while'
                    
                    # *** KONIEC ZMIAN ***
                        
                    # else (wywłaszczenie nie jest opłacalne/możliwe):
                    print(f"    Brak opłacalnego wywłaszczenia dla Aplikacji {app.id}.")
                    pass # Kontynuuj pętlę 'for', sprawdzając kolejną aplikację
            
            # Koniec pętli 'for'
            if not alokacja_w_rundzie:
                print("    W tej rundzie nie dokonano żadnej alokacji. Kończę cykl szeregowania.")
                break # Wyjście z pętli 'while'
    

    def find_best_node(self, task_requirements: ResourceVector) -> Node | None:
        """
        Znajduje najlepszy węzeł dla zadania.
        Implementuje 'First Fit'. Można tu dodać sortowanie węzłów.
        """
        # TODO: Dodać sortowanie węzłów (np. BestFit, WorstFit)
        # Na razie prosty FirstFit
        
        # Iterujemy po posortowanej liście (tutaj wg ID, ale można zmienić)
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            if node.can_fit(task_requirements):
                return node
        return None

    def find_preemption_candidate(self, winner_app: Application, winner_task: Task) -> tuple[list[Task], Node] | tuple[None, None]:
        """
        Implementacja logiki 'Znajdź_Opłacalne_Wywłaszczenie' wg Ad 3.
        Szuka 'najtańszego' zbioru zadań-ofiar na jednym węźle.
        """
        s_W = winner_app.s_i
        D_W = winner_task.requirements
        
        # 1. Znajdź aplikację-ofiarę P (z najwyższym s_P)
        try:
            victim_app = max(
                (app for app in self.apps.values() if app.running_tasks), # Tylko te, które mają co wywłaszczyć
                key=lambda a: a.s_i
            )
        except ValueError:
            return None, None # Brak uruchomionych zadań w klastrze

        s_P = victim_app.s_i
        
        # Warunek wstępny: Ofiara musi mieć wyższy udział niż zwycięzca
        if not (s_P > s_W):
            return None, None
            
        best_candidate = None # Przechowuje (total_cost, victim_set, node)
        
        # 2. Iteruj przez wszystkie węzły k=1..K
        for node in self.nodes.values():
            
            # 3. Znajdź zadania ofiary P na tym węźle k
            victim_tasks_on_node = [
                task for task in node.running_tasks.values() 
                if task.app_id == victim_app.id
            ]
            
            if not victim_tasks_on_node:
                continue # Brak zadań ofiary na tym węźle, idź do następnego

            # 4. Oblicz koszt i posortuj
            tasks_with_cost = [
                (self._calculate_task_cost(task), task) 
                for task in victim_tasks_on_node
            ]
            # Sortuj wg kosztu (x[0])
            tasks_with_cost.sort(key=lambda x: x[0]) 
            
            # 5. Buduj "najtańszy zbiór na węźle k" (tau_Pk)
            tau_Pk = []
            freed_resources = np.zeros_like(self.R_total)
            current_total_cost = 0.0
            
            found_sufficient_set = False

            for (cost, task) in tasks_with_cost:
                tau_Pk.append(task)
                freed_resources += task.requirements
                current_total_cost += cost
                
                # 6. Sprawdź, czy T_W się zmieści
                # Warunek Wykonalności: (C_k - freed_resources) + D_W <= R_k
                if np.all((node.C_k - freed_resources) + D_W <= node.R_k):
                    found_sufficient_set = True
                    break # Mamy najtańszy *wystarczający* zbiór na tym węźle
            
            # 7. Jeśli znaleziono ("z górki")
            if found_sufficient_set:
                # 8. Oblicz hipotetyczne udziały
                U_W_prime = winner_app.U_i + D_W
                U_P_prime = victim_app.U_i - freed_resources
                
                s_W_prime = self._calculate_s_i(U_W_prime)
                s_P_prime = self._calculate_s_i(U_P_prime)
                
                # 9. Sprawdź Warunek Zachowania Hierarchii
                if not (s_P_prime > s_W_prime):
                    continue # Ten zestaw jest zły, idź do następnego węzła

                # 10. Sprawdź Warunek Zysku (Poprawa sprawiedliwości)
                # zysk = (max(s_P, s_W) - max(s_P', s_W')) = s_P - s_P'
                zysk = s_P - s_P_prime 
                
                if not (zysk > self.EPSILON):
                    continue # Zysk za mały, idź do następnego węzła

                # 11. Sprawdź Warunek Ekonomiczny
                koszt = current_total_cost
                
                if (zysk * self.ALPHA) > (koszt * self.BETA):
                    # Mamy WAŻNEGO kandydata z tego węzła!
                    candidate = (koszt, tau_Pk.copy(), node)
                    
                    # Sprawdź, czy jest lepszy (tańszy) niż dotychczasowy najlepszy
                    if best_candidate is None or candidate[0] < best_candidate[0]:
                        best_candidate = candidate
        
        # Koniec pętli po węzłach
        # 12. Wybierz najlepszego kandydata
        if best_candidate:
            # Zwróć najlepszego (najtańszego) kandydata ze wszystkich węzłów
            return best_candidate[1], best_candidate[2] # (victim_set, node)
        
        return None, None # Nie znaleziono żadnego kandydata
