from threading import Lock

class Blackboard:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Blackboard, cls).__new__(cls)
                    cls._instance._shared_space = {}
                    cls._instance._current_problem = None
        return cls._instance

    def write(self, agent_name, contribution):
        with self._lock:
            if self._current_problem:
                if self._current_problem not in self._shared_space:
                    self._shared_space[self._current_problem] = []
                self._shared_space[self._current_problem].append({
                    "agent": agent_name,
                    "contribution": contribution
                })

    def read(self, problem_id):
        with self._lock:
            return self._shared_space.get(problem_id, [])

    def set_current_problem(self, problem_id):
        with self._lock:
            self._current_problem = problem_id

    def clear_problem(self, problem_id):
        with self._lock:
            if problem_id in self._shared_space:
                del self._shared_space[problem_id]
