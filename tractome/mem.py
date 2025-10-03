from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterState:
    """A class to represent the state of the application."""

    nb_clusters: int
    streamline_ids: np.ndarray
    max_clusters: int


class StateManager:
    """A class to manage the state of the application."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the StateManager if one does not exist."""
        if not cls._instance:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_size=10):
        """Initialize the state manager.

        Parameters
        ----------
        max_size : int, optional
            The maximum number of states to keep in history.
        """
        self._states = []
        self._max_size = max_size
        self._current_index = -1  # -1 means no state yet

    def add_state(self, state):
        """Add a new state.

        Parameters
        ----------
        state : ClusterState
            The state to add.
        """
        self._states.append(state)
        # Enforce max_size
        if len(self._states) > self._max_size:
            self._states = self._states[-self._max_size :]
        self._current_index = len(self._states) - 1

    def get_latest_state(self):
        """Get the current state (not always the last one).

        Returns
        -------
        ClusterState
            The current state.
        """
        if not self._states or self._current_index == -1:
            raise ValueError("No states available.")
        return self._states[self._current_index]

    def can_move_back(self):
        """Check if it's possible to move back to a previous state.

        Returns
        -------
        bool
            True if there are previous states to move back to, False otherwise.
        """
        return self._current_index > 0

    def move_back(self):
        """
        Move the pointer to the previous state (do not remove).

        Returns
        -------
        ClusterState
            The new current state after moving back.
        """
        if not self.can_move_back():
            raise ValueError("No previous state to move back to.")
        self._current_index -= 1
        return self.get_latest_state()

    def can_move_next(self):
        """Check if it's possible to move forward to a next state.

        Returns
        -------
        bool
            True if there is a next state to move forward to, False otherwise.
        """
        return self._current_index < len(self._states) - 1

    def move_next(self):
        """
        Move the pointer to the next state.

        Returns
        -------
        ClusterState
            The new current state after moving next.
        """
        if not self.can_move_next():
            raise ValueError("No next state to move forward to.")
        self._current_index += 1
        return self.get_latest_state()

    @property
    def history_size(self):
        """Get the number of states in the history.

        Returns
        -------
        int
            The number of states in the history.
        """
        return len(self._states)

    def get_all_states(self):
        """Get all states in history.

        Returns
        -------
        list
            A list of all states.
        """
        return list(self._states)

    def get_current_index(self):
        """Get the current index in the history.

        Returns
        -------
        int
            The current index.
        """
        return self._current_index
