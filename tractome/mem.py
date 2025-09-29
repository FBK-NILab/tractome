import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterState:
    """A class to represent the state of the application."""

    nb_clusters: int
    streamline_ids: np.ndarray


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
        self._states = collections.deque(maxlen=max_size)

    def add_state(self, state):
        """Add a new state.

        Parameters
        ----------
        state : ClusterState
            The state to add.
        """
        self._states.append(state)

    def get_latest_state(self):
        """Latest state.

        Returns
        -------
        ClusterState
            The latest state.
        """
        if not self._states:
            raise ValueError("No states available.")
        return self._states[-1]

    def can_move_back(self):
        """Check if it's possible to move back to a previous state.

        Returns
        -------
        bool
            True if there are previous states to move back to, False otherwise.
        """
        return self.history_size > 1

    def move_back(self):
        """
        Remove the current state and move to the previous state.

        Returns
        -------
        ClusterState
            The new latest state after moving back.
        """
        if not self._states:
            raise ValueError("No states to move back to.")
        self._states.pop()
        return self._states[-1]

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
