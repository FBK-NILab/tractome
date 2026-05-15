from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np


@dataclass
class ClusterState:
    """A class to represent the state of the application."""

    nb_clusters: int
    streamline_ids: np.ndarray
    max_clusters: int
    tractogram_states: Optional[dict] = None
    filtered_streamline_ids: Optional[np.ndarray] = None


class StateManager:
    """A class to manage the state of the application."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the StateManager if one does not exist."""
        if not cls._instance:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_size=50):
        """Initialize the state manager.

        Parameters
        ----------
        max_size : int, optional
            The maximum number of states to keep in history.
        """
        self._states = []
        self._max_size = max_size
        self._current_index = -1  # -1 means no state yet
        self.t1_state = [0, 0, 0]
        self.t1_slice_visibility = [True, True, True]
        self.t1_slice_visibility_2d = [False, False, True]
        self.view_mode = "3D"
        self.mesh_photographic = True
        self.mesh_project = False
        self.mesh_projection_threshold = 2.5
        self.mesh_opacity = 100
        self.mesh_visible = True
        self.parcel_size = 100
        self.parcel_visible = True
        self.roi_opacity = 80
        self.roi_create_mode = None  # None | "sphere" | "rectangle"

    def has_states(self):
        """Check if there are any states in the history.

        Returns
        -------
        bool
            True if there are states in the history, False otherwise.
        """
        return len(self._states) > 0

    def add_state(self, state):
        """Add a new state.

        Parameters
        ----------
        state : ClusterState
            The state to add.
        """
        if self._current_index < len(self._states) - 1:
            self._states = self._states[: self._current_index + 1]
        self._states.append(state)
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

    def create_cluster_entry(self, cluster_id, streamline_ids, color, radius):
        """Create a new cluster entry.

        Parameters
        ----------
        cluster_id : int
            The ID of the cluster.
        streamline_ids : np.ndarray
            The IDs of the streamlines in the cluster.
        color : tuple
            The color of the cluster.
        radius : float
            The radius of the cluster.

        Returns
        -------
        dict
            The cluster entry.
        """
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is None:
            latest_state.tractogram_states = {}
        latest_state.tractogram_states[cluster_id] = {
            "streamline_ids": streamline_ids,
            "color": color,
            "selected": False,
            "expanded": False,
            "rep_actor": None,
            "visible": True,
            "lines_actor": None,
            "radius": radius,
        }
        return latest_state.tractogram_states[cluster_id]

    def toggle_cluster_selection(self, cluster_id):
        """Toggle the selection state of a cluster.

        Parameters
        ----------
        cluster_id : int
            The ID of the cluster.
        value : bool, optional
            The value to set the selection state to.
        """
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            if cluster_id in latest_state.tractogram_states:
                self._toggle_cluster_selection(
                    latest_state.tractogram_states[cluster_id]
                )
            else:
                logging.warning(f"Cluster {cluster_id} not found.")
        else:
            logging.warning("No states available.")

    def _toggle_cluster_selection(self, state_data, *, value=None):
        if value is not None:
            state_data["selected"] = value
        else:
            state_data["selected"] = not state_data["selected"]

    def select_all_clusters(self):
        """Select all clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                self._toggle_cluster_selection(
                    latest_state.tractogram_states[cluster_id], value=True
                )
        else:
            logging.warning("No states available.")

    def select_none_clusters(self):
        """Select none clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                self._toggle_cluster_selection(
                    latest_state.tractogram_states[cluster_id], value=False
                )
        else:
            logging.warning("No states available.")

    def swap_clusters(self):
        """Swap the selection state of all clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                self._toggle_cluster_selection(
                    latest_state.tractogram_states[cluster_id]
                )
        else:
            logging.warning("No states available.")

    def expand_clusters(self):
        """Expand selected clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                if latest_state.tractogram_states[cluster_id]["selected"]:
                    latest_state.tractogram_states[cluster_id]["expanded"] = True
        else:
            logging.warning("No states available.")

    def collapse_clusters(self):
        """Collapse selected clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                if latest_state.tractogram_states[cluster_id]["selected"]:
                    latest_state.tractogram_states[cluster_id]["expanded"] = False
        else:
            logging.warning("No states available.")

    def show_clusters(self):
        """Show selected clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                latest_state.tractogram_states[cluster_id]["visible"] = True
        else:
            logging.warning("No states available.")

    def hide_clusters(self):
        """Hide selected clusters."""
        latest_state = self.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id in latest_state.tractogram_states:
                if not latest_state.tractogram_states[cluster_id]["selected"]:
                    latest_state.tractogram_states[cluster_id]["visible"] = False
        else:
            logging.warning("No states available.")


state_manager = StateManager()
