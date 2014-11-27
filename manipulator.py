# -*- coding: utf-8 -*-

"""This module implements the logic of the operations for selecting,
unselecting, expanding, hiding, showing etc. of clusters of
streamlines. It basically provides set operations for streamline IDs.

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

import numpy as np
import random
import copy
import code



def clustering_random(data, clusters_number, streamline_ids, seed=0):
    """Create fake clustering just playing randomly with streamline
    ids. For testing purpose.
    """
    if clusters_number > len(streamline_ids):
        clusters_number = len(streamline_ids)
        print "clustering_function WARNING: clusters_number > len(streamline_ids)"
        
    random.seed(seed)
    representatives = set(random.sample(streamline_ids, clusters_number))
    what_remains = set(streamline_ids).difference(representatives)
    random_splits_ids = np.random.randint(low=0, high=len(representatives), size=len(what_remains))
    clusters = {}
    what_remains = np.array(list(what_remains))
    for i, representative in enumerate(representatives):
        clusters[representative] = set(what_remains[random_splits_ids==i].tolist()).union([representative])

    return clusters


class Manipulator(object):
    """This class provides the functions to manipulate streamline IDs
    that can suit an interactive session. It provides history
    capabilities and other amenities.
    """
    
    def __init__(self, initial_clusters, clustering_function):
        """Initialize the object.
        """
        self.initial_clusters = copy.deepcopy(initial_clusters)
        self.history = []
        self.streamline_ids = 0
        self.numstream_handler = EventHook()
        self.numrep_handler = EventHook()
        self.remselect_handler = EventHook()
        self.clusters_reset(initial_clusters)
        self.simple_history_start()
        self.clustering_function = clustering_function
        
        

    def clusters_reset(self, clusters):
        """Standard operations to do whenever self.clusters is
        modified.
        """
        self.clusters = clusters
        self.streamline_ids = reduce(set.union, clusters.values())
        self.representative_ids = set(self.clusters.keys())
        self.selected = set()
        self.expanded = set()
        self.show_representatives = True
        self.history.append('clusters_reset('+str(clusters)+')')
        self.numstream_handler.fire(len(self.streamline_ids))
        self.numrep_handler.fire(len(self.representative_ids))
    
    def select(self, representative_id):
        """Select one representative.
        """
        assert(representative_id in self.representative_ids)
        assert(representative_id not in self.selected)
        self.selected.add(representative_id)
        self.history.append('select('+str(representative_id)+')')
        self.select_action(representative_id)


    def select_action(self, representative_id):
        """This is the actual action to perform in the application.
        """
        pass


    def unselect(self, representative_id):
        """Select one representative.
        """
        assert(representative_id in self.representative_ids)
        assert(representative_id in self.selected)
        self.selected.remove(representative_id)
        self.history.append('unselect('+str(representative_id)+')')
        self.unselect_action(representative_id)
        

    def unselect_action(self, representative_id):
        """This is the actual action to perform in the application.
        """
        pass


    def select_toggle(self, representative_id):
        """Toggle for dispatching select or unselect.
        """
        if representative_id not in self.selected:
            self.select(representative_id)
        else:
            self.unselect(representative_id)
        

    def select_all(self):
        """Select all streamlines.
        """
        # it is safer to make a copy of the ids to avoid subtle
        # dependecies when manipulating self.representative_ids in
        # later steps.
        self.selected = set(copy.deepcopy(self.representative_ids))
        self.history.append('select_all()')
        self.select_all_action()


    def select_all_action(self):
        pass
        

    def unselect_all(self):
        self.selected = set()
        self.history.append('unselect_all()')
        self.unselect_all_action()
        

    def unselect_all_action(self):
        pass
        

    def select_all_toggle(self):
        if self.selected == self.representative_ids:
            self.unselect_all()
        else:
            self.select_all()


    def remove_selected(self):
        """Remove all clusters whose representative is selected.
        """
        clusters = {}
        for representative in set(self.representative_ids).difference(self.selected):
            clusters[representative] = self.clusters[representative]
        self.clusters_reset(clusters)
        self.history.append('remove_selected()')
        self.remselect_handler.fire(True)
        self.remove_selected_action()
        


    def remove_selected_action(self):
        pass
        

    def remove_unselected(self):
        """Remove all clusters whose representative is not selected.
        """
        clusters = {}
        for representative in self.selected:
            clusters[representative] = self.clusters[representative]
        self.clusters_reset(clusters)
        self.history.append('remove_unselected()')
        self.simple_history_store()
        self.remselect_handler.fire(True)
        self.remove_unselected_action()


    def remove_unselected_action(self):
        pass
        

    def recluster(self, clustering_parameter, data=None):
        """Operate self.clustering with clustering_parameter on the
        current objects.
        """
        streamline_ids_new = reduce(set.union, self.clusters.values())
        clusters_new = self.clustering_function(data, clustering_parameter, streamline_ids_new)
        # sanity check:
        assert(streamline_ids_new == reduce(set.union, clusters_new.values()))
        self.history.append('recluster('+str(clustering_parameter)+')')
        self.clusters_reset(clusters_new)
        self.simple_history_store()
        self.recluster_action()


    def recluster_action(self):
        pass


    def invert(self):
        """Invert the selection of representatives.
        """
        self.selected = self.representative_ids.difference(self.selected)
        self.history.append('invert()')
        self.invert_action()


    def invert_action(self):
        pass


    def show_representatives(self):
        """Show representatives.
        """
        self.show_representatives = True
        self.history.append('show_representatives()')
        

    def hide_representatives(self):
        """Do not show representatives.
        """
        self.show_representatives = False
        self.history.append('hide_representatives()')


    def expand_collapse_selected(self):
        """Toggle expand/collapse status of selected representatives.
        """
        self.expand = not self.expand
        self.history.append('expand_collapse_selected()')
        self.expand_collapse_selected_action()


    def expand_collapse_selected_action(self):
        pass
        

    def replay_history(self, until=None):
        """Create a Manipulator object by replaying the history of
        self starting from self.initial_clusters.
        """
        m = Manipulator(self.initial_clusters, self.clustering_function)
        # skip the first action in the history because already done
        # during __init__():
        c = ['m.'+h for h in self.history[1:until]]
        c = '; '.join(c)
        c = code.compile_command(c)
        exec(c)
        return m

    def simple_history_start(self):
        """Start simple history.
        """
        self.simple_history = [copy.deepcopy(self.clusters)]
        self.simple_history_pointer = 0
        print "Simple history started."


    def simple_history_store(self):
        """Store the current clusters into the history. If there are
        future steps stored in the history, then throw them away
        first.
        """
        if self.simple_history_pointer < len(self.simple_history)-1:
            print "Removing future steps."
            self.simple_history = self.simple_history[:self.simple_history_pointer+1]

        print "Appending to the existing simple history."
        self.simple_history.append(copy.deepcopy(self.clusters))
        self.simple_history_pointer += 1

    
    def simple_history_back_one_step(self):
        """Go back one step into the past.
        """
        if self.simple_history_pointer == 0:
            print "We are at the initial step and cannot go backward."
        else:
            self.simple_history_pointer -= 1
            self.clusters_reset(self.simple_history[self.simple_history_pointer])
            self.recluster_action()
        

    def simple_history_forward_one_step(self):
        """Go one step into the future.
        """
        if self.simple_history_pointer == len(self.simple_history)-1:
            print "We are at the last step and cannot go forward."
        else:
            self.simple_history_pointer += 1
            self.clusters_reset(self.simple_history[self.simple_history_pointer])
            self.recluster_action()


    def get_state(self):
        """Create a dictionary from which it is possible to
        reconstruct the current Manipulator.
        """
        state = {}
        state['clusters'] = copy.deepcopy(self.clusters)
        state['selected'] = copy.deepcopy(self.selected)
        state['expand'] = self.expand
        state['simple_history'] = copy.deepcopy(self.simple_history)
        state['simple_history_pointer'] = copy.deepcopy(self.simple_history_pointer)
        return state


    def set_state(self, state):
        """Set the current object with a given state. Useful to
        serialize the Manipulator to file etc.
        """
        self.clusters_reset(state['clusters'])
        self.selected = state['selected']
        self.expand = state['expand']
        self.simple_history = state['simple_history']
        self.simple_history_pointer = state['simple_history_pointer']
        self.recluster_action()
        

    def __str__(self):
        string = "Clusters: " + str(self.clusters)
        string += "\n"
        string += "Selected: " + str(self.selected)
        string += "\n"
        string += "Show Representatives: " + str(self.show_representatives)
        string += "\n"
        string += "Initial Clusters: " + str(self.initial_clusters)
        string += "\n"
        string += "History: " + str(self.history)
        return string


class EventHook(object):
    """
    Class that allows simulating the events-delegate functionality.
    Check later if there is a simplest way to do it in a more
    Pythonian way.
    """
    def __init__(self):
        self.__handlers = []
 
    def __iadd__(self, handler):
        self.__handlers.append(handler)
        return self
 
    def __isub__(self, handler):
        self.__handlers.remove(handler)
        return self
         
    def fire(self, *args, **keywargs):
        for handler in self.__handlers:
            handler(*args, **keywargs)
         
    def clearObjectHandlers(self, inObject):
        for theHandler in self.__handlers:
            if theHandler.im_self == inObject:
                self -= theHandler
    

if __name__ == '__main__':

    seed = 0
    random.seed(0)
    np.random.seed(0)

    clustering = clustering_random
    k = 3

    streamline_ids = np.arange(10, dtype=np.int)

    initial_clusters = clustering(None, k, streamline_ids)
    
    m = Manipulator(initial_clusters, clustering)

    print "Initial setting:"
    print "clusters:", m.clusters
    print "selected:", m.selected
    print "Select 3 and 6:"
    m.select_toggle(3)
    m.select_toggle(6)
    print "clusters:", m.clusters
    print "selected:", m.selected
    print "Invert selection:"
    m.invert()
    print "clusters:", m.clusters
    print "selected:", m.selected    
    print "Remove unselected:"
    m.remove_unselected()
    print "clusters:", m.clusters
    print "selected:", m.selected
    print "Re-cluster into 2 clusters:"
    m.recluster(2)
    print "clusters:", m.clusters
    print "selected:", m.selected
    print "Select cluster 8:"
    m.select_toggle(8)
    print "clusters:", m.clusters
    print "selected:", m.selected
    print "Re-cluster into 2 clusters:"
    print "Remove unselected:"
    m.remove_unselected()
    print "clusters:", m.clusters
    print "selected:", m.selected
    m.recluster(2)
    print "clusters:", m.clusters
    print "selected:", m.selected


