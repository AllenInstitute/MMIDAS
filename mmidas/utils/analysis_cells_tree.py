import json
from flatten_json import flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

class Node():
    '''
        Simple Node class. Each instance contains a list of children and parents.
    '''

    def __init__(self,name,C_list=[],P_list=[]):
        self.name=name
        self.C_name_list = C_list[P_list==name]
        self.P_name = P_list[C_list==name]
        return
    def __repr__(self):
        #Invoked when printing a list of Node objects
        return self.name
    def __str__(self):
        #Invoked when printing a single Node object
        return self.name
    def __eq__(self,other):
        
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False
    def children(self,C_list=[],P_list=[]):
        return [Node(n,C_list,P_list) for n in self.C_name_list]

def get_valid_classifications(current_node_list,C_list,P_list,valid_classes):
    '''
        Recursively generates all possible classifications that are valid,
        based on the hierarchical tree defined by `C_list` and `P_list` \n
        `current_node_list` is a list of Node objects. It is initialized as a list with only the root Node.
    '''
    
    current_node_list.sort(key=lambda x: x.name)
    valid_classes.append(sorted([node.name for node in current_node_list]))
    for node in current_node_list:
        current_node_list_copy = current_node_list.copy()
        children_node_list = node.children(C_list=C_list,P_list=P_list)
        if len(children_node_list)>0:
            current_node_list_copy.remove(node)
            current_node_list_copy.extend(children_node_list)
            if sorted([node.name for node in current_node_list_copy]) not in valid_classes:
                valid_classes = get_valid_classifications(current_node_list_copy,C_list=C_list,P_list=P_list,valid_classes=valid_classes)
    return valid_classes


class HTree():
    '''
        Class to work with hierarchical tree .csv generated for the transcriptomic data.
        `htree_file` is full path to a .csv. The original .csv was generated from dend.RData,
        processed with `dend_functions.R` and `dend_parents.R` (Ref. Rohan/Zizhen)
    '''
    def __init__(self, htree_df=None, htree_file=None):
        
        #Load and rename columns from filename
        if htree_file is not None:
            htree_df = pd.read_csv(htree_file)
            htree_df = htree_df[['x', 'y', 'leaf', 'label', 'parent', 'col']]

        if 'keep' in htree_df:
            htree_df.drop(not htree_df.keep, axis=0)

        htree_df = htree_df.rename(columns={'label': 'child','leaf': 'isleaf'})

        #Sanitize values
        htree_df['isleaf'].fillna(False,inplace=True)
        htree_df['y'].values[htree_df['isleaf'].values] = 0.0
        htree_df['col'].fillna('#000000',inplace=True)
        htree_df['parent'].fillna('root',inplace=True)
        htree_df['child'] = np.array([c.strip() for c in htree_df['child']])

        #Sorting for convenience
        htree_df = htree_df.sort_values(by=['y', 'x'], axis=0, ascending=[True, True]).copy(deep=True)
        htree_df = htree_df.reset_index(drop=True).copy(deep=True)
        
        #Set class attributes using dataframe columns
        for c in htree_df.columns:
            setattr(self, c, htree_df[c].values)
        return
    
    def obj2df(self):
        '''Convert HTree object to a pandas dataframe'''
        htree_df = pd.DataFrame({key:val for (key,val) in self.__dict__.items()})
        return htree_df
    
    def df2obj(self,htree_df):
        '''Convert a valid pandas dataframe to a HTree object'''
        for key in htree_df.columns:
            setattr(self, key, htree_df[key].values)
        return
    
    def get_marker(self, exclude=[]):
        '''Return a list of marker genes for each leaf node in the tree'''

        if len(exclude)==0:
            subclass_list = ['L2/3', 'L4', 'L5', 'L6', 'IT', 'PT', 'NP', 'CT', 'VISp', 'ALM', 'Sst', 'Vip', 'Lamp5', 'Pvalb', 'Sncg', 'Serpinf1']
        
        t_clusters = self.child[self.isleaf]
        marker_genes = []
        for ttype in t_clusters:
            indxs = [ch for ch in range(len(ttype)) if ttype[ch].find(' ')==0]
            indxs  = np.array(indxs + [len(ttype)])
            for i_idx in range(len(indxs)-1):
                tmp_gene = ttype[indxs[i_idx]+1:indxs[i_idx+1]]
                if tmp_gene not in subclass_list:
                    marker_genes.append(tmp_gene)

        return np.unique(marker_genes)


    def plot(self,figsize=(15,10),fontsize=10,skeletononly=True,
             skeletoncol='#BBBBBB',skeletonalpha=1.0, ls='-',txtleafonly=True,
             fig=None, ax=None, linewidth=1, save=False, path=[], n_node=0, 
             marker='s', marker_size=12, hline_nodes=[], n_c=[], cell_count=[0], 
             add_marker=False, margin_y=0.001):
        '''Plot the hierarchical tree. '''
        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()

        y_node = []
        col = self.col
        col[~self.isleaf] = '#000000'
        if n_node > 50:
            self.x = 2*self.x
            a = 2
        else:
            self.x = 4*self.x
            a = 3
        # Labels are shown only for children nodes
        if skeletononly==False:
            if txtleafonly==False:
                ss=1
                for i, label in enumerate(self.child):
                    plt.text(self.x[i], self.y[i], label,
                             color=self.col[i],
                             horizontalalignment='center',
                             verticalalignment='top',
                             rotation=90,
                             fontsize=fontsize - 4)
            else:
                for i in np.flatnonzero(self.isleaf):
                    label = self.child[i]
                    plt.text(self.x[i], self.y[i], label,
                            color='black',
                            horizontalalignment='center',
                            verticalalignment='top',
                            rotation=90,
                            fontsize=fontsize)

        for parent in np.unique(self.parent):
            #Get position of the parent node:
            p_ind = np.flatnonzero(self.child==parent).squeeze()
            if p_ind.size==0: #Enters here for any root node
                p_ind = np.flatnonzero(self.parent==parent).squeeze()
                xp = self.x[p_ind]
                yp = 1.1*np.max(self.y)
            else:
                xp = self.x[p_ind]
                yp = self.y[p_ind]

            for nd in hline_nodes:
                if parent == nd:
                    y_node.append(yp)

            all_c_inds = np.flatnonzero(np.isin(self.parent, parent))
            for c_ind in all_c_inds:
                xc = self.x[c_ind]
                yc = self.y[c_ind]
                plt.plot([xc, xc], [yc, yp], color=skeletoncol, alpha=skeletonalpha, ls=ls, linewidth=linewidth)
                plt.plot([xc, xp], [yp, yp], color=skeletoncol, alpha=skeletonalpha, ls=ls, linewidth=linewidth)
        if skeletononly==False:
            ax.axis('off')
            ax.set_xlim([np.min(self.x) - a, np.max(self.x) + a])
            ax.set_ylim([np.min(self.y), 1.1*np.max(self.y)])
            plt.tight_layout()
            
        if add_marker:
            print('add marker')
            ax.axis('off')
            ax.set_xlim([np.min(self.x) - 1, np.max(self.x) + 1])
            ax.set_ylim([np.min(self.y), 1.1 * np.max(self.y)])
            for i, s in enumerate(self.child):
                if i < n_node:
                    if self.y[i] > 0:
                        m_y = self.y[i] + margin_y*self.y[i]
                    else:
                        m_y = margin_y
                    print(i, s, self.col[i])
                    if isinstance(marker, list):
                        ax.plot(self.x[i], m_y, marker[i], color=self.col[i], ms=marker_size)
                    else:
                        ax.plot(self.x[i], m_y, marker, color=self.col[i], ms=marker_size)
        plt.tight_layout()
        if save:
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.savefig(path + '/subtree.png', dpi=600)

        return
    
    def plotnodes(self,nodelist,fig=None):
        '''Plot the nodes in the list nodelist with a red square marker.'''
        ind = np.isin(self.child,nodelist)
        plt.plot(self.x[ind], self.y[ind],'s',color='r')
        return

    def get_descendants(self,node:str,leafonly=False):
        '''Return a list consisting of all descendents for a given node. Given node is excluded.\n
        'node' is of type str \n
        `leafonly=True` returns only leaf node descendants'''

        descendants = []
        current_node = self.child[self.parent == node].tolist()
        descendants.extend(current_node)
        while current_node:
            parent = current_node.pop(0)
            next_node = self.child[self.parent == parent].tolist()
            current_node.extend(next_node)
            descendants.extend(next_node)
        if leafonly:
            descendants = list(set(descendants) & set(self.child[self.isleaf]))
        return descendants

    def get_all_descendants(self,leafonly=False):
        '''Return a dict consisting of node names as keys and, corresp. descendant list as values.\n
        `leafonly=True` returns only leaf node descendants'''
        descendant_dict = {}
        for key in np.unique(np.concatenate([self.child,self.parent])):
            descendant_dict[key]=self.get_descendants(node=key,leafonly=leafonly)
        return descendant_dict

    def get_ancestors(self, node, rootnode=None):
        '''Return a list consisting of all ancestors 
        (till `rootnode` if provided) for a given node.'''

        ancestors = []
        current_node = node
        while current_node:
            current_node = self.parent[self.child == current_node]
            ancestors.extend(current_node)
            if current_node==rootnode:
                current_node=[]
        return ancestors

    def get_mergeseq(self):
        '''Returns `ordered_merges` consisting of \n
        1. list of children to merge \n
        2. parent label to merge the children into \n
        3. number of remaining nodes in the tree'''

        # Log changes for every merge step
        ordered_merge_parents = np.setdiff1d(self.parent,self.child[self.isleaf])
        y = []
        for label in ordered_merge_parents:
            if np.isin(label,self.child):
                y.extend(self.y[self.child==label])
            else:
                y.extend([np.max(self.y)+0.1])
        
        #Lowest value is merged first
        ind = np.argsort(y)
        ordered_merge_parents = ordered_merge_parents[ind].tolist()
        ordered_merges = []
        while len(ordered_merge_parents) > 1:
            # Best merger based on sorted list
            parent = ordered_merge_parents.pop(0)
            children = self.child[self.parent == parent].tolist()
            ordered_merges.append([children, parent])
        return ordered_merges

    def get_subtree(self, node):
        '''Return a subtree from the current tree'''
        subtree_node_list = self.get_descendants(node=node)+[node]
        if len(subtree_node_list)>1:
            subtree_df = self.obj2df()
            subtree_df = subtree_df[subtree_df['child'].isin(subtree_node_list)]
        else:
            print('Node not found in current tree')
        return HTree(htree_df=subtree_df)

    def update_layout(self):
        '''Update `x` positions of tree based on newly assigned leaf nodes.
        '''
        #Update x position for leaf nodes to evenly distribute them.
        all_child = self.child[self.isleaf]
        all_child_x = self.x[self.isleaf]
        sortind = np.argsort(all_child_x)
        new_x = 0
        for (this_child,this_x) in zip(all_child[sortind],all_child_x[sortind]):
            self.x[self.child==this_child]=new_x
            new_x = new_x+1
            
            
        parents = self.child[~self.isleaf].tolist() 
        for node in parents:
            descendant_leaf_nodes = self.get_descendants(node=node,leafonly=True)
            parent_ind = np.isin(self.child,[node])
            descendant_leaf_ind = np.isin(self.child,descendant_leaf_nodes)
            self.x[parent_ind] = np.mean(self.x[descendant_leaf_ind])
        return 


def do_merges(labels, list_changes=[], n_merges=0, verbose=False):
    """Perform n_merges on an array of labels using the list of changes at each merge. 
    If labels are leaf node labels, then the do_merges() gives successive horizontal cuts of the hierarchical tree.
    
    Arguments:
        labels -- label array to update
    
    Keyword Arguments:
        list_changes  -- output of Htree.get_mergeseq()
        n_merges -- int, can be at most len(list_changes)
    
    Returns:
        labels -- array of updated labels. Same size as input, non-unique entries are allowed.
    """
    assert isinstance(labels,np.ndarray), 'labels must be a numpy array'
    for i in range(n_merges):
        if i < len(list_changes):
            c_nodes_list = list_changes[i][0]
            p_node = list_changes[i][1]
            for c_node in c_nodes_list:
                n_samples = np.sum([labels == c_node])
                labels[labels == c_node] = p_node
                if verbose:
                    print(n_samples,' in ',c_node, ' --> ' ,p_node)
        else:
            print('Exiting after performing max allowed merges =',
                  len(list_changes))
            break
    return labels


def simplify_tree(pruned_subtree,skip_nodes=None):
    '''pruned subtree has nodes that have a single child node. In the returned simplified tree,
    the parent is directly connected to the child, and such intermediate nodes are removed.'''
    
    simple_tree = deepcopy(pruned_subtree)
    if skip_nodes is None:
        X = pd.Series(pruned_subtree.parent).value_counts().to_frame()
        skip_nodes = X.iloc[X[0].values==1].index.values.tolist()

    for node in skip_nodes:
        node_parent = np.unique(simple_tree.parent[simple_tree.child == node])
        node_child = np.unique(simple_tree.child[simple_tree.parent == node])

        #Ignore root node special case:
        if node_parent.size != 0:
            #print(simple_tree.obj2df().to_string())
            print('Remove {} and link {} to {}'.format(node, node_parent, node_child))
            simple_tree.parent[simple_tree.parent == node]=node_parent

            #Remove rows containing this particular node as parent or child
            simple_tree_df=simple_tree.obj2df()
            simple_tree_df.drop(simple_tree_df[(simple_tree_df.child == node) | (simple_tree_df.parent == node)].index, inplace=True)

            #Reinitialize tree from the dataframe
            simple_tree=HTree(htree_df=simple_tree_df)
        
    return simple_tree,skip_nodes


def dend_json_to_df(json_file):
    """
    Convert a json file to a pandas dataframe
    """
    with open(json_file, 'r') as f:
        s = f.read()
        s = s.replace('\t', '')
        s = s.replace('\n', '')
        s = s.replace(',}', '}')
        s = s.replace(',]', ']')
        dend = json.loads(s)

    flatten_dend = flatten(dend)
    label, members, height, color, index, midpoint = [], [], [], [], [], []
    org_label, parent, leaf, cex, xpos = [], [], [], [], []
    dend_keys = list(flatten_dend.keys())

    for i, _ in enumerate(dend_keys):
        if i < 1:
            index = i
        if index < len(dend_keys):
            entry = dend_keys[index]
            if 'leaf_attribute' in entry:
                ind_0 = [i for i, x in enumerate(entry) if x == '0']
                tag = entry[:ind_0[-1] + 2]
                key = tag + '_row'
                label.append(flatten_dend[key])
                key = tag + 'members'
                members.append(flatten_dend[key])
                key = tag + 'height'
                height.append(flatten_dend[key])
                key = tag + 'nodePar.col'
                color.append(flatten_dend[key])
                midpoint.append('')
                key = tag + 'nodePar.cex'
                cex.append(flatten_dend[key])
                leaf.append(True)
                number_ind = label[-1].find('_')
                xpos.append(np.float16(label[-1][:number_ind]))
                ind_child = [i for i, _ in enumerate(entry[:-8])
                             if entry[i:i + 8] == 'children']
                key_parent = entry[
                             :ind_child[-2] + 10] + '_node_attributes_0__row'
                if key_parent in flatten_dend:
                    parent.append(flatten_dend[key_parent])
                else:
                    parent.append('')
                index += 21
            if 'node_attribute' in entry:
                ind_0 = [i for i, x in enumerate(entry) if x == '0']
                tag = entry[:ind_0[-1] + 2]
                key = tag + '_row'
                label.append(flatten_dend[key])
                key = tag + 'members'
                members.append(flatten_dend[key])
                key = tag + 'height'
                height.append(flatten_dend[key])
                color.append('')
                key = tag + 'midpoint'
                midpoint.append(flatten_dend[key])
                cex.append('')
                leaf.append(False)
                xpos.append(0.)
                ind_child = [i for i, _ in enumerate(entry[:-8])
                             if entry[i:i + 8] == 'children']
                if len(ind_child) > 0:
                    if len(ind_child) > 1:
                        key_parent = entry[:ind_child[-2] + 10] + \
                                     '_node_attributes_0__row'
                    else:
                        key_parent = 'node_attributes_0__row'
                    parent.append(flatten_dend[key_parent])
                else:
                    parent.append('')
                index += 15

    # find x position for all non leaf nodes
    x = np.array(xpos)
    for i, l in enumerate(label):
        if not leaf[i]:
            parent_ind = np.where(np.array(parent) == l)[0]
            x[i] = np.mean(x[parent_ind])

    # build a dataframe from the flatten dendrogram
    dend_df = pd.DataFrame({'x': list(x),
                            'y': height,
                            'cex': cex,
                            'col': color,
                            'members': members,
                            'midpoint': midpoint,
                            'height': height,
                            'leaf': leaf,
                            'label': label,
                            'parent': parent})

    # reverse the order nodes in the dataframe
    dend_df = dend_df.iloc[::-1].reset_index(drop=True)

    # replace empty values with nan
    dend_df = dend_df.replace(r'', np.NaN)

    # start the dataframe index from 1
    dend_df.index += 1

    return dend_df
