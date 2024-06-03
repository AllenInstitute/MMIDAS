import scipy.stats as stats
import numpy as np
from mmidas.utils.analysis_cells_tree import HTree, do_merges

resolution = 600

def corr_analysis(state, cell):
        """ Compute correlation between cell types and gene expression
        input args
            state (np.array): cell type states
            cell (np.array): gene expression matrix

        return
            all_corr (list): list of correlation coefficients
            all_geneID (list): list of gene IDs
        """

        n_gene = cell.shape[-1]
        all_corr, all_geneID = [], []
        for s in range(state.shape[-1]):
            # compute cross correlation using Pearson correction coefficient
            cor_coef, p_val = np.zeros(n_gene), np.zeros(n_gene)
            for g in range(n_gene):
                if np.max(cell[:, g]) > 0:
                    zind = np.where(cell[:, g] > 0)
                    if len(zind[0]) > 4:
                        cor_coef[g], p_val[g] = \
                            stats.pearsonr(state[zind[0], s],
                                           cell[zind[0], g])
                    else:
                        cor_coef[g], p_val[g] = 0, 0
                else:
                    cor_coef[g], p_val[g] = 0, 0

            g_id = np.argsort(np.abs(cor_coef))
            all_corr.append(np.sort(np.abs(cor_coef)))
            all_geneID.append(g_id)

        return all_corr, all_geneID


def get_merged_types(htree_file, cells_labels, num_classes=0, ref_leaf=[], node='n4'):
    """ Merge cell types based on the hierarchical tree
    input args
        htree_file (str): path to the hierarchical tree
        cells_labels (np.array): cell type labels
        num_classes (int, optional): number of original classes
        ref_leaf (list, optional): reference leaf nodes
        node (str, optional): reference node for cutting the tree
        
        return
        merged_cells_labels (np.array): merged cell type labels
        mod_subtree (HTree): modified subtree
    """
    # get the tree
    htree = HTree(htree_file=htree_file)
    htree.parent = np.array([c.strip() for c in htree.parent])
    htree.child = np.array([c.strip() for c in htree.child])

    # get a subtree according the to the given node:
    subtree = htree.get_subtree(node=node)
    if len(ref_leaf) > 0:
        ref_leaf = np.array(ref_leaf)
        in_idx = np.array([(ref_leaf == c).any() for c in subtree.child[subtree.isleaf]])
        subtree.child = np.concatenate((subtree.child[subtree.isleaf][in_idx], subtree.child[~subtree.isleaf]))
        subtree.parent = np.concatenate((subtree.parent[subtree.isleaf][in_idx], subtree.parent[~subtree.isleaf]))
        subtree.col = np.concatenate((subtree.col[subtree.isleaf][in_idx], subtree.col[~subtree.isleaf]))
        subtree.x = np.concatenate((subtree.x[subtree.isleaf][in_idx], subtree.x[~subtree.isleaf]))
        subtree.y = np.concatenate((subtree.y[subtree.isleaf][in_idx], subtree.y[~subtree.isleaf]))
        subtree.isleaf = np.concatenate((subtree.isleaf[subtree.isleaf][in_idx], subtree.isleaf[~subtree.isleaf]))

    # get a list of merges to carry out, sorted by the depth
    L = subtree.get_mergeseq()

    if num_classes == 0:
        go = len(L)
    else:
        go = num_classes

    merged_cells_labels = do_merges(labels=cells_labels,
                                    list_changes=L,
                                    n_merges=(go-1), verbose=False)

    unique_merged_cells_labels = do_merges(labels=subtree.child[subtree.isleaf],
                                    list_changes=L,
                                    n_merges=(go-1), verbose=False)

    # Obtain all relevant ancestor nodes:
    kept_leaf_nodes = list(set(unique_merged_cells_labels.tolist()))
    kept_tree_nodes = []
    for node in kept_leaf_nodes:
        kept_tree_nodes.extend(subtree.get_ancestors(node))
        kept_tree_nodes.extend([node])

    kept_subtree_df = subtree.obj2df()
    kept_subtree_df = kept_subtree_df[kept_subtree_df['child'].isin(kept_tree_nodes)]

    #Plot updated tree:
    kept_subtree = HTree(htree_df=kept_subtree_df)

    kept_subtree_df['isleaf'].loc[kept_subtree_df['child'].isin(kept_leaf_nodes)] = True
    kept_subtree_df['y'].loc[kept_subtree_df['child'].isin(kept_leaf_nodes)] = 0.0
    mod_subtree = HTree(htree_df=kept_subtree_df)

    mod_subtree.update_layout()

    return merged_cells_labels, mod_subtree, subtree
