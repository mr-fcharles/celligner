from celligner.params import *
#from celligner import limma
from celligner.fcomputer import FComputer

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import umap.umap_ as umap

import scanpy as sc
from anndata import AnnData

import os
import pickle
import gc

import pandas as pd
import numpy as np

import torch

import mnnpy


class Celligner(object):
    def __init__(
        self,
        topKGenes=TOP_K_GENES,
        pca_ncomp=PCA_NCOMP,
        snn_n_neighbors=SNN_N_NEIGHBORS,
        cpca_ncomp=CPCA_NCOMP,
        louvain_kwargs=LOUVAIN_PARAMS,
        mnn_kwargs=MNN_PARAMS,
        umap_kwargs=UMAP_PARAMS,
        mnn_method="mnn_marioni",
        low_mem=False,
        device="cpu",
        alpha=1.0,
        **kwargs
    ):
        """
        Initialize Celligner object

        Args:
            topKGenes (int, optional): see params.py. Defaults to 1000.
            pca_ncomp (int, optional): see params.py. Defaults to 70.
            cpca_ncomp (int, optional): see params.py. Defaults to 4.
            louvain_kwargs (dict, optional): see params.py
            mnn_kwargs (dict, optional): see params.py 
            umap_kwargs (dict, optional): see params.py
            mnn_method (str, optional): Only default "mnn_marioni" supported right now.
            low_mem (bool, optional): adviced if you have less than 32Gb of RAM. Defaults to False.
        """
        
        self.topKGenes = topKGenes
        self.pca_ncomp = pca_ncomp
        self.snn_n_neighbors = snn_n_neighbors
        self.cpca_ncomp = cpca_ncomp
        self.louvain_kwargs = louvain_kwargs
        self.mnn_kwargs = mnn_kwargs
        self.umap_kwargs = umap_kwargs
        self.mnn_method = mnn_method
        self.low_mem = low_mem

        self.ref_input = None
        self.ref_clusters = None
        self.ref_de_genes = None
        
        self.target_input = None
        self.target_clusters = None
        self.target_de_genes = None

        self.de_genes = None
        self.cpca_loadings = None
        self.cpca_explained_var = None
        self.combined_output = None
        
        self.umap_reduced = None
        self.output_clusters = None
        self.tumor_CL_dist = None

        self.device = device
        self.alpha = alpha


    def __checkExpression(self, expression, is_reference,compute_cPCs=True,target=False):
        """
        Checks gene overlap with reference, checks for NaNs, then does mean-centering.

        Args:
            expression (pd.Dataframe): expression data as samples (rows) x genes (columns)
            is_reference (bool): whether the expression is a reference or target

        Raises:
            ValueError: if some common genes are missing from the expression dataset
            ValueError: if the expression matrix contains nan values

        Returns:
            (pd.Dataframe): the expression matrix
        """
        # Check gene overlap
        if expression.loc[:, expression.columns.isin(self.common_genes)].shape[1] < len(self.common_genes):
            if not is_reference:
                raise ValueError("Some genes from reference dataset not found in target dataset")
            else:
                raise ValueError("Some genes from previously fit target dataset not found in new reference dataset")
        
        expression = expression.loc[:, self.common_genes].astype(float)
        
        # Raise issue if there are any NaNs in the expression dataframe
        if expression.isnull().values.any():
            raise ValueError("Expression dataframe contains NaNs")

        # Mean center the expression dataframe
        if compute_cPCs and target:
            self.means_target = expression.mean(0)
            expression = expression.sub(self.means_target, 1)
        elif not compute_cPCs and target:
            assert self.means_target is not None, "No means found, run transform with compute_cPCs=True at least once"
            expression = expression.sub(self.means_target, 1)
        elif compute_cPCs and not target:
            self.means_ref = expression.mean(0)
            expression = expression.sub(self.means_ref, 1)
        
        return expression


    def __cluster(self, expression):
        """
        Cluster expression in (n=70)-dimensional PCA space using a shared nearest neighbor based method

        Args:
            expression (pd.Dataframe): expression data as samples (rows) x genes (columns)

        Returns:
            (list): cluster label for each sample
        """
        # Create anndata object
        adata = AnnData(expression, dtype='float64')

        # Find PCs
        print("Doing PCA..")
        sc.tl.pca(adata, n_comps=self.pca_ncomp, zero_center=True, svd_solver='arpack')

        # Find shared nearest neighbors (SNN) in PC space
        # Might produce different results from the R version as ScanPy and Seurat differ in their implementation.
        print("Computing neighbors..")
        sc.pp.neighbors(adata, knn=True, use_rep='X_pca', n_neighbors=20, n_pcs=self.pca_ncomp)
        
        print("Clustering..")
        sc.tl.louvain(adata, use_weights=True, **self.louvain_kwargs)
        fit_clusters = adata.obs["louvain"].values.astype(int)
        
        del adata
        gc.collect()

        return fit_clusters


    def __runDiffExprOnClusters(self, expression, clusters,n_jobs=1):
        """
        Runs limma (R) on the clustered data.

        Args:
            expression (pd.Dataframe): expression data
            clusters (list): the cluster labels (per sample)

        Returns:
            (pd.Dataframe): limmapy results
        """

        n_clusts = len(set(clusters))
        print("Running differential expression on " + str(n_clusts) + " clusters..")
        clusts = set(clusters) - set([-1])
        
        # make a design matrix
        design_matrix = pd.DataFrame(
            index=expression.index,
            data=np.array([clusters == i for i in clusts]).T,
            columns=["C" + str(i) + "C" for i in clusts],
        )
        design_matrix.index = design_matrix.index.astype(str).str.replace("-", ".")
        design_matrix = design_matrix[design_matrix.sum(1) > 0]
        
        # creating the matrix
        data = expression.T
        data = data[data.columns[clusters != -1].tolist()]

        #running differential expression
        de = FComputer(n_jobs=8)
        de.fit_sklearn(data.values, design_matrix.values.astype(float),gene_names=data.index)
        de.eBayes()
        de.Fstats()

        #formatting output
        res = pd.DataFrame()
        res['F'] = de.F
        res['index'] = data.index
        res.set_index('index', inplace=True)
        
        return res.sort_values(by="F", ascending=False)
    

    def __runCPCA(self, centered_ref_input, centered_target_input):
        """
        Perform contrastive PCA on the centered reference and target expression datasets

        Args:
            centered_ref_input (pd.DataFrame): reference expression matrix where the cluster mean has been subtracted
            centered_target_input (pd.DataFrame): target expression matrix where the cluster mean has been subtracted

        Returns:
            (ndarray, ncomponents x ngenes): principal axes in feature space
            (ndarray, ncomponents,): variance explained by each component

        """
        target_cov = centered_target_input.cov().values
        ref_cov = centered_ref_input.cov().values
        
        # Step 1: Convert correlation matrix to torch tensor and compute eigenvalues and eigenvectors
        diff_cov = torch.tensor(target_cov - self.alpha * ref_cov,dtype=torch.float32,device=self.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(diff_cov)
        
        # Step 2: Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
        # Step 3: Select the top n_components if specified
        if self.cpca_ncomp is not None:
            sorted_eigenvalues = sorted_eigenvalues[:self.cpca_ncomp]
            sorted_eigenvectors = sorted_eigenvectors[:, :self.cpca_ncomp]
        
        # Step 4: Compute the explained variance
        explained_variance = sorted_eigenvalues / torch.sum(sorted_eigenvalues)

        # Step 5 project the target and reference data onto the cPCs
        target_pcs = pd.DataFrame(centered_target_input.values @ sorted_eigenvectors.cpu().numpy(), index=centered_target_input.index, columns=["cPC" + str(i) for i in range(sorted_eigenvectors.shape[1])])
        ref_pcs = pd.DataFrame(centered_ref_input.values @ sorted_eigenvectors.cpu().numpy(), index=centered_ref_input.index, columns=["cPC" + str(i) for i in range(sorted_eigenvectors.shape[1])])
        
        return sorted_eigenvectors.cpu().numpy(), explained_variance.cpu().numpy(), target_pcs, ref_pcs


    def fit(self, ref_expr,n_jobs=1):
        """
        Fit the model to the reference expression dataset - cluster + find differentially expressed genes.

        Args:
            ref_expr (pd.Dataframe): reference expression matrix of samples (rows) by genes (columns), 
                where genes are ensembl gene IDs. Data should be log2(X+1) TPM data. 
                In the standard Celligner pipeline this the cell line data.

        Raises:
                ValueError: if only 1 cluster is found in the PCs of the expression
        """
        
        self.common_genes = list(ref_expr.columns)
        self.ref_input = self.__checkExpression(ref_expr, is_reference=True,compute_cPCs=True,target=False)
        
        # Cluster and find differential expression for reference data
        self.ref_clusters = self.__cluster(self.ref_input)
        if len(set(self.ref_clusters)) < 2:
            raise ValueError("Only one cluster found in reference data, no differential expression possible")
        self.ref_de_genes = self.__runDiffExprOnClusters(self.ref_input, self.ref_clusters,n_jobs=n_jobs)

        return self
    
    def __regress_out_cpcs(self, input_data, pcs, loadings, regression_method):
        """
        Regress out the cPCs from the input data.

        Args:
            input_data (pd.DataFrame): The input data to transform.
            pcs (pd.DataFrame): The principal components.
            loadings (np.ndarray): The cPCA loadings.
            regression_method (str): The regression method to use ('by_sample', 'by_cPC', or 'back_project').

        Returns:
            pd.DataFrame: The transformed data with cPCs regressed out.
        """
        if regression_method == "by_sample":
            input_data = input_data.sort_index()
            pcs = pcs.sort_index()
            return input_data - LinearRegression(fit_intercept=False).fit(pcs, input_data).predict(pcs).T

        elif regression_method == "by_cPC":
            return (input_data.T - LinearRegression(fit_intercept=False)
                    .fit(loadings, input_data.T)
                    .predict(loadings)).T

        elif regression_method == "back_project":
            return pd.DataFrame(
                input_data.values - (input_data.values @ loadings @ loadings.T),
                index=input_data.index,
                columns=input_data.columns
            )

        else:
            raise ValueError("Invalid regression method")


    def transform(self, target_expr=None, compute_cPCs=True,transform_source="target",regression_method="back_project"):
        """
        Aligns samples in the target dataset to samples in the reference dataset.

        Args:
            target_expr (pd.DataFrame, optional): The target expression matrix, where rows represent samples and columns represent genes identified by their Ensembl gene IDs. The data should be in log2(X+1) TPM format. In the standard Celligner pipeline, this corresponds to the tumor data (TCGA). Set to None if re-running transform with new reference data.
            compute_cPCs (bool, optional): A flag indicating whether to compute contrastive principal components (cPCs) from the fitted reference and target expression. Defaults to True.

        Raises:
            ValueError: Raised if compute_cPCs is True but there is no reference input (fit has not been run).
            ValueError: Raised if compute_cPCs is False but there are no previously computed cPCs available (transform has not been previously run).
            ValueError: Raised if no target expression is provided and there is no previously provided target data.
            ValueError: Raised if no target expression is provided and compute_cPCs is true; there is no use case for this.
            ValueError: Raised if there are not enough clusters to compute differentially expressed genes for the target dataset.
        """

        if self.ref_input is None and compute_cPCs:
            raise ValueError("Need fitted reference dataset to compute cPCs, run fit function first")

        if not compute_cPCs and self.cpca_loadings is None:
            raise ValueError("No cPCs found, transform needs to be run with compute_cPCs==True at least once")

        if target_expr is None and self.target_input is None:
            raise ValueError("No previous data found for target, transform needs to be run with target expression at least once")

        if not compute_cPCs and target_expr is None:
            raise ValueError("No use case for running transform without new target data when compute_cPCs==True")
        
        if transform_source == 'target':
            target = True
        elif transform_source == 'reference':
            target = False
        else:
            raise ValueError("Invalid transform source, must be 'target' or 'reference'")

        if compute_cPCs:
            
            if target_expr is not None:
                
                self.target_input = self.__checkExpression(target_expr, is_reference=False,compute_cPCs=True,target=True)

                # Cluster and find differential expression for target data
                self.target_clusters = self.__cluster(self.target_input)

                if len(set(self.target_clusters)) < 2:
                    raise ValueError("Only one cluster found in reference data, no differential expression possible")
                self.target_de_genes = self.__runDiffExprOnClusters(self.target_input, self.target_clusters)

                # Union of the top 1000 differentially expressed genes in each dataset
                self.de_genes = pd.Series(list(self.ref_de_genes[:self.topKGenes].index) +
                                          list(self.target_de_genes[:self.topKGenes].index)).drop_duplicates().to_list()

            else:
                print("INFO: No new target expression provided, using previously provided target dataset")

            # Subtract cluster average from cluster samples
            centered_ref_input = pd.concat(
                [
                    self.ref_input.loc[self.ref_clusters == val] - self.ref_input.loc[self.ref_clusters == val].mean(axis=0)
                    for val in set(self.ref_clusters)
                ]
            ).loc[self.ref_input.index]
            
            centered_target_input = pd.concat(
                [
                    self.target_input.loc[self.target_clusters == val] - self.target_input.loc[self.target_clusters == val].mean(axis=0)
                    for val in set(self.target_clusters)
                ]
            ).loc[self.target_input.index]
            
            # Compute contrastive PCs
            print("Running cPCA..")
            self.cpca_loadings, self.cpca_explained_var, self.target_pcs, self.ref_pcs = self.__runCPCA(centered_ref_input, centered_target_input)

            del centered_ref_input, centered_target_input
            gc.collect()

            print("Regressing top cPCs out of reference dataset..")
            transformed_ref = self.__regress_out_cpcs(self.ref_input, self.ref_pcs, self.cpca_loadings, regression_method)
            self.ref_transformed = transformed_ref

            print("Regressing top cPCs out of target dataset..")
            transformed_target = self.__regress_out_cpcs(self.target_input, self.target_pcs, self.cpca_loadings, regression_method)
            self.target_transformed = transformed_target

        # Using previously computed cPCs - for multi-dataset alignment
        else:
            # Allow some genes to be missing in new target dataset
            #TODO: implement a strategy to handle missing values
            target_expr = self.__checkExpression(target_expr, is_reference=False,compute_cPCs=False,target=target)
            
            if target:
                correct_pcs = self.target_pcs
            else:
                correct_pcs = self.ref_pcs

            transformed_target = self.__regress_out_cpcs(target_expr, correct_pcs, self.cpca_loadings, regression_method)
            num_target_samples = transformed_target.shape[0]
            #concatenate new targer with old target (this to facilitate the MNN step)

            if target:
                transformed_target = pd.concat([transformed_target,self.target_input])
            else:
                transformed_target = pd.concat([transformed_target,self.ref_input])
            #recover reference
            transformed_ref = self.ref_transformed   

        # Do MNN 
        print("Doing the MNN analysis using Marioni et al. method..")
        # Use top DE genes only
        varsubset = np.array([1 if i in self.de_genes else 0 for i in self.target_input.columns]).astype(bool)
        target_corrected, self.mnn_pairs = mnnpy.marioniCorrect(
            transformed_ref,
            transformed_target,
            var_index=list(range(len(self.ref_input.columns))),
            var_subset=varsubset,
            **self.mnn_kwargs,
        )

        if compute_cPCs:
            self.combined_output =  pd.concat([target_corrected, transformed_ref])
        else:
            #return only the new target samples
            target_corrected = target_corrected.iloc[:num_target_samples]
            return target_corrected

        print('Done')

        return self


    def computeMetricsForOutput(self, umap_rand_seed=14, UMAP_only=False, model_ids=None, tumor_ids=None):
        """
        Compute UMAP embedding and optionally clusters and tumor - model distance.
        
        Args:
            UMAP_only (bool, optional): Only recompute the UMAP. Defaults to False.
            umap_rand_seed (int, optional): Set seed for UMAP, to try an alternative. Defaults to 14.
            model_ids (list, optional): model IDs for computing tumor-CL distance. Defaults to None, in which case the reference index is used.
            tumor_ids (list, optional): tumor IDs for computing tumor-CL distance. Defaults to None, in which case the target index is used.
        
        Raises:
            ValueError: if there is no corrected expression matrix
        """
        if self.combined_output is None:
            raise ValueError("No corrected expression matrix found, run this function after transform()")

        print("Computing UMAP embedding...")
        # Compute UMAP embedding for results
        pca = PCA(self.pca_ncomp)
        pcs = pca.fit_transform(self.combined_output)
        
        umap_reduced = umap.UMAP(**self.umap_kwargs, random_state=umap_rand_seed).fit_transform(pcs)
        self.umap_reduced = pd.DataFrame(umap_reduced, index=self.combined_output.index, columns=['umap1','umap2'])

        if not UMAP_only:
            
            print('Computing clusters..')
            self.output_clusters = self.__cluster(self.combined_output)

            print("Computing tumor-CL distance..")
            pcs = pd.DataFrame(pcs, index=self.combined_output.index)
            if model_ids is None: model_ids = self.ref_input.index
            if tumor_ids is None: tumor_ids = self.target_input.index
            model_pcs = pcs[pcs.index.isin(model_ids)]
            tumor_pcs = pcs[pcs.index.isin(tumor_ids)]
            
            self.tumor_CL_dist = pd.DataFrame(metrics.pairwise_distances(tumor_pcs, model_pcs), index=tumor_pcs.index, columns=model_pcs.index)
        
        return self


    def makeNewReference(self):
        """
        Make a new reference dataset from the previously transformed reference+target datasets. 
        Used for multi-dataset alignment with previously computed cPCs and DE genes.
        
        """
        self.ref_input = self.combined_output
        self.target_input = None
        return self
    
    
    def save(self, file_name):
        """
        Save the model as a pickle file

        Args:
            file_name (str): name of file in which to save the model
        """
        # save the model
        with open(os.path.normpath(file_name), "wb") as f:
            pickle.dump(self, f)


    def load(self, file_name):
        """
        Load the model from a pickle file

        Args:
            file_name (str): pickle file to load the model from
        """
        with open(os.path.normpath(file_name), "rb") as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__)
        return self