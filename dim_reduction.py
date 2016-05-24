"""
Fits and transforms high dimensional data to low dimensional data.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.metrics import mean_squared_error

def pca(data, n_components, train, test):
    # PCA
    '''pca = decomposition.PCA()
    pca.fit(data)
    print 'explained variance ratio'
    print pca.explained_variance_ratio_
    plt.rc("font", size=20)
    plt.plot(range(1,data.shape[1]+1), pca.explained_variance_ratio_)
    plt.xlabel('Component number')
    plt.ylabel('Explained variance ratio')
    plt.title('Scree Plot')
    plt.show()
    plt.close()'''
    
    pca = PCA(n_components).fit(data[train])
    data_reduced = pca.transform(data)
    
    name = 'PCA'

    return data_reduced, name, pca.inverse_transform

def kpca(data, n_components, train, test, kernel='linear', gamma=None, degree=3, coef0=1, alpha=0.1, evaluation=False):
    # Kernel PCA
    
    kpca = KernelPCA(n_components, fit_inverse_transform=True, kernel=kernel, gamma=gamma, degree=degree, 
                     coef0=coef0, alpha=alpha).fit(data[train])
    
    data_reduced = kpca.transform(data)
    
    if evaluation:
        data_rec = kpca.inverse_transform(data_reduced)
        loss = mean_squared_error(data[test], data_rec[test])
        return loss
    
    #name = 'Kernel PCA ('+kernel+')'
    name = 'Kernel PCA'

    return data_reduced, name, kpca.inverse_transform

def tsvd(data, n_components, train, test):
    # Truncated SVD
    
    tsvd = TruncatedSVD(n_components).fit(data[train])
    data_reduced = tsvd.transform(data)
    #print('explained variance ratio:')
    #print(tsvd.explained_variance_ratio_)
        
    name = 'Truncated SVD'

    return data_reduced, name, tsvd.inverse_transform
