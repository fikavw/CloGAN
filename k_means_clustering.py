import numpy as np
from sklearn.cluster import KMeans

class KMeans1():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape
        
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
        
        ##initialize K means
        kmeans = x[np.random.permutation(np.arange(N))[:self.n_cluster],:] ##dimension (K x D)

        t = 0
        while t < self.max_iter+1:

            ## calc memberships
            dist=np.zeros((N,self.n_cluster))
            for k in range(self.n_cluster):
                dist[:,k] = np.linalg.norm((x - kmeans[k,:]), axis=1)

            memberships = np.argmin(dist,axis=1)
            one_hot_r = np.zeros((N, self.n_cluster))
            one_hot_r[np.arange(N),memberships]= 1


            ##compute J_new with current memberships and means
            J_new=0
            for n in range(N):
                for k in range(self.n_cluster):
                    J_new = J_new + one_hot_r[n,k]*dist[n,k]
            J_new = J_new/N

            # break if converged
            if t == 0:
                J = J_new + self.e*1000000 
                
            if np.abs(J-J_new) <=self.e:
                print('converged')
                break
                
            J = J_new

            ##compute new kmeans
            num_per_k = np.sum(one_hot_r, axis=0)
            kmeans_new = np.dot(one_hot_r.T, x)
            # weights_k = np.ones((self.n_cluster,))/np.sum(one_hot_r, axis=0)
            for k in range(self.n_cluster):
                if num_per_k[k]>0:
                    weights_k = 1/num_per_k[k]
                    kmeans[k,:]=kmeans_new[k,:]*weights_k
                else:
                    kmeans[k,:] = kmeans[k,:] ##that particular centroid remains unchanged 
            
            t +=1
            
            
        num_updates = t
                
        return kmeans, memberships, num_updates

    
    
    
class KMeans2():
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', random_state=None):
        
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.random_state = random_state
        
        self.km = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances, random_state=self.random_state)
    
    def fit(self, x):
    
        self.km.fit(x)
        
        memberships = self.km.labels_
        
        num_updates = self.max_iter
        
        kmeans = self.km.cluster_centers_
    
        return kmeans, memberships, num_updates
  

def pick_fromgroups2(memberships, per_class):

    '''Returns subgroups with maximally heterogeneous superlabels
        
    '''
    
    unique_membs = np.unique(memberships)

    per_group=int(per_class/unique_membs.shape[0])
   
    sizes=[]
    for i in range(unique_membs.shape[0]):
        sizes.append(np.where(memberships==i)[0].shape[0]) 
        

    holder=[]
    c=0
    for c in range(unique_membs.shape[0]):
        inds_c = np.where(memberships==c)[0]
        if inds_c.shape[0]>per_group:
            holder.extend(inds_c[:per_group])
        elif inds_c.shape[0]<=per_group:
            if inds_c.shape[0]==0:
                pass
            else:
                holder.extend(inds_c)

    inds_all = np.arange(memberships.shape[0])

    num_missing=per_class-len(holder)

    inds_diff = np.setdiff1d(inds_all,np.array(holder))   

    np.random.shuffle(inds_diff)

    holder.extend(inds_diff[:num_missing])

    chosen_inds = holder 
    
    
    return np.array(chosen_inds), memberships[chosen_inds]




class clustering():
    def __init__(self, n_cluster, cluster_type='kcenters', k_version=1):
        
        if cluster_type=='kcenters':
            if k_version==1:
                self.kmeans_clusters = KMeans1(n_cluster, max_iter=100, e=0.0001) ## my own implementation
            
            elif k_version==2:
                self.kmeans_clusters = KMeans2(n_cluster, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', random_state=None)
        

    def update_n_cluster(self, n_cluster):
        
        self.kmeans_clusters.n_cluster = n_cluster
        

    def pick_kcentered_images(self, images_perclass, per_class, flatten=True):
        '''Input is image vector with N x num_channels x space_size x space_size ]
            Perform clustering per class 
        '''
        # N x nc x space_size x space_size 
        images = np.copy(images_perclass)
        
        if flatten==True:
            images = np.reshape(images, (images.shape[0],images.shape[1]*images.shape[2]*images.shape[3]))
            
        kmeans, memberships, num_updates = self.kmeans_clusters.fit(images)
    
        chosen_inds, superlabels = pick_fromgroups2(memberships, per_class)
        
      
        return chosen_inds, superlabels
    
    
 
