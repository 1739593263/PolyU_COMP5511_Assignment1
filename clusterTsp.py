import numpy as np
import pandas as pd
import geatpy as ea
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # pip install scikit-learn

class ClusterTsp(ea.Problem):
    def __init__(self):
        self.origin_info = pd.read_csv('TSP.csv', delimiter=",")
        self.address = self.origin_info.loc[:, ['XCOORD', 'YCOORD']]
        # self.address = self.address[:,'XCOORD']+100
        for addr in range(100):
            self.address.loc[self.address.shape[0]] = [self.address.iloc[addr]['XCOORD']+100, self.address.iloc[addr]['YCOORD']]     

    def initProblem(self, addr):
        name = 'Cluster tsp' 
        M = 1
        maxormins = [1]             # targeted function - 1 minimize - 2 maximize
        Dim = addr.shape[0]-1       # decision variable: each city except start city can only have one-time pass.
        varTypes = [1] * Dim        # 0=serialize 1=discrete
        lb = [0] * Dim              # bottom limitation
        ub = [Dim] * Dim            # top limitation
        lbin = [0] * Dim            # bottom border（0-exclude, 1-include）
        ubin = [1] * Dim            # top border（0-exclude, 1-include）
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def clusterFunc(self):
        # Number of clusters
        num_clusters = 2

        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.address['cluster'] = kmeans.fit_predict(self.address[['XCOORD', 'YCOORD']])
        addresses = self.address
        # Print the resulting DataFrame with cluster labels

        # print(addresses)
        clusters = {}
        for i in range(num_clusters):
            clusters[i] = addresses.loc[addresses.cluster==i, ['XCOORD', 'YCOORD']]

        return clusters
        # Visualize the clusters
        # plt.figure(figsize=(10, 6))
        # plt.scatter(self.address['XCOORD'], self.address['YCOORD'], c=addresses['cluster'], cmap='viridis', marker='o')
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')  # Cluster centers
        # plt.title('K-means Clustering of Addresses')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.show()
    
    def aimFunc(self, pop):
        x = pop.Phen.copy() # target matrix
        # fitness[]
        X = np.hstack([x, x[:, [0]]]).astype(int)   # make each route go back to the start point at the end
        ObjV = [] # distance list
        
        for i in range(pop.sizes):
            journey = self.address.iloc[X[i]]
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0))) # Calculate the total distance
            ObjV.append(round(distance, 3))   
        pop.ObjV = np.array([ObjV]).T

def runAlgorithm(problem, population):
    # Strengthen Elitist GA Algorithm(增强精英保留的遗传算法类)
    algorithm = ea.soea_SEGA_templet(problem, population)
    algorithm.MAXGEN = 100      # Evolving times
    algorithm.mutOper.Pm = 0.8  # Mutation rate
    algorithm.logTras = 0       # Log generation (0-no log)
    algorithm.verbose = False   # If printing out logs
    algorithm.drawing = 0       # drawing method (0-no draw, 1-draw res, 2-draw processing ani, 3-draw deci ani)

    [BestIndi, population] = algorithm.run()
    # print('evaluation num: %s' % algorithm.evalsNum) # NIND * MAXGEN
    # print('pass time: %s' % algorithm.passTime)
    return [BestIndi, population]

def runUnitAlgorithm(problem, populations):
    # soea_multi_SEGA_templet : class - Multi-population Strengthen Elitist GA Algorithm(增强精英保留的多种群协同遗传算法类)
    algorithm = ea.soea_multi_SEGA_templet(problem, populations)
    algorithm.MAXGEN = 500
    algorithm.logTras = 0       # Log generation (0-no log)
    algorithm.verbose = False   # If printing out logs
    algorithm.drawing = 0       # drawing method (0-no draw, 1-draw res, 2-draw processing ani, 3-draw deci ani)
    [BestIndi, pre_population] = algorithm.run()
    BestIndi.Encoding = "P"
    return BestIndi
    
def drawRes(BestIndi, problem):
    if BestIndi.sizes != 0:
        print('lowest distance: %s' % BestIndi.ObjV[0][0])
        print('Optimal route:')
        best_journey = np.hstack([0, BestIndi.Phen[0,:], 0])
        for i in range(best_journey.size):
            print(int(best_journey[i]), end=' ')
        print()
        # print(best_journey)
        # painting
        plt.figure()
        plt.plot(problem.address.iloc[best_journey]['XCOORD'], 
                problem.address.iloc[best_journey]['YCOORD'], 
                c='black')
        plt.plot(problem.address.iloc[best_journey]['XCOORD'], 
                problem.address.iloc[best_journey]['YCOORD'], 
                'o', c='black')

        plt.grid(True)
        plt.title('Cluster TSP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    else:
        print('NO AVAILABLE ROUTE')

if __name__=="__main__":
    problem = ClusterTsp()
    clusters = problem.clusterFunc()
    NIND = 50       # chorm size in a population
    Encoding = 'P'  # Permutation
    populations = []
    
    for i in range(len(clusters)):
        problem.initProblem(clusters[i])
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        [BestIndi, population] = runAlgorithm(problem, population)
        populations.append(population)


    combined_addr = None
    pre_population = None
    for i in range(len(clusters)):
        
        if i==0:
            pre_population = populations[i]
            combined_addr = clusters[i]
            continue

        # concating two clusters to build a new env
        tmp = combined_addr
        combined_addr = pd.concat([clusters[i], tmp], ignore_index=True)
        
        problem.initProblem(combined_addr)
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        pre_population.Field = Field
        populations[i].Field = Field
        # update target matrix to the matched values after the env amplifying 
        populations[i].Phen+=(len(pre_population.Phen))

        pre_population = runUnitAlgorithm(problem, [pre_population, populations[i]])
    
    drawRes(pre_population, problem)