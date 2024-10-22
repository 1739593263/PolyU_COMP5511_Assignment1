import numpy as np
import pandas as pd
import geatpy as ea
import matplotlib.pyplot as plt

class MultiOO(ea.Problem):
    def __init__(self):
        self.beta = 0.3             # variance for the weight calculation
        self.origin_info = pd.read_csv('TSP.csv', delimiter=",")
        self.addresses = self.origin_info.loc[:, ['XCOORD', 'YCOORD']]
        self.profits = self.origin_info.loc[:, ['PROFIT']]
    
    def initso(self):
        name = 'weighting objective functions-based method in tsp'
        M = 1
        maxormins = [1]             # targeted function - 1 minimize - -1 maximize
        Dim = self.addresses.shape[0]-1   # decision variable: each city except start city can only have one-time pass.
        varTypes = [0] * Dim        # 0=serialize 1=discrete
        lb = [0] * Dim              # bottom limitation
        ub = [Dim] * Dim            # top limitation
        lbin = [0] * Dim            # bottom border（0-exclude, 1-include）
        ubin = [1] * Dim            # top border（0-exclude, 1-include）
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def calWeight(self, distance, profit):
        return distance - self.beta * profit

    def aimFunc(self, pop):
        x = pop.Phen.copy() # target matrix
        # fitness[]
        X = np.hstack([x, x[:, [0]]]).astype(int)   # make each route go back to the start point at the end
        ObjV = [] # distance list
        
        for i in range(pop.sizes):
            journey = self.addresses.loc[X[i]]
            distance = 0.0 
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0))) # Calculate the total distance
            # Calculate the total profit
            profit = 0.0
            # print(self.profits)
            # print(self.profits.iloc[j])
            profit = np.sum(self.profits.iloc[:,0])
            # for j in x[i]:
            #     profit=profit+self.profits.iloc[j,0]
            weight = self.calWeight(distance, profit)
            ObjV.append(round(weight, 3))
        pop.ObjV = np.array([ObjV]).T
    
def drawRes(Type, BestIndi, problem):
    if BestIndi.sizes != 0:
        print('lowest weight: %s' % BestIndi.ObjV[0][0])
        print('Optimal route:')
        best_journey = np.hstack([0, BestIndi.Phen[0,:], 0])
        for i in range(best_journey.size):
            print(int(best_journey[i]), end=' ')
        print()

        # painting
        plt.figure()
        plt.plot(problem.addresses.iloc[best_journey, 0], 
                problem.addresses.iloc[best_journey, 1], 
                c='black')
        plt.plot(problem.addresses.iloc[best_journey, 0], 
                problem.addresses.iloc[best_journey, 1], 
                'o', c='black')

        plt.grid(True)
        plt.title('Multi-Object Optimization TSP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    else:
        print('NO AVAILABLE ROUTE')

if __name__=="__main__":
    problem = MultiOO()
    problem.initso()
    Encoding = 'P'    # Permutation
    NIND = 100        # Individual size in the population
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    
    # Strengthen Elitist GA Algorithm(增强精英保留的遗传算法类)
    algorithm = ea.soea_SEGA_templet(problem, population)
    algorithm.MAXGEN = 500      # Evolved times
    algorithm.mutOper.Pm = 0.8  # Mutation rate
    algorithm.logTras = 0       # Log generation (0-no log)
    algorithm.verbose = False   # If printing out logs
    algorithm.drawing = 0       # drawing method (0-no draw, 1-draw res, 2-draw processing ani, 3-draw deci ani)
    
    [BestIndi, population] = algorithm.run()
    # BestIndi.save()             # save the optimal route into the file   

    print('evaluation num: %s' % algorithm.evalsNum)
    print('pass time: %s' % algorithm.passTime)
    drawRes('weighting objective function', BestIndi, problem)
    