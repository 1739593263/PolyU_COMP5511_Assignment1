import numpy as np
import pandas as pd
import geatpy as ea
import matplotlib.pyplot as plt
import math as math

class timeCons(ea.Problem):
    def __init__(self):
        self.beta = 0.3             # variance for the weight calculation
        self.origin_info = pd.read_csv('TSP.csv', delimiter=",")
        self.addresses = self.origin_info.loc[:, ['XCOORD', 'YCOORD']]
        self.profits = self.origin_info.loc[:, ['PROFIT']]
        self.intervals = self.origin_info.loc[:, ['READY TIME', 'DUE TIME']]

    def initmo(self):
        name = 'Time window constraint problem'
        M = 2
        maxormins = [1, -1]
        Dim = self.addresses.shape[0]-1
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [Dim] * Dim
        lbin = [0] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        x = pop.Phen.copy() # target matrix
        # fitness[]
        X = np.hstack([x, x[:, [0]]]).astype(int)   # make each route go back to the start point at the end
        ObjV_dist = []
        ObjV_pro = []
        cv = []

        for i in range(pop.sizes):
            journey = self.addresses.loc[X[i]]
            interval = self.intervals.loc[X[i]]

            # Calculate the total distance
            distance = 0.0 
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0)))
            # Calculate the total profit
            profit = 0.0
            profit = np.sum(self.profits.iloc[:,0])
            # Calculate the constraint
            constraint = 0.0
            time = 0.0
            for i in range(len(interval)-1):
                constraint=constraint+(interval.iloc[i, 0]-time if time<interval.iloc[i, 0] else 
                            time-interval.iloc[i, 1] if time>interval.iloc[i, 1] else 0)
                
                time = time+math.sqrt((journey.iloc[i+1, 1]-journey.iloc[i, 1])**2 + (journey.iloc[i+1,0]-journey.iloc[i,0])**2)
            # constraint = np.sum(np.where(time < interval.iloc[:, 0], interval.iloc[:, 0] - time, 
            #             np.where(time > interval.iloc[:, 1], time - interval.iloc[:, 1], 0)))
            # c1 = np.sum(interval.iloc[:, 0]-time) # constrain1: READY TIME <= time
            # c2 = np.sum(time-interval.iloc[:, 1]) # constrain1: DUE TIME <= time

            ObjV_dist.append(round(distance, 3))
            ObjV_pro.append(round(profit, 3))
            cv.append(round(constraint, 3))
            time = time+distance
            # print(time)
            
        pop.ObjV = np.hstack([np.array([ObjV_dist]).T, np.array([ObjV_pro]).T])
        pop.CV = np.array([cv]).T
    
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
    problem = timeCons()
    problem.initmo()
    # print(problem.intervals)
    Encoding = 'P'    # Permutation
    NIND = 100        # Individual size in the population
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    
    # moea_awGA_templet : class - 多目标进化优化awGA算法类
    algorithm = ea.moea_awGA_templet(problem, population)
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