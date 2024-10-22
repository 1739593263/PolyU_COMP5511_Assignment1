import numpy as np
import pandas as pd
import geatpy as ea
import matplotlib.pyplot as plt
from math import sin, cos

# from Algorithm.dynamic_SEGA_templet import dynamic_SEGA_templet
# from MyPopulation import Population

class DynamicPro(ea.Problem):
    def __init__(self):
        self.origin_info = pd.read_csv('TSP.csv', delimiter=",")
        self.address = self.origin_info.loc[:, ['XCOORD', 'YCOORD']]
        # self.address = np.loadtxt('TSP.csv', delimiter=",", skiprows=1, usecols=(1, 2))
        self.env = 0

        name = 'dynamic optimization problem' 
        M = 1
        maxormins = [1]             # targeted function - 1 minimize - 2 maximize
        Dim = 50 + 10*self.env - 1  # decision variable: each city except start city can only have one-time pass.
        varTypes = [1] * Dim        # 0=serialize 1=discrete
        lb = [0] * Dim              # bottom limitation
        ub = [Dim] * Dim            # top limitation
        lbin = [0] * Dim            # bottom border（0-exclude, 1-include）
        ubin = [1] * Dim            # top border（0-exclude, 1-include）
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        # print(pop)
        x = pop.Phen.copy() # target matrix
        # fitness[]
        X = np.hstack([x, x[:, [0]]]).astype(int)   # make each route go back to the start point at the end
        ObjV = [] # distance list
        # print(self.address)
        
        for i in range(pop.sizes):
            # journey = self.address[X[i], :]
            journey = self.address.iloc[X[i]]
            # print(journey)
            distance = 0.0 
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0))) # Calculate the total distance
            ObjV.append(round(distance, 3))
        # print(len(ObjV))
        # print()   
        pop.ObjV = np.array([ObjV]).T

    def update_info_env(self, env):
        x_new = 2 * env * np.cos(np.pi * env / 2)
        y_new = 2 * env * np.sin(np.pi * env / 2)
        self.address['XCOORD'] = self.address['XCOORD'] + x_new
        self.address['YCOORD'] = self.address['YCOORD'] + y_new

        # adjusting the decision nums and  (since some customers are not reached)
        Dim = 50 + 10*self.env - 1          # decision variable: each city except start city can only have one-time pass.
        varTypes = [1] * Dim        # 0=serialize 1=discrete
        lb = [0] * Dim              # bottom limitation
        ub = [Dim] * Dim            # top limitation
        lbin = [0] * Dim            # bottom border（0-exclude, 1-include）
        ubin = [1] * Dim            # top border（0-exclude, 1-include）
        # self.ranges = [0]*self.varTypes + [Dim]*self.varTypes
        ea.Problem.__init__(self, self.name, self.M, self.maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        print(f"update info at env {env}")

def runAlgorithm(problem, population):
    # Strengthen Elitist GA Algorithm(增强精英保留的遗传算法类)
    algorithm = ea.soea_SEGA_templet(problem, population)
    algorithm.MAXGEN = 100      # Evolving times
    algorithm.mutOper.Pm = 0.8  # Mutation rate
    algorithm.logTras = 0       # Log generation (0-no log)
    algorithm.verbose = False   # If printing out logs
    algorithm.drawing = 0       # drawing method (0-no draw, 1-draw res, 2-draw processing ani, 3-draw deci ani)

    [BestIndi, population] = algorithm.run()
    print('evaluation num: %s' % algorithm.evalsNum) # NIND * MAXGEN
    print('pass time: %s' % algorithm.passTime)
    return [BestIndi, population]

def drawRes(Type, BestIndi, problem):
    if BestIndi.sizes != 0:
        print(Type, 'lowest distance: %s' % BestIndi.ObjV[0][0])
        print(Type, 'Optimal route:')
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
        plt.title('Dynamic TSP Solution - %s'%Type)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    else:
        print('NO AVAILABLE ROUTE')

def main():
    problem = DynamicPro()
    NIND = 1        # chorm size in a population
    Encoding = 'P'  # Permutation
    MAXGEN = 100
    env = 0
    reuse_population = None
    reuse_BestIndi = None
    new_population = None
    new_BestIndi = None
    # print(problem.ub - problem.lb + 1)
    for gen in range(0, MAXGEN, 100):
        # ranges = (problem.borders) * (50 + 10*env)
        # NIND = 50 + 10*env      # chorm size in a population
        # print(NIND)
        # ranges[1]=ranges[1]-1
        if (gen%100 == 0):
            problem.update_info_env(env)
            env = (env+1)%6
            problem.env = env
    
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        print("Reachable customers num: ", problem.Dim+1)
        reuse_population = reuse_population if reuse_population is not None else ea.Population(Encoding, Field, NIND)
        reuse_population.Field = Field  # add the new Field to reuse_population for the next evolving.
        new_population = ea.Population(Encoding, Field, NIND)
        # reuse_population.sizes = new_population.sizes
        # print(reuse_population.ChromNum," ", new_population.ChromNum)
        # print(reuse_population.Lind," ", new_population.Lind)
        
        print('======================REUSING THE SOLUTION======================')
        [reuse_BestIndi, population] = runAlgorithm(problem, reuse_population)
        reuse_population = reuse_BestIndi

        print('======================WITHOUT REUSING THE SOLUTION======================')
        [new_BestIndi, population] = runAlgorithm(problem, new_population)
        new_population = new_BestIndi
        print('\n','******************','\n')
        
    drawRes("reusing the solution", reuse_BestIndi, problem)
    
    print()

    drawRes("without reusing the solution", new_BestIndi, problem)

if __name__=="__main__":
    main()
