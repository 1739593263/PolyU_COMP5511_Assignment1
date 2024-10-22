import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt

class CTSP(ea.Problem):
    def __init__(self):
        name = 'classic tsp' 
        M = 1
        self.address = np.loadtxt('TSP.csv', delimiter=",", skiprows=1, usecols=(1, 2))
        maxormins = [1]             # targeted function - 1 minimize - 2 maximize
        Dim = self.address.shape[0]-1   # decision variable: each city except start city can only have one-time pass.
        varTypes = [1] * Dim        # 0=serialize 1=discrete
        lb = [0] * Dim              # bottom limitation
        ub = [Dim] * Dim            # top limitation
        lbin = [0] * Dim            # bottom border（0-exclude, 1-include）
        ubin = [1] * Dim            # top border（0-exclude, 1-include）
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        x = pop.Phen.copy() # target matrix
        # fitness[]
        X = np.hstack([x, x[:, [0]]]).astype(int)   # make each route go back to the start point at the end
        ObjV = [] # distance list
        
        for i in range(pop.sizes):
            journey = self.address[X[i], :]
            distance = 0.0 
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0))) # Calculate the total distance
            ObjV.append(round(distance, 3))   
        pop.ObjV = np.array([ObjV]).T
    

if __name__=="__main__":
    problem = CTSP()

    Encoding = 'P'  # Permutation
    NIND = 100        # Individual size in the population
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    
    # Strengthen Elitist GA Algorithm(增强精英保留的遗传算法类)
    algorithm = ea.soea_SEGA_templet(problem, population)
    algorithm.MAXGEN = 1350     # Evolved times
    algorithm.mutOper.Pm = 0.8  # Mutation rate
    algorithm.logTras = 0       # Log generation (0-no log)
    algorithm.verbose = False   # If printing out logs
    algorithm.drawing = 1       # drawing method (0-no draw, 1-draw res, 2-draw processing ani, 3-draw deci ani)
    
    [BestIndi, population] = algorithm.run()
    # BestIndi.save()             # save the optimal route into the file   

    print('evaluation num: %s' % algorithm.evalsNum)
    print('pass time: %s' % algorithm.passTime)
    
    if BestIndi.sizes != 0:
        print('lowest distance: %s' % BestIndi.ObjV[0][0])
        print('Optimal route:')
        best_journey = np.hstack([0, BestIndi.Phen[0,:], 0])
        for i in range(best_journey.size):
            print(int(best_journey[i]), end=' ')
        print()

        # painting
        plt.figure()
        plt.plot(problem.address[best_journey.astype(int), 0], 
                problem.address[best_journey.astype(int), 1], 
                c='black')
        plt.plot(problem.address[best_journey.astype(int), 0], 
                problem.address[best_journey.astype(int), 1], 
                'o', c='black')
        
        # for i in range(len(best_journey)):
        #     plt.text(problem.address[int(best_journey[i]), 0], 
        #              problem.address[int(best_journey[i]), 1], 
        #              chr(int(best_journey[i])), fontsize=20)

        plt.grid(True)
        plt.title('Classical TSP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    else:
        print('NO AVAILABLE ROUTE')

        
