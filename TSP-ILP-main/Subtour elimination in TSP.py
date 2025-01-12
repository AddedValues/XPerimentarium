# %% [markdown]
# # INTRODUCTION
# 
# In this article, I will explain and implement the well known Traveling Salesman Problem aka TSP with special focus on subtour elimination methods. We will use python to implement the  MILP formulation.The dataset contains the coordinates of various cities of India. The aim is to find a shortest path that cover all cities. We will cover the  following things in the article
# 
# 
# 1. **Input the data and visualize the problem**<br/><br/>
# 2. **Model TSP as MILP formulation w/o Subtour contraints**<br/><br/>
# 3. **Implement Subtour Elimination Method 1: MTZ's Approach**<br/><br/>
# 
# 4. **Implement Subtour Elimination Method 2: DFJ's Approach**<br/><br/>
# 5. **Compare  MTZ's formulation and DFJ's formulation**<br/><br/>
# 6. **Conclusion**<br/><br/>
# 
# 

# %% [markdown]
# # 1 Input the data and  problem visualization

# %% [markdown]
# The csv file *"tsp_city_data.csv"* contains the names of cities in India with thier latitute and longitute information. The first city *"Delhi"* is assumed to be starting point of trip (depot). The data input to TSP model is the distance matrix which stores the distance (or travel time or cost) from each city (location) to every other city. Thus, for a traveling salesman problem for *N* cities (location), the distance matrix is of size *N x N*.
# The varible * no_of_locs* in the code is used to define the first n no. of cities we want to include in our TSP problem data. The value is set 20 for now.  The pyhton *pandas* library is used to read csv file and distance matrix "*dis_mat*".

# %%
#import libraries
# %matplotlib inline
import os
import pulp
import pandas as pd
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import time
import copy

# %%
# This function takes locations as input and plot a scatter plot
def plot_fig(loc,heading="plot"):

    plt.figure(figsize=(10,10))
    for i,row in loc.iterrows():
        if i==0:
            plt.scatter(row["x"],row["y"],c='r')
            plt.text(row["x"]+0.2, row["y"]+0.2, 'DELHI (depot) ')
        else:
            plt.scatter(row["x"], row["y"], c='black')
            plt.text(row["x"] + 0.2, row["y"] + 0.2,full_data.loc[i]['CITy'] )
        plt.ylim(6,36)
        plt.xlim(66,96)
        plt.title(heading)

# This function takes route plan as input and return the ordered routes and subtours, if present
def get_plan(r0):
    r=copy.copy(r0)
    route = []
    while len(r) != 0:
        plan = [r[0]]
        del (r[0])
        l = 0
        while len(plan) > l:
            l = len(plan)
            for i, j in enumerate(r):
                if plan[-1][1] == j[0]:
                    plan.append(j)
                    del (r[i])
        route.append(plan)
    return(route)


# %%
os.chdir('TSP-ILP-main')
# set no of cities
no_of_locs = 6
data = pd.read_csv("tsp_city_data.csv")
full_data = data.iloc[0:no_of_locs,:]
d = full_data[['x','y']]
dis_mat = pd.DataFrame(distance_matrix(d.values,d.values), index=d.index, columns=d.index)
print("----------data--------------")
print(full_data)
print("-----------------------------")
plot_fig(d,heading="Problem Visualization")
   
plt.show()


# %% [markdown]
# # 2 Model TSP in  MILP  without Subtour elimination constraints
# 
# TSP problem can be modeled as Mixed Integer Linear Program. The LP Model is exlained as follows<br><br>
# **Data** <br><br>
# N= Number of location including depot (starting point) <br>
# $C_{i,j}$ = Edge cost from node i to node j  where i,j= [1...N]<br><br>
# **Decision Variable**<br> <br>
# $x_{i,j}$ = 1 if solution has direct path from node i to j, otherwise 0 <br>
# 
# 
# The LP model is formilated as follows<br>
# 
# ### MIN          $C_{i,j}$  $x_{i,j} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(1)$
# s.t <br><br>
# $\sum_{j=1}^{N} x_{i,j} = 1 \;\;\;\;\;\;\;\;\;  i ={1...N} \;\;\;\;\;\;\;\;\; (2)$<br> <br>
# $\sum_{i=1}^{N} x_{i,j} = 1  \;\;\;\;\;\;\;\;\; j ={1...N} \;\;\;\;\;\;\;\;\; (3)$<br> <br>
# $ x_{i,i} = 0  \;\;\;\;\;\;\;\;\; i ={1...N} \;\;\;\;\;\;\;\;\; (4)$<br> <br>
# 
# The objective (1) minimize the cost of tour. Constraints (2) and (3) ensures that for each node, there is only one outflow and inflow edge respectively. Thus, ach node is visited only once. Constraint (4) restrict outflow to one's own node.
# 
# 
# 
# In this article, python PuLP library is used for implementing MILP model in python. PuLP is an LP modeler written in Python. PuLP can call variety of LP solvers like CBC, CPLEX, GUROBI, SCIP to solve linear problems.It can be installed from the link https://pypi.org/project/PuLP/. The CBC solver is preinstalled in the PuLP library while one  has to install other solvers like gurobi, cplex  separately to use in PuLP. In this implementation CBC is used as LP solver.
# <br/><br/>
# 
# 

# %%
model = pulp.LpProblem('tsp',pulp.LpMinimize)

#define variable
x = pulp.LpVariable.dicts("x",((i,j) for i in range(no_of_locs) for j in range(no_of_locs)), cat='Binary')

#set objective
model+=pulp.lpSum(dis_mat[i][j]* x[i,j] for i in range(no_of_locs) for j in range(no_of_locs))

# st constraints
for i in range(no_of_locs):
    model += x[i,i] == 0
    model += pulp.lpSum(x[i,j] for j in range(no_of_locs)) == 1
    model += pulp.lpSum(x[j,i] for j in range(no_of_locs)) == 1

status = model.solve()
#status=model.solver()
print("-----------------")
print(status, pulp.LpStatus[status], pulp.value(model.objective))
route = [(i,j) for i in range(no_of_locs) for j in range(no_of_locs) if pulp.value(x[i,j]) == 1]

print(get_plan(route))
print(route)

# %%

plot_fig(d,heading="solution Visualization")
arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
for i, j in route:
    plt.annotate('', xy=[d.iloc[j]['x'], d.iloc[j]['y']], xytext=[d.iloc[i]['x'], d.iloc[i]['y']], arrowprops=arrowprops)

# %% [markdown]
# ![image info](./not_expected.jpg)

# %% [markdown]
# The optimal solution given by the LP model has subtours i.e <br>
# 1. **Tour 1 :** Delhi > Nagpur > Rajkot
# 2. **Tour 2 :** Kolkata > Dispur > Agartala
# 
# The solution given by the model has 2 tours but what required is the single tour that starts with depot (Delhi) and visit all locations one by one and ends at Delhi. To solve this problem and to get desired single tour, the subtour elimination constraints need to be added in the LP Model. <br> A subtour is a disjoint tour that does not include all the cities. The subtour elimination constraints are added to the LP model to eliminate the subtours and get the desired single tour. 
# 
# There are 2 well known formulations, DSF and MTZ (named after their authors). This article cover both the ideas and the implementation in python.
# 

# %% [markdown]
# # 3. MTZ Method for subtour elimination
# This formulation was proposed by Miller, Tucker, Zemlin. To eliminate subtours, decision variables representing times at which a location is visited is added. Variable for all locations except depot node are added.
# $t_{i}$ = time at which location i is visited , i =[2,...N]
# Finally what is required the constraint <br><br>
# $ t_{j}$ > $t_{i} \;\;\;\;\;\;\;\; if \;\;x_{i,j} =1$ 
# which can be formulated as LP constraints as follows <br>
# 
# $t_{j} \geq t_{i} - B(1-x_{i,j})  \;\;\;\;\;\;\;\;\;\;\;\;\; (5)\;\;\;\;\;$
# for some large value of B
# 
# ### How does constraint (5) remove subtours ?
# **Spoiler alert**: It does not remove subtours. It just makes the subtours infeasible by enforcing a monotonous timing of endpoints of route segments. <br>
# Lets takes an previous example and take the subtour Kolkata (k) > Dispur(d) > Agartala(a) <br>
# so, $x_{k,d}=1$, $x_{d,a}=1$, $x_{a,k}=1$
# Now as per constraint (5)
# 
# $t_{d} > t_{k}\;\;\;\;\;\;\;\;\;\;\;\;\; (6)$, <br>
# $t_{a} >  t_{d}\;\;\;\;\;\;\;\;\;\;\;\;\; (7)$, <br>
# $t_{k} > t_{a}\;\;\;\;\;\;\;\;\;\;\;\;\; (8)$ <br>
# from (6) and (7) $\;\;\;t_{a} > t_{k}$ but as per constraint (8) $\;\;\;\;t_{k} > t_{a}$  which is not possible.<br>
# **So adding constraint (5) will eliminate the sobtour.**
# 
# The complete Lp model is formulated as follows <br><br>
# **Data** <br><br>
# N= Number of location including depot (starting point) <br>
# $C_{i,j}$ = Edge cost from node i to node j  where i,j= [1...N]<br><br>
# **Decision Variable**<br> <br>
# $x_{i,j}$ = 1 if solution has direct path from node i to j, otherwise 0 <br>
# $t_{i}$ = time at which location i is visited , i =[2,...N]
# 
# 
# The LP model is formulated as follows<br>
# 
# ### MIN          $C_{i,j}$  $x_{i,j} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(9)$
# s.t <br><br>
# $\sum_{j=1}^{N} x_{i,j} = 1 \;\;\;\;\;\;\;\;\;  i ={1...N} \;\;\;\;\;\;\;\;\; (10)$<br> <br>
# $\sum_{i=1}^{N} x_{i,j} = 1  \;\;\;\;\;\;\;\;\; j ={1...N} \;\;\;\;\;\;\;\;\; (11)$<br> <br>
# $ x_{i,i} = 0  \;\;\;\;\;\;\;\;\; i ={1...N} \;\;\;\;\;\;\;\;\; (12)$<br> <br>
# $t_{j} \geq t_{i} - B(1-x_{i,j}) \;\;\;\;\;i,j=[2,....N] \;\;\;\;\;\;\;\;\;\;\;\;\; (13)$ <br><br>
# $x_{i,j}=\{0,1\}$ , $\;\;\; t_{i}=[1,...N-1]\;\;\;\;\;\;\;\;\;(14) $
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
start_t=time.time()
model=pulp.LpProblem('tsp',pulp.LpMinimize)

#define variable
x = pulp.LpVariable.dicts("x",((i,j) for i in range(no_of_locs) for j in range(no_of_locs)), cat='Binary')
t = pulp.LpVariable.dicts("t", (i for i in range(no_of_locs)), lowBound=1, upBound=no_of_locs, cat='Integer')

#set objective
model += pulp.lpSum(dis_mat[i][j]* x[i,j] for i in range(no_of_locs) for j in range(no_of_locs))

# st constraints
for i in range(no_of_locs):
    model += x[i,i]==0
    model += pulp.lpSum(x[i,j] for j in range(no_of_locs))==1
    model += pulp.lpSum(x[j,i] for j in range(no_of_locs)) == 1

#eliminate subtours
for i in range(no_of_locs):
    for j in range(no_of_locs):
        if i!=j and (i!=0 and j!=0):
            model += t[j]>=t[i]+1 - (2*no_of_locs)*(1-x[i,j])

status=model.solve()
#status=model.solver()
print("-----------------")
print(status,pulp.LpStatus[status],pulp.value(model.objective))
route = [(i,j) for i in range(no_of_locs) for j in range(no_of_locs) if pulp.value(x[i,j])==1]
print(route)
plot_fig(loc=d, heading="solution Visualization")
arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
for i, j in route:
    plt.annotate('', xy=[d.iloc[j]['x'], d.iloc[j]['y']], xytext=[d.iloc[i]['x'], d.iloc[i]['y']], arrowprops=arrowprops)

print("time taken by MTZ formulation = ", time.time()-start_t)

# %% [markdown]
# ![image info](./1.jpeg)

# %% [markdown]
# ![image_info](./wait.jpeg)

# %% [markdown]
# # 4. DFJ Method for subtour elimination
# This formulation was proposed by  Dantzig, Fulkerson,Jhonson.To eliminate subtours, for every set **S** of cities , add a constraint saying that the tour leaves **S** at least once. <br><br>
# $\sum_{i\in S} \sum_{j\notin S}\;\; x_{i,j}>=1\;\;\;\;\;\;\;\; S \subseteq \{1,2,....n\} ,\;\;\; 1 \leq |S| \geq N-1   \;\;\;\;\;\;(15)$<br><br>
# 
# ## How does this constraint eliminate subtours?
# Lets takes an same example and take a set $S_{i}$= {kolkata, Dispur, Agartala} and the rest of the cities be represented by $s_{i}^{'}$={ Delhi(del),Rajkot(r), Nagpur(n)}<br>
# Now as per constraint (15), the new constraint added is as follows <br>
# $ \sum_{i \in s_{i}^{'}} x_{kolkata,i} \sum_{i \in s_{i}^{'}} x_{Dispur,i}\sum_{i \in s_{i}^{'}} x_{Agartala,i} \geq 1\;\;\;\;\;\;\;\;\;\; (16)$ <br>
# Since there is no edge going to any other node is this set (due to subtour), this equation is not satisfied for the the set $S_{i}$= {{kolkata, Dispur, Agartala}. So, by adding constraint (15), this solution becomes infeasible and all subtours will be eliminated.
# 
# ## Modification in DFJ Method 
# 
# For *N* cities,the nunber of posssible sets add up to $ 2^n$ i.e the number of constraints grow exponetially. So, Instead of adding constraints for all the possible sets, only some constraints are added. Given a solution to LP model(without having subtour elimination constraints) with subtours, one can quickly find the subset for which subtour DFS subtour constraint is eliminated. In the example above, one needs to add only 2 constraints and not $2^5$ constraints.<br><br>
# 
# $ \sum_{i \in s_{i}^{'}} x_{kolkata,i} \sum_{i \in s_{i}^{'}} x_{Dispur,i}\sum_{i \in s_{i}^{'}} x_{Agartala,i} \geq 1\;\;\;\;\;\;\;\;\;\; (17)$ <br><br>
# $ \sum_{i \in s_{i}^{'}} x_{Delhi,i} \sum_{i \in s_{i}^{'}} x_{Rajkot,i}\sum_{i \in s_{i}^{'}} x_{Nagpur,i} \geq 1\;\;\;\;\;\;\;\;\;\; (18)$ <br><br>
# These constraints can also be written as<br><br>
# $ x_{kolkata,Dispur}+x_{Dispur,Agartala} +x_{agartala,kolkata} \leq 2 \;\;\;\;\;\;\;(19)$<br>
# $ x_{Dispur,kolkata}+x_{kolkata,agartala}+x_{Agartala,Dispur}  \leq 2 \;\;\;\;\;\;\;(20)$<br>
# $ x_{Delhi,Rajkot}+x_{Rajkot,Nagpur} +x_{Nagpur,Delhi} \leq 2 \;\;\;\;\;\;\;(21)$<br>
# $ x_{Rajkot,Delhi}+x_{Delhi,Nagpur} +x_{Nagpur,Rajkot} \leq 2 \;\;\;\;\;\;\;(22)$<br>
# 
# 
# 
# 
# 
# So, the higer level algorithm is as follows<br>
# ### Higher level Algorithm for DFS
# step 1. Solve TSP problem with LP formulation w/o Subtour Constraints<br><br>
# step 2. If no subtour present in the current solution, goto step 6<br><br>
# step 3. Add subtour constraint **only** for the subtours present in current solution.<br><br>
# step 4. Solve TSP problem with newly added constraint.<br><br>
# step 5. goto step 2<br><br>
# step 6. Return the final TSP solution<br><br>
# 
# 

# %%
start = time.time()
model = pulp.LpProblem('tsp',pulp.LpMinimize)

#define variable
x = pulp.LpVariable.dicts("x",((i,j) for i in range(no_of_locs) for j in range(no_of_locs)), cat='Binary')

#set objective
model += pulp.lpSum(dis_mat[i][j]* x[i,j] for i in range(no_of_locs) for j in range(no_of_locs))

# st constraints
for i in range(no_of_locs):
    model += x[i,i] == 0
    model += pulp.lpSum(x[i,j] for j in range(no_of_locs)) == 1
    model += pulp.lpSum(x[j,i] for j in range(no_of_locs)) == 1
    
status=model.solve()

route = [(i,j) for i in range(no_of_locs) for j in range(no_of_locs) if pulp.value(x[i,j])==1]
route_plan = get_plan(route)
subtour = []
while len(route_plan) != 1:
    for i in range(len(route_plan)):
        print(f'Subtour {i}: ', route_plan[i])
        model += pulp.lpSum(x[route_plan[i][j][0], route_plan[i][j][1]] \
                            for j in range(len(route_plan[i]))) <= len(route_plan[i])-1

    status = model.solve()
    route = [(i,j) for i in range(no_of_locs) for j in range(no_of_locs) if pulp.value(x[i, j]) == 1]
    route_plan = get_plan(route)
    subtour.append(len(route_plan))

print("-----------------")
print(status,pulp.LpStatus[status], pulp.value(model.objective))
print(route_plan)
print("no. of times LP model is solved = ",len(subtour))
print("subtour log (no. of subtours in each solution))",subtour)
print("Time taken by DFS formulation = ", time.time()-start)
plot_fig(d, heading="solution Visualization")
arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
for i, j in route_plan[0]:
    plt.annotate('', xy=[d.iloc[j]['x'], d.iloc[j]['y']], xytext=[d.iloc[i]['x'], d.iloc[i]['y']], arrowprops=arrowprops)

plt.show()

print("total time = ",time.time()-start)

# %% [markdown]
# # Compare MTZ's Formulation vs DFJ's formulation
# Since two approaches for subtour elimination have been discussed in the this articel, its time to compare the two.
# MTZ's approach introduces $n^2$ constraints  (one for each pair (i,j) where i,j=[1..n]) while DFJ's apporach introduce subtour constraints for all possible sets of locations i.e $2^n$ for n locations. Thus, MTZ's apporach adds polynomial number of constraints while DFJ's approach introduce exponential number of constraints.<br> <br>
# 
# In terms of decision variables, MTZ approach introduces *n* new decision variables ($t_{i}$ for i =[1..n]).ON the other hand, DFS introduces no new decision variable. MTZ's approach has to be solved only once to get an optimal solution While DFJ is generally implemented as modified version and it is solved iterativey ( i.e LP model has to be solved multiple times with new subtour constraints added every time).<br><br>
# 
# There is no clear winner among the two and for some problems DFJ gives solution faster than MTZ  and for some problems MTZ is faster. But When DHJ has an efficient branch and bound approach due to which it become more efficient than MTZ. Also, MTZ’s formulation is weakeri.e the feasible region has the same integer points, but includes more fractional points.
# 
# 
# 
# 
# 

# %% [markdown]
# # Conclusion
# 
# In this article, ILP formulation of TSP is explained with special focus on subtour elimination approaches. TSP problem is a special case of Vehicle Routing Problem (VRP) with no. of vehicle equal to 1. But, subtour elimination is a core issue in VRP as well which is solved by using same techniques. 


