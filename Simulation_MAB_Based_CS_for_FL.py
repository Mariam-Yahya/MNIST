import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
from math import sqrt
from math import log2
from math import log10
from pathlib import Path
import random 


def simulate_points(nb_points=20,width=500,height=500):
    #Simulation window parameters
    xMin=0;xMax=width
    yMin=0;yMax=height
    xDelta=xMax-xMin;yDelta=yMax-yMin #rectangle dimensions
    areaTotal=xDelta*yDelta

    #Point process parameters
    lambda0=nb_points/(width*height)

    #Simulate Poisson point process

    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()     #Poisson number of points
    while numbPoints != nb_points :
        numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()  

    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin  #x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin  #y coordinates of Poisson points

    return[xx,yy]


def calculate_distances(pts,width=500,height=500):
    xAP=width/2
    yAP=height/2
    distances=[]
    for i in range(len(pts[0])) :
        distances.append(sqrt( (pts[0][i]-xAP)**2 + (pts[1][i]-yAP)**2  )*0.001 )
    return distances


path_points=Path("points.txt")
if(path_points and path_points.is_file()):
    with open("points.txt", 'r') as f:
        points = [line.rstrip('\n') for line in f]
        xx=[float(points[i]) for i in range(len(points)) if i % 2 == 0 ]
        yy=[float(points[i]) for i in range(len(points)) if i % 2 == 1 ]
        points = [xx , yy]
else :
    points=simulate_points()

with open("points.txt", 'w') as f:
    for i in range(len(points[0])):
        f.write(str(float(points[0][i])) + '\n' + str(float(points[1][i])) + '\n')


path_distances=Path("distances.txt")
if(path_distances and path_distances.is_file()):
    with open("distances.txt", 'r') as f:
        distances = [line.rstrip('\n') for line in f]
else :
    distances=calculate_distances(pts=points)

with open("distances.txt", 'w') as f:
    for d in distances:
        f.write(str(d) + '\n')


distances = [float(i) for i in distances]

#Plotting
plt.scatter(points[0],points[1], edgecolor='b', facecolor='none', alpha=0.5 )
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def simulate_path_loss(distances) :
    return list(map(lambda d : 128.1 + 37.6 * log10(d),distances))

path_loss=simulate_path_loss(distances)

def simulate_avg_distribution_rate(uplink_power,sigma,path_loss) :
    avg_channel_gain = list(np.random.exponential(1,20))
    avg_dstrb_rate=list()
    for i in range(len(path_loss)) :
        avg_dstrb_rate.append(log2( 1 +   10**((uplink_power-30)/10)    * abs ( avg_channel_gain[i] )**2  /  ( 10**(sigma-30/10) * (10**(path_loss[i])) / 10) )  )
    return avg_dstrb_rate

avg_distribution_rate_uplink=simulate_avg_distribution_rate(uplink_power=23,sigma=-107,path_loss=path_loss)
avg_distribution_rate_downlink=simulate_avg_distribution_rate(uplink_power=23,sigma=-107,path_loss=path_loss)



def simulate_avg_distribution_time(m0,B,avg_distribution_rate):   
    return list( map(lambda distrb_rate : m0 / (B * distrb_rate ) ,avg_distribution_rate))

avg_distribution_time_uplink=simulate_avg_distribution_time(m0=5*10**3,B=15*1e3,avg_distribution_rate=avg_distribution_rate_uplink)
avg_distribution_time_downlink=simulate_avg_distribution_time(m0=5*10**3,B=15*10**3,avg_distribution_rate=avg_distribution_rate_downlink)

print("average distribution time uplink : \n" + str(avg_distribution_time_uplink))
print("average distribution time downnlink : \n" + str(avg_distribution_time_downlink))


def simulate_update_time(batch):
    computing_capability=[float(np.random.uniform((0.5*i+0.5)*20,(0.5*i+1.5)*20,1)) for i in range(20)]
    return list(map( lambda cmpt_cap : batch / cmpt_cap,computing_capability))

update_times=simulate_update_time(batch=20)
print("local update time : \n" + str(update_times) )


total_time=[avg_distribution_time_downlink[k] + avg_distribution_time_uplink[k] + update_times[k] for k in range(20) ]
print(total_time)

def reward_arm_delay(avg_distribution_time_uplink, avg_distribution_time_downlink, update_times , k):
    return 1 - (avg_distribution_time_downlink[k] + avg_distribution_time_uplink[k] + update_times[k]) / 5


class UCB1():
    def __init__(self, counts, values):
        self.counts = counts # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
        self.values = values # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        return

    # Initialise k number of arms
    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
    
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        
        for arm in range(n_arms):
            bonus = sqrt((2 * log10(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        
        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return

    

def test_algorithm(algo, arms, num_sims, horizon):
    
    # Initialise variables for duration of accumulated simulation (num_sims * horizon_per_simulation)
    chosen_arms = [0.0 for i in range(num_sims * horizon)]
    rewards = [0.0 for i in range(num_sims * horizon)]
    cumulative_rewards = [0 for i in range(num_sims * horizon)]
    sim_nums = [0.0 for i in range(num_sims *horizon)]
    times = [0.0 for i in range (num_sims*horizon)]
    cumulative_regret = [0.0 for i in range(num_sims*horizon)]
    best_arms = [0.0 for i in range(num_sims*horizon)]
    

    for sim in range(num_sims):
        sim = sim + 1
        algo.initialize(len(arms))
        print("simulation number :"+str(sim))
        for t in range(horizon):
            t = t + 1
            index = (sim -1) * horizon + t -1
            sim_nums[index] = sim
            times[index] = t
            
            # Selection of best arm and engaging it
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            
            # Simulate arms
            avg_distribution_rate_uplink=simulate_avg_distribution_rate(uplink_power=23,sigma=-107,path_loss=path_loss)
            avg_distribution_rate_downlink=simulate_avg_distribution_rate(uplink_power=23,sigma=-107,path_loss=path_loss)
            avg_distribution_time_uplink=simulate_avg_distribution_time(m0=5*10**3,B=15*1e3,avg_distribution_rate=avg_distribution_rate_uplink)
            avg_distribution_time_downlink=simulate_avg_distribution_time(m0=5*10**3,B=15*10**3,avg_distribution_rate=avg_distribution_rate_downlink)
            update_times=simulate_update_time(batch=20)
            
            # Engage chosen Delay Arm and obtain reward info
            reward = reward_arm_delay(avg_distribution_time_uplink,avg_distribution_time_downlink,update_times,chosen_arm)
            rewards[index] = reward

            # Get all the rewards of other arms if they were to be played
            rewards_all_arms=[reward_arm_delay(avg_distribution_time_uplink,avg_distribution_time_downlink,update_times,i) for i in arms ] 
            optimal_reward=max(rewards_all_arms)

            best_arm=rewards_all_arms.index(max(rewards_all_arms))
            best_arms[index]=best_arm
            if t ==1:
                cumulative_rewards[index] = reward
                cumulative_regret[index] = optimal_reward - reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index-1] + reward
                cumulative_regret[index] = cumulative_regret[index-1] + optimal_reward - reward
                
            algo.update(chosen_arm, reward)
    
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards, cumulative_regret,best_arms,delay]

random.seed(1)


n_arms=len(distances)
nb_sims=250
nb_rounds=10000
arms = [i for i in range(20)]
algo = UCB1([], [])
algo.initialize(n_arms)
results = test_algorithm(algo, arms, nb_sims, nb_rounds)



sim_nums=results[0]
times=results[1]
chosen_arms=results[2] 
rewards=results[3] 
cumulative_rewards=results[4]
cumulative_regret=results[5]
best_arms=results[6]


def mean_result(result,step):
    avg_result=[0 for i in range(step)]
    for i in range(len(result)):
        avg_result[i % step] = avg_result[i % step] + result[i % step]/(len(result)/step)
    return avg_result

best_chosen_arm_mean=[]
lst=[]
for i in range(nb_rounds):
    for j in range(i,len(chosen_arms),nb_rounds):
        lst.append(chosen_arms[j])
    best_chosen_arm_mean.append(lst)
    lst=[]

for i in range(len(best_chosen_arm_mean)):
    best_chosen_arm_mean[i]=best_chosen_arm_mean[i].count(best_arms[i])/len(best_chosen_arm_mean[i])




plt.title('mean cumulative rewards \n'+ str(nb_sims)+' simulations. '+str(n_arms)+' Clients ')
plt.plot(mean_result(cumulative_rewards,nb_rounds))
plt.xlabel("round")
plt.ylabel("cumulative rewards")
plt.legend()
plt.show()
plt.clf()

plt.title('Mean Rate of Choosing Best Arm from \n'+ str(nb_sims)+' simulations. '+str(n_arms)+' Clients ')
plt.plot(best_chosen_arm_mean)
plt.xlabel("round")
plt.ylabel("mean of best arm chosen")
plt.legend()
plt.show()
plt.clf()

plt.title('Mean Cumulative Regret from \n'+ str(nb_sims)+' simulations. '+str(n_arms)+' Clients ')
plt.plot(mean_result(cumulative_regret,nb_rounds))
plt.xlabel("round")
plt.ylabel("cumulative regret")
plt.show()
plt.clf()