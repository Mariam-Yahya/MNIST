from math import sqrt
from math import log
import random 
import matplotlib.pyplot as plt


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
            bonus = sqrt((2 * log(total_counts)) / float(self.counts[arm]))
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
    

class BernoulliArm():
    def __init__(self, p):
        self.p = p
    
    # Reward system based on Bernoulli
    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

    
def test_algorithm(algo, arms, num_sims, horizon):
    
    # Initialise variables for duration of accumulated simulation (num_sims * horizon_per_simulation)
    chosen_arms = [0.0 for i in range(num_sims * horizon)]
    rewards = [0.0 for i in range(num_sims * horizon)]
    cumulative_rewards = [0 for i in range(num_sims * horizon)]
    sim_nums = [0.0 for i in range(num_sims *horizon)]
    times = [0.0 for i in range (num_sims*horizon)]
    cumulative_regret = [0.0 for i in range(num_sims*horizon)]
    
    for sim in range(num_sims):
        sim = sim + 1
        algo.initialize(len(arms))
        
        for t in range(horizon):
            t = t + 1
            index = (sim -1) * horizon + t -1
            sim_nums[index] = sim
            times[index] = t
            
            # Selection of best arm and engaging it
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            
            # Engage chosen Bernoulli Arm and obtain reward info
            reward = arms[chosen_arm].draw()
            rewards[index] = reward
            
            # Get all the rewards of other arms if they were to be played
            other_arms=arms[:chosen_arm]+arms[chosen_arm+1:]
            rewards_all_arms=[i.draw() for i in other_arms] + [reward]
            optimal_reward=max(rewards_all_arms)
            
            if t ==1:
                cumulative_rewards[index] = reward
                cumulative_regret[index] = optimal_reward - reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index-1] + reward
                cumulative_regret[index] = cumulative_regret[index-1] + optimal_reward - reward
                
            algo.update(chosen_arm, reward)
    
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards, cumulative_regret]




random.seed(1)

means = [0.9, 0.6, 0.4, 0.2]
n_arms=len(means)
nb_sims=5000
nb_rounds=250
arms = list(map(lambda mu: BernoulliArm(mu), means))
algo = UCB1([], [])
algo.initialize(n_arms)
results = test_algorithm(algo, arms, 5000, 250)

sim_nums=results[0]
times=results[1]
chosen_arms=results[2] 
rewards=results[3] 
cumulative_rewards=results[4]
cumulative_regret=results[5]

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
    best_chosen_arm_mean[i]=best_chosen_arm_mean[i].count(means.index(max(means)))/len(best_chosen_arm_mean[i])



plt.title('mean cumulative rewards \n'+ str(nb_sims)+' simulations. '+str(len(means))+' Arms = '+str(means))
plt.plot(mean_result(cumulative_rewards,nb_rounds))
plt.xlabel("round")
plt.ylabel("cumulative rewards")
plt.legend()
plt.show()
plt.clf()

plt.title('Mean Rate of Choosing Best Arm from \n'+ str(nb_sims)+' simulations. '+str(len(means))+' Arms = '+str(means))
plt.plot(best_chosen_arm_mean)
plt.xlabel("round")
plt.ylabel("mean of best arm chosen")
plt.legend()
plt.show()
plt.clf()

plt.title('Mean Cumulative Regret from \n'+ str(nb_sims)+' simulations. '+str(len(means))+' Arms = '+str(means))
plt.plot(mean_result(cumulative_regret,nb_rounds))
plt.xlabel("round")
plt.ylabel("cumulative regret")
plt.show()
plt.clf()




