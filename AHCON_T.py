from Game import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import GradientTape, expand_dims
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

class AHCON_Agent:
    def __init__(self, discount_factor = 0.9, temperature = 0.01, H_e = 30, H_p = 30, eta_e = 0.001, eta_p = 0.001, board = game(), eval_n=None, policy_n=None):
        '''
        An TD learning Agent is initialized with 
        discount factor - for TD learning
        temerature - action selection, the smaller means actions are chosen according to policy network and higher means at random
        H_e, H_p - hidden layer units
        eta_e, eta_p - learningrate Adam
        board - a game is passed on
        eval_n, policy_n - load saved models
        '''
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.eta_e = eta_e
        self.eta_p = eta_p
        
        self.rounds_trained=0 #only needed for teaching

        
        if eval_n is None:
            self.eval = keras.Sequential([
                layers.Dense(units = H_e, activation = self.sigmoid, input_shape = [145]),
                layers.Dense(units = 1, activation = self.sigmoid) 
            ])
        else:
            self.eval = keras.models.load_model (eval_n)
        
        
        # Only one policy network for moving "North" and turn data for other cases
        if policy_n is None:
            self.policy =keras.Sequential([
                layers.Dense(units = H_p, activation = self.sigmoid, input_shape = [145]),
                layers.Dense(units = 1, activation = self.sigmoid)
            ])
        else:
            self.policy = keras.models.load_model(policy_n)
            
            
        # 0 gives Nourth, 1 E, 2 S, 3 W
        
        self.optimizer_e = keras.optimizers.Adam(learning_rate=self.eta_e)
        
        self.optimizer_p = keras.optimizers.Adam(learning_rate=self.eta_p)
        
        self.loss_funktion = keras.losses.Huber()
        
        self.board = board

    def sigmoid(self,x):
        '''
        activation function for NN
        '''
        return (keras.activations.sigmoid(x)-0.5)*2
      
    def select_action(self, show_prob=False):
        '''
        action selection function
        Output: chosen direction and state of agents looking in chosen direction
        '''
        prob=[]
        sum=0
        for i in range(4):
            x = np.exp(self.policy(expand_dims(self.board.get_data(i),0))/ self.temperature)
            prob.append(x[0,0])
            sum += x[0,0]
        if show_prob:
            print(prob)
        chosen_dir = random.choices([0,1,2,3],weights = prob)[0]
        return (chosen_dir,self.board.get_data(chosen_dir))

        
    def train(self, rounds = 10,analyse = False, temperature_l = None,food_pos_list=None):
        '''
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temerature is set to 0.01
        '''
        if food_pos_list is None:
            food_pos_list = [None]*rounds
        self.board.restart(food_pos=food_pos_list[0])
        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in range(rounds):
            real_reinforcement = 0
            if temperature_l is None:
                #self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature=0.01
            else:
                self.temperature = temperature_l[i]
                
            # play moves until agent dies
            while(self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                e = self.eval(expand_dims(x,0))
                direction, turned_x = self.select_action()
                
                r = self.board.move_agent(direction=direction)
                
                real_reinforcement = r + self.discount_factor * real_reinforcement
                
                
                loss_funktion = self.loss_funktion
                
                y = self.board.get_data()
                
                e_prime = np.array([[r]]) + self.discount_factor * self.eval(expand_dims(y,0))[0,0]
                
                
                #Backprop eval
                with GradientTape() as tape:
                    e = self.eval(expand_dims(x,0))
                    loss_value = loss_funktion(e_prime,e)
                    
                # Update the weights of eval to minimize the loss value.
                gradients = tape.gradient(loss_value, self.eval.trainable_weights)
                self.optimizer_e.apply_gradients(zip(gradients, self.eval.trainable_weights))
            
                
                #Backprop policy 
                with GradientTape() as tape2:
                    e = self.policy(expand_dims(turned_x,0))
                    loss_value = loss_funktion(e_prime,e)

                # Update the weights of eval to minimize the loss value.
                gradients = tape2.gradient(loss_value, self.policy.trainable_weights)
                self.optimizer_p.apply_gradients(zip(gradients, self.policy.trainable_weights))
                
                
            real_reinforcement_list[i]=real_reinforcement
            lifetime[i]=self.board.time
            food_found[i]=self.board.food_found
            energy[i]=self.board.energy
            
            if (((i*20)/rounds)%1 == 0) and analyse:
                np.save("run/rl",real_reinforcement_list)
                np.save("run/lt",lifetime)
                np.save("run/ff",food_found)
                np.save("run/e",energy)


            self.board.restart(food_pos=food_pos_list[min(i+1,rounds-1)])
        
        if analyse:
            np.save("run/rl",real_reinforcement_list)
            np.save("run/lt",lifetime)
            np.save("run/ff",food_found)
            np.save("run/e",energy)
            
        return (real_reinforcement_list,lifetime,food_found,energy)
    
    
    
    def train_teaching_on_policy(self, rounds = 10, temperature_l = None, teaching=True, prio=False,pl=0.01,food_pos_list=None):
        '''
        only replays actions that are on policy, i.e. probability of >=pl
        if PER is performed, this doesn't make a difference, i.e. with PER there is no differentiation between policy/non policy actions
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temperature is set to 0.01
        incorporates experience replay from the most recont 100 actions and teaching
        experiences are always saved as [old state (not turned),direction,reinforcement,new state (not turned)] (+[td error] potentially)
        If teaching=False, then only experience replay will be performed.
        If prio=True, then prioritized exp. replay will be performed instead of regular exp. replay.
        '''
        if food_pos_list is None:
            food_pos_list = [None]*rounds
        self.board.restart(food_pos=food_pos_list[0])
        
        recent_exp=[]

        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in range(rounds):
            real_reinforcement = 0
            if temperature_l is None:
                #self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature=0.01
            else:
                self.temperature = temperature_l[i]
                
            # play moves until agent dies
            while(self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                direction, turned_x = self.select_action()
                
                r = self.board.move_agent(direction=direction)

          

                
                real_reinforcement = r + self.discount_factor * real_reinforcement
                
                
                loss_funktion = self.loss_funktion

                
                y = self.board.get_data()
                
                e_prime = np.array([[r]]) + self.discount_factor * self.eval(expand_dims(y,0))[0,0]
                
                
                
                #Backprop eval
                with GradientTape() as tape:
                    e = self.eval(expand_dims(x,0))
                    loss_value = loss_funktion(e_prime,e)
                    
                # Update the weights of eval to minimize the loss value.
                gradients = tape.gradient(loss_value, self.eval.trainable_weights)
                self.optimizer_e.apply_gradients(zip(gradients, self.eval.trainable_weights))
            
                
                #Backprop policy 
                with GradientTape() as tape2:
                    e = self.policy(expand_dims(turned_x,0))
                    loss_value = loss_funktion(e_prime,e)

                # Update the weights of eval to minimize the loss value.
                gradients = tape2.gradient(loss_value, self.policy.trainable_weights)
                self.optimizer_p.apply_gradients(zip(gradients, self.policy.trainable_weights))

                recent_exp.append([x,direction,r,self.board.get_data(),abs(loss_value)])
                if len(recent_exp)>100:
                    recent_exp=recent_exp[1:]
            
                
                
            real_reinforcement_list[i]=real_reinforcement
            lifetime[i]=self.board.time
            food_found[i]=self.board.food_found
            energy[i]=self.board.energy


            if not prio:

                for h in range(self.replay_number(self.rounds_trained)):#replays past experiences. replays only policy actions.
                    all_off_policy=False
                    not_allowed=[]
                    done=False
                    while not done:
                        k=self.choose_experience(len(recent_exp)-len(not_allowed))
                        exp=recent_exp[k+len([x for x in not_allowed if x<=k+len([y for y in not_allowed if y<=x])])]
                        prob=[]
                        for i in range(4):
                            x = np.exp(self.policy(expand_dims(self.board.turn_input_even_more(exp[0],i),0))/ self.temperature)
                            prob.append(x[0,0])
                        if prob[exp[1]]/sum(prob)>=pl:
                            self.train_experience(exp)
                            done=True
                        else:
                            not_allowed.append(k+len([x for x in not_allowed if x<=k+len([y for y in not_allowed if y<=x])]))
                            if len(not_allowed)==len(recent_exp):
                                done=True
                                print("No policy actions left")
                                all_off_policy=True
                    if all_off_policy==True:
                        print("all off policy")
                        break



            if prio:

                probs=[(recent_exp[t][-1]+0.001)**0.7 for t in range(len(recent_exp))]
                w_list=[(len(recent_exp)*probs[t]/sum(probs))**(-0.5-0.5*self.rounds_trained/rounds) for t in range(len(recent_exp))]#contains the weights w_i; missing '/max w_i'
                for h in range(self.replay_number(self.rounds_trained)):
                    exp=random.choices(recent_exp,weights=probs)[0]
                    t=recent_exp.index(exp)
                    new_delta=self.train_experience(exp,w_list[t]/max(w_list))
                    recent_exp[t][-1]=new_delta
                    probs[t]=(recent_exp[t][-1]+0.001)**0.7
                    w_list[t]=(len(recent_exp)*probs[t]/sum(probs))**(-0.5-0.5*self.rounds_trained/rounds)



            
            if teaching:

                prob=self.teaching_prob(self.rounds_trained)
                lists=['l1','l2','l3','l4','l5','l6','l7']
                l_name=random.choice(lists)
                with open(f'teaching/{l_name}.pickle', 'rb') as handle:
                    l = pickle.load(handle)
                for exp in l:
                    if prob>=random.random():
                        self.train_experience(exp) 


            self.rounds_trained+=1
            self.board.restart(food_pos=food_pos_list[min(i+1,rounds-1)])

        
            
        return (real_reinforcement_list,lifetime,food_found,energy)
            
            
                
    def run(self):
        '''
        runs agent with temperature 0.01 and returns a visual
        '''
        vis=[]
        self.temperature=0.01
        while(self.board.is_alive()):
            vis.append(self.board.vis())
            
            self.board.move_enemies()
            #self.board.visual.append(self.board.vis())
            x = self.board.get_data()
            e = self.eval(expand_dims(x,0))
            
            direction, _ = self.select_action()
            r = self.board.move_agent(direction=direction)
           
        vis.append(self.board.vis())
        ff = self.board.food_found
            
        self.board.restart()
        return (ff,vis)
    
    def run_no_vis(self, temperature=0.01,visual=False,food_pos=None):
        '''
        runs agent 
        Output:
        real reinforcemnt
        time lasted
        food found
        energy left at the end
        visual
        '''
        self.board.restart(food_pos=food_pos)
        self.temperature=temperature
        real_reinforcement = 0
        vis=[]
        while(self.board.is_alive()):
            self.board.move_enemies()

            x = self.board.get_data()
            e = self.eval(expand_dims(x,0))
            
            direction, _ = self.select_action()
            r = self.board.move_agent(direction=direction)
            real_reinforcement = r + self.discount_factor * real_reinforcement
            if visual:
                vis.append(self.board.vis())
            
        stats = (real_reinforcement,self.board.time,self.board.food_found,self.board.energy,vis) 
        self.board.restart()
        return stats
    
    def analyse_run(self, rounds=50,visual=False,food_pos_list=None):
        '''
        runds agents for multiple rounds
        Output:
        real reinforcemnt
        time lasted
        food found
        energy left at the end
        visuals
        '''
        if food_pos_list is None:
            food_pos_list = [None]*rounds
        real_reinforcement_list = np.zeros(rounds)
        lifetime = np.zeros(rounds)
        food_found = np.zeros(rounds)
        energy = np.zeros(rounds)
        vis=[]
        for i in range(rounds):
            rr, lt, ff, e, v = self.run_no_vis(visual=visual,food_pos=food_pos_list[i])
            real_reinforcement_list[i] = rr 
            lifetime[i] =lt
            food_found[i] =ff
            energy[i] =e
            vis.append(v)
        return (real_reinforcement_list,lifetime,food_found,energy, vis)
    
    
    def choose_experience(self,m): #Appendix B from the paper.
        w=min(3,1+0.02*m)
        rand=random.random()
        return int(m*math.log(1+rand*(math.e**w-1))/w)

    def teaching_prob(self,round):
        return(2/5*math.e**(-round/100)+1/10)

    def replay_number(self,round):
        if round<14:
            return 12
        if round<26:
            return 11
        if round<38:
            return 10
        if round<51:
            return 9
        if round<76:
            return 8
        if round<101:
            return 7
        if round<151:
            return 6
        if round<201:
            return 5
        return 4
    
    def train_teaching(self, rounds = 10, temperature_l = None, teaching=True, prio=False,food_pos_list=None):
        '''
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temperature is set to 0.01
        incorporates experience replay from the most recont 100 actions and teaching
        experiences are always saved as [old state (not turned),direction,reinforcement,new state (not turned)] (+[td error] potentially)
        If teaching=False, then only experience replay will be performed.
        If prio=True, then prioritized exp. replay will be performed instead of regular exp. replay.
        '''
        
        if food_pos_list is None:
            food_pos_list = [None]*rounds
        self.board.restart(food_pos=food_pos_list[0])
        
        recent_exp=[]

        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in range(rounds):
            real_reinforcement = 0
            if temperature_l is None:
                #self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature=0.01
            else:
                self.temperature = temperature_l[i]
                
            # play moves until agent dies
            while(self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                direction, turned_x = self.select_action()
                
                r = self.board.move_agent(direction=direction)

          

                
                real_reinforcement = r + self.discount_factor * real_reinforcement
                
                
                loss_funktion = self.loss_funktion

                
                y = self.board.get_data()
                
                e_prime = np.array([[r]]) + self.discount_factor * self.eval(expand_dims(y,0))[0,0]
                
                
                
                #Backprop eval
                with GradientTape() as tape:
                    e = self.eval(expand_dims(x,0))
                    loss_value = loss_funktion(e_prime,e)
                    
                # Update the weights of eval to minimize the loss value.
                gradients = tape.gradient(loss_value, self.eval.trainable_weights)
                self.optimizer_e.apply_gradients(zip(gradients, self.eval.trainable_weights))
            
                
                #Backprop policy 
                with GradientTape() as tape2:
                    e = self.policy(expand_dims(turned_x,0))
                    loss_value = loss_funktion(e_prime,e)

                # Update the weights of eval to minimize the loss value.
                gradients = tape2.gradient(loss_value, self.policy.trainable_weights)
                self.optimizer_p.apply_gradients(zip(gradients, self.policy.trainable_weights))

                recent_exp.append([x,direction,r,self.board.get_data(),abs(loss_value)])
                if len(recent_exp)>100:
                    recent_exp=recent_exp[1:]
            
                
                
            real_reinforcement_list[i]=real_reinforcement
            lifetime[i]=self.board.time
            food_found[i]=self.board.food_found
            energy[i]=self.board.energy


            if not prio:

                for h in range(self.replay_number(self.rounds_trained)):#replays past experiences
                    k=self.choose_experience(len(recent_exp))
                    self.train_experience(recent_exp[k])


            if prio:

                probs=[(recent_exp[t][-1]+0.001)**0.7 for t in range(len(recent_exp))]
                w_list=[(len(recent_exp)*probs[t]/sum(probs))**(-0.5-0.5*self.rounds_trained/rounds) for t in range(len(recent_exp))]#contains the weights w_i; missing '/max w_i'
                for h in range(self.replay_number(self.rounds_trained)):
                    exp=random.choices(recent_exp,weights=probs)[0]
                    t=recent_exp.index(exp)
                    new_delta=self.train_experience(exp,w_list[t]/max(w_list))
                    recent_exp[t][-1]=new_delta
                    probs[t]=(recent_exp[t][-1]+0.001)**0.7
                    w_list[t]=(len(recent_exp)*probs[t]/sum(probs))**(-0.5-0.5*self.rounds_trained/rounds)



            
            if teaching:

                prob=self.teaching_prob(self.rounds_trained)
                lists=['l1','l2','l3','l4','l5','l6','l7']
                l_name=random.choice(lists)
                with open(f'teaching/{l_name}.pickle', 'rb') as handle:
                    l = pickle.load(handle)
                for exp in l:
                    if prob>=random.random():
                        self.train_experience(exp) 


            self.rounds_trained+=1
            self.board.restart(food_pos=food_pos_list[min(i+1,rounds-1)])

        
            
        return (real_reinforcement_list,lifetime,food_found,energy)
    
    
    def train_experience(self,experience,weight=1):
        '''
        train agent on one experience [old state,direction,reinforcement,new state]
        '''

        x = experience[0]
        direction, turned_x = experience[1],self.board.turn_input_even_more(x,experience[1])

        r = experience[2]

        loss_funktion = self.loss_funktion

        
        
        y = experience[3]
                
        e_prime = np.array([[r]]) + self.discount_factor * self.eval(expand_dims(y,0))[0,0]
                  

        #Backprop policy
        with GradientTape() as tape:
            e = self.policy(expand_dims(turned_x,0))
            loss_value = loss_funktion(e_prime,e)*weight

        # Update the weights of policy to minimize the loss value.
        gradients = tape.gradient(loss_value, self.policy.trainable_weights)
        self.optimizer_p.apply_gradients(zip(gradients, self.policy.trainable_weights))

        return(abs(loss_value))
    
    




