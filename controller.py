import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from config.params import get_mappo_params, get_maddpg_params
from rollout import Rollout
import config.global_params as gp

class Controller:
    def __init__(self, gen_params):
        self.device_num = gen_params.device_num
        
        # algorithm params
        alg_params = None
        if (not gen_params.evaluate and gen_params.train_mode == "mappo") or \
           (gen_params.evaluate and gen_params.eval_mode) == "mappo":
            alg_params = get_mappo_params()
        if (not gen_params.evaluate and gen_params.train_mode == "maddpg") or \
           (gen_params.evaluate and gen_params.eval_mode) == "maddpg":
            alg_params = get_maddpg_params()
        
        print("The training mode is in controller: ", gen_params.train_mode)
        print("The evaluation mode is in controller: ", gen_params.eval_mode)
        
        # project storage path
        root_path = gp.settings.exp_result_dir
        print(f"The project dir is {root_path}")

        # seed 
        if not gen_params.evaluate and gen_params.load_weights:
            self.weights_dir = root_path + gen_params.weights_dir
            gp.settings.weight_dir = self.weights_dir
            train_info_path  = self.weights_dir + f"train_info_{gen_params.resume_episode}.pkl"
            with open(train_info_path, "rb") as f:
                train_info = pickle.load(f)
            self.seed = train_info["train_seed"]
            run_dir = train_info["run_dir"]
            resume_episode = gen_params.resume_episode
            assert gen_params.resume_episode == train_info["resume_episode"], \
                "The resume episode in alg_params does not match that in train_info!"
        else:
            # fix random seed
            self.seed = gen_params.evaluate and gen_params.eval_seed or gen_params.train_seed
            # runtime storage path
            import datetime
            run_path =  (
                    (gen_params.evaluate and "evaluate" or "train")
                    + "/"
                    + (gen_params.evaluate and gen_params.eval_mode or gen_params.train_mode)
                    + "_s_"
                    + str(self.seed)
                    + "_t_"
                    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                    + "_d_"
                    + gen_params.run_desc
            )
            run_dir = run_path + "/"
            gp.settings.weight_dir = root_path + gen_params.weights_dir + run_dir
            resume_episode = 0
        
        gp.settings.run_dir = run_dir
        print(f"The runtime file dir is {run_dir}")
        print(f"The weight file dir is {gp.settings.weight_dir}")
        plot_dir = root_path + alg_params.plot_dir + run_dir
        gp.settings.plot_dir = plot_dir
        print(f"The plot file dir is {plot_dir}")
        gp.settings.seed = self.seed

        # rollout
        self.rollout = Rollout(gen_params, alg_params)
        self.rollout.resume_episode = resume_episode + 1
        # training
        if not gen_params.evaluate:
            self.train_episodes = alg_params.train_episodes
            root_path = gp.settings.exp_result_dir
            run_dir = gp.settings.run_dir
            self.results_dir = root_path + alg_params.results_dir + run_dir
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            self.joint_rewards_col = []
            self.device_rewards_col = []
            self.joint_cost_col = []
            self.device_costs_col = []
            self.edge_comp_ql_col = []
            self.device_comp_qls_col = []
            self.device_comp_dlys_col = []
            self.device_csum_engys_col = []
            self.device_comp_expns_col = []
            self.device_overtime_nums_col = []
        # evaluation
        else:
            self.eval_episodes = gen_params.eval_episodes
    
    def train(self):
        
        start_epi = self.rollout.resume_episode
        for e_id in range(start_epi, self.train_episodes + 1):
            print("------------------train episode: " + str(e_id) + "------------------")
            
            visualize = False
            # if e_id == 1 or e_id == 2: #FOR DEBUG
            #     visualize = True
            if e_id % 400 == 0:
                visualize = True

            joint_rewards, device_rewards, \
            joint_cost, device_costs, \
            edge_comp_ql, device_comp_qls, \
            device_comp_dlys, device_csum_engys, \
            device_esum_engys, device_overtime_nums = self.rollout.run(e_id, visualize=visualize)
            
            # collection
            self.joint_rewards_col.append(joint_rewards)
            self.device_rewards_col.append(device_rewards)
            self.joint_cost_col.append(joint_cost)
            self.device_costs_col.append(device_costs)
            self.edge_comp_ql_col.append(edge_comp_ql)
            self.device_comp_qls_col.append(device_comp_qls)
            self.device_comp_dlys_col.append(device_comp_dlys)
            self.device_csum_engys_col.append(device_csum_engys)
            self.device_comp_expns_col.append(device_esum_engys)
            self.device_overtime_nums_col.append(device_overtime_nums)
            
            if e_id % 1000 == 0:
                with open(self.results_dir + "joint_rewards_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.joint_rewards_col, f)
                with open(self.results_dir + "device_rewards_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_rewards_col, f)
                with open(self.results_dir + "joint_costs_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.joint_cost_col, f)
                with open(self.results_dir + "device_costs_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_costs_col, f)
                with open(self.results_dir + "edge_comp_qls_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.edge_comp_ql_col, f)
                with open(self.results_dir + "device_comp_qls_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_qls_col, f)
                with open(self.results_dir + "device_comp_dlys_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_dlys_col, f)
                with open(self.results_dir + "device_csum_engys_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_csum_engys_col, f)
                with open(self.results_dir + "device_comp_expns_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_expns_col, f)
                with open(self.results_dir + "device_overtime_nums_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_overtime_nums_col, f)
        self.rollout.writer.close()

    def evaluate(self):
        joint_rewards = np.zeros([self.device_type_num], dtype = np.float32)
        device_rewards = np.zeros([self.device_num], dtype = np.float32)
        joint_cost = 0
        device_costs = np.zeros([self.device_num], dtype = np.float32)
        edge_comp_ql = 0
        device_comp_qls = np.zeros([self.device_num], dtype = np.float32)
        device_comp_dlys = np.zeros([self.device_num], dtype = np.float32)
        device_csum_engys = np.zeros([self.device_num], dtype = np.float32)
        device_esum_engys = np.zeros([self.device_num], dtype = np.float32)
        device_overtime_nums = np.zeros([self.device_num], dtype = np.float32)
        
        for e_id in range(1, self.eval_episodes + 1):
            print("------------------evaluate episode: " + str(e_id) + "------------------")
            joint_rewards_, device_rewards_, \
            joint_cost_, device_costs_, \
            edge_comp_ql_, device_comp_qls_, \
            device_comp_dlys_, device_csum_engys_, \
            device_comp_expns_, device_overtime_nums_ = self.rollout.run(e_id)
            
            joint_rewards += joint_rewards_
            device_rewards += device_rewards_
            joint_cost += joint_cost_
            device_costs += device_costs_
            edge_comp_ql += edge_comp_ql_
            device_comp_qls += device_comp_qls_
            device_comp_dlys += device_comp_dlys_
            device_csum_engys += device_csum_engys_
            device_esum_engys += device_comp_expns_
            device_overtime_nums += device_overtime_nums_
        self.rollout.writer.close()

        # averages
        joint_rewards /= self.eval_episodes
        device_rewards /= self.eval_episodes
        joint_cost /= self.eval_episodes
        device_costs /= self.eval_episodes
        edge_comp_ql /= self.eval_episodes
        device_comp_qls /= self.eval_episodes
        device_comp_dlys /= self.eval_episodes
        device_csum_engys /= self.eval_episodes
        device_esum_engys /= self.eval_episodes
        device_overtime_nums /= self.eval_episodes
        
        return joint_rewards, device_rewards, \
               joint_cost, device_costs, \
               edge_comp_ql, device_comp_qls, \
               device_comp_dlys, device_csum_engys, \
               device_esum_engys, device_overtime_nums