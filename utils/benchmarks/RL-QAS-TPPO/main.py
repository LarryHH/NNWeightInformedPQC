"""
python -m utils.benchmarks.RL-QAS-TPPO.VQC.main_TPPO
"""

# import argparse
# import subprocess
# import copy
# from argparse import Namespace 


# agent_choices = ["A2C", "A3C", "DDQN", "DQN", "DQN_PER", "DQN_rank", "Dueling_DQN" , "PPO", "TPPO" ]
# task_choices = ['VQE', 'VQSD', 'VQC', 'State_Prep']


# def run_main(args):
#     # Now you can use dot notation (args.task) instead of args['task']
#     print(f"→ Running Task: {args.task} | Agent: {args.agent} | Config: {args.config} | Seed: {args.seed}")



#     subprocess.run([
#         "python", f"{args.task}/main_{args.agent}.py",
#         "--seed", f"{args.seed}",
#         "--config", f"{args.task}/configuration_files/{args.agent}/{args.config}",
#         "--experiment_name", f"{args.agent}/"
#     ])

# if __name__ == "__main__":
#     args_dict = {
#         "task": "VQC",
#         "agent": "TPPO",
#         "seed": '0',
#         "config": 'iris_coblya_2q_VQC.cfg'
#     }

#     # 2. Convert the dictionary into a Namespace object
#     args = Namespace(**args_dict)

#     print("\n### Running Bench-RLQAS with the following setup: ###")
#     print("> Task:", args.task)
#     print("> Agent:", args.agent)
#     print("> Seed:", args.seed)
#     print("> Config:", args.config)
#     print()

#     run_main(args)