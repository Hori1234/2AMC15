import subprocess


sigmas = [1] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gammas = [0.6, 0.9]
epsilons = [0.1, 0.5, 0.8]
episodes = [10, 30, 60, 100] #, 200, 300, 400, 500, 800, 1000, 2000]
iters = [500000, 1000000]
nconvergences = [10000]
replace_agent_after_episodes = [False] #[False, True]
replace_to_starts = [False] #[False, True]
n_runs_per_combi = 1

n_combis = len(sigmas)*len(gammas)*len(epsilons)*len(episodes)*len(iters)*len(nconvergences)*len(replace_agent_after_episodes)*len(replace_to_starts)*n_runs_per_combi
count = 1

for sigma in sigmas:
    for gamma in gammas:
        for epsilon in epsilons:
            for episode in episodes:
                for iter in iters:
                    for nconvergence in nconvergences:
                        for replace_agent_after_episode in replace_agent_after_episodes:
                            for replace_to_start in replace_to_starts:
                                for i in range(n_runs_per_combi):


                                    print(f"Run {count}/{n_combis} -- {round(count/n_combis,2)}%")

                                    output_path = f"sigma-{sigma}-gamma-{gamma}-epsilon-{epsilon}-episode-{episode}-nconvergence-{nconvergence}-iter-{iter}-replace-{replace_agent_after_episode}-replace_start-{replace_to_start}"
                                    print(output_path)
                                    command = f"python train.py grid_configs/final_test_single.grd results/mc_tests_final --no_gui --iter 100000 --fname {output_path} --sigma {sigma} --gamma {gamma} --epsilon {epsilon} --episode {episode} --nconvergence {nconvergence} --replace_agent_after_episode {replace_agent_after_episode} --replace_to_start {replace_to_start}"

                                    # Run the command in the command prompt or terminal
                                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                                    
                                    # Print the error if there is any
                                    print(result.stderr)
                                    # Print the output
                                    print(result.stdout)

                                    count += 1