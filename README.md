# Competition_Football

## Environment

Check details in Jidi Environments

[Football 5vs5](http://www.jidiai.cn/env_detail?envid=71)

[Football 11vs11](http://www.jidiai.cn/env_detail?envid=34)


## Football
<b>Tags: </b>`discrete action space` `discrete observation space`

<b>Environment Introduction: </b>Agents participate football game. In this series games, agents from two sides 
participate the football game and the aim is to score more goals to win the game.

<b>Environment Rules:</b> 
1. This game has two sides(teams). In this game, each side controls 11 players in 11-player teams(11vs11 scenario) or 4 players 
   in 5-player teams except the goal-keeper(5vs5 scenario). Rules are similar to the [official football (soccer) rules](https://www.rulesofsport.com/sports/football.html), 
   including offsides, yellow and red cards. There are, however, small differences.
2. Game consists of two halves, 45 minutes (1500 steps) each, 3000 steps in total. Kick-off at the beginning of each 
   half is done by a different team, but there is no sides swap (game is fully symmetrical).
3. Teams do not switch sides during the game. Left/right sides are assigned randomly.
4. Reward: each team obtains a +1 reward when scoring a goal, and a −1 reward when conceding one to the opposing team.
5. Non-cup scoring rules apply, i.e. the team which scored more goal wins; otherwise it is a draw.
6. There is no walkover applied when the number of players on the team goes below 7. There are no substitute players. 
   There is no extra time applied. Game ends after 3000 steps.

<b>Action space: </b>a list with length n_action_dim，here n_action_dim=1 an each element is a [Discrete](https://github.com/openai/gym/blob/master/gym/spaces/discrete.py) object in Gym, like [Discrete(19)]. 
The action input to the environment is a matrix with size 1*19. Here 1 represents the dimension of action space, 
and 19 represents the value of the action (the action is supposed to be a one-hot vector inside a list, like [action_list], 
action_list=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]). In each turn, the agent is supposed to generate one of 19 actions 
(numbered from 0 to 18) from the [default action set](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#default-action-set) and set the correspounding index of the action_list into 1.

<b>Observation space: </b>observation is a dictionary and contains keys"obs" and "controlled_player_index"。
The correspounding value of "obs" is a [dictionary](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#observations--actions). 
"controlled_player_index" represents the index of the controlled agent. Here, agents in left team are integers 
from 0 to 10 and agents in right team are integers from 11 to 21.

<b>Reward: </b> each team obtains a +1 reward when scoring a goal, and a −1 reward when conceding one to the opposing team.

<b>Environment end conditions: </b> Game ends after 3000 steps.

<b>Evaluation Guide: </b>During verification and evaluation, the platform runs the user code on the single-core CPU 
(GPU is not supported temporarily), and limits the time for the user to return action in each step to no more than 2s 
and memory to no more than 500M. The scores in the leaderboard are calculated and ranked according to the average score 
of the latest 30 games. In the evaluation of competition, all submissions are evaluated using [Swiss round](http://www.jidiai.cn/discussion?disID=18). 

---
## Dependency

>conda create -n football python=3.8.5

>conda activate football

>pip install -r requirements.txt

>install [gfootball environment](https://github.com/google-research/football)

>copy the `env/football_scenarios/malib_5_vs_5.py` file under folder like 
`~/anaconda3/envs/env_name/lib/python3.x/site-packages/gfootball/scenarios`using environment `env_name` or 
`~/anaconda3/lib/python3.x/site-packages/gfootball/scenarios` using base environment.

---

## Run a game

>python run_log.py

---

## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --env "football_11_vs_11_stochastic" --my_ai "random" --opponent "random"

in which you are controlling agent 1 which is green.

## Ready to submit

1. Random policy --> *agents/random/submission.py*
2. RL policy --> *all files in agents/football_5v5_mappo* or *all files in agents/football_11v11_mappo*
