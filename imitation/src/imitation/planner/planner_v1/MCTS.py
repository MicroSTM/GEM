import random
import numpy as np
from anytree import AnyNode as Node
from anytree import RenderTree
import pdb


class MCTS:
    def __init__(
        self,
        agent_id,
        action_space,
        transition,  # attached,
        reward,
        is_terminal,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        seed=-1,
    ):
        self.agent_id = agent_id
        self.transition = transition
        # self.attached = attached
        self.reward = reward
        self.is_terminal = is_terminal
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.action_space = list(action_space)
        self.rollout_policy = lambda state: random.choice(self.action_space)
        self.seed = seed
        # np.random.seed(seed)
        # random.seed(seed)
        self.nb_nodes = 0
        # print('MCTS action_space', action_space)

    def run(self, curr_root, t, expected_action, dT=1):
        random.shuffle(self.action_space)
        curr_root = self.expand(curr_root, t, expected_action)
        for explore_step in range(1, self.num_simulation + 1):
            # if explore_step % 10 == 0 and self.num_simulation > 10:
            #     print("simulation step:", explore_step, "out of", self.num_simulation)
            curr_node = curr_root
            node_path = [curr_node]

            tmp_t = t - 1
            # if explore_step % 10 == 0 and self.num_simulation > 10:
            #     print('curr_node', curr_node.action, curr_node.state[0]['pos'])
            while curr_node.is_expanded:

                next_node = self.select_child(curr_node)
                # if explore_step % 10 == 0 and self.num_simulation > 10:
                #     print('next_node', next_node.action, next_node.state[0]['pos'])

                node_path.append(next_node)

                curr_node = next_node
                tmp_t += 1

            leaf_node = self.expand(curr_node, tmp_t, expected_action)
            value = self.rollout(leaf_node, tmp_t, expected_action)
            # if explore_step % 10 == 0 and self.num_simulation > 10:
            #     print('value', value)
            #     pdb.set_trace()
            self.backup(self.agent_id, value, node_path)

        # next_root = None #self.select_next_root(curr_root)
        # action_taken = list(next_root.id.keys())[0]
        action_taken, children_visit, next_root, selected_child_index = self.select_next_root(curr_root)
        return next_root, action_taken, children_visit, selected_child_index

    def calculate_score(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        action_prior = child.action_prior

        if self_visit_count == 0:
            u_score = np.inf
            q_score = 0
        else:
            exploration_rate = (
                np.log((1 + parent_visit_count + self.c_base) / self.c_base)
                + self.c_init
            )
            u_score = (
                exploration_rate
                * action_prior
                * np.sqrt(parent_visit_count)
                / float(1 + self_visit_count)
            )
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score

        # print(child.action, self.c_base, self.c_init, q_score, u_score, score)
        return score

    def select_child(self, curr_node):
        scores = [
            self.calculate_score(curr_node, child) for child in curr_node.children
        ]
        if len(scores) == 0:
            return None
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child

    def get_action_prior(self):
        action_prior = {
            action: 1 / len(self.action_space) for action in self.action_space
        }
        return action_prior

    def expand(self, leaf_node, t, expected_action):
        curr_state = leaf_node.state
        if not self.is_terminal(self.agent_id, curr_state, t):
            leaf_node.is_expanded = True
            leaf_node = self.initialize_children(leaf_node, expected_action)
        return leaf_node

    def rollout(self, leaf_node, t, expected_action):
        reached_terminal = False
        curr_state = leaf_node.state
        rewards = []
        sum_reward = 0
        for rollout_step in range(self.max_rollout_steps):
            action = self.rollout_policy(curr_state)
            expected_action = self.rollout_policy(curr_state)
            # print(action)            
            rewards.append(
                self.reward(
                    self.agent_id,
                    curr_state,
                    action,
                    t + rollout_step + 1,
                    t + self.max_rollout_steps,
                )
            )
            
            if self.is_terminal(
                self.agent_id, curr_state, t + rollout_step + 1
            ):  # or t + rollout_step + 1 >= self.nb_steps:
                # f = open("rollout_terminal_heuristic_{}_{}.txt".format(dis, self.num_simulations), "a+")
                # print(1, file=f)
                reached_terminal = True
                break

            next_state = self.transition(
                self.agent_id,
                curr_state,
                action,
                expected_action,
                self.c_init,
                self.c_base,
            )
            # pdb.set_trace()
            curr_state = next_state

        if rewards:
            sum_reward = rewards[-1]
            for r in reversed(rewards[:-1]):
                sum_reward = sum_reward * 0.95 + r
        return sum_reward

    def backup(self, agent_id, value, node_list):
        # if value > 0:
        #     print("=====================")
        cur_value = value
        for node in reversed(node_list):
            state = node.state
            action = node.action
            reward = self.reward(agent_id, state, action, 0, 0)
            # print('backup', [s['cpos'] for s in state], action, reward)
            cur_value = cur_value * 0.95 + reward
            node.sum_value += cur_value
            node.num_visited += 1
            # if value > 0:
            #     print(node)

    def select_next_root(self, curr_root):
        children_visit = [child.num_visited for child in curr_root.children]
        children_value = [child.sum_value for child in curr_root.children]
        print('children_visit:', children_visit)
        print('children_value:', children_value)
        print('actions:', [child.action for child in curr_root.children])
        maxIndex = np.argwhere(children_visit == np.max(children_visit)).flatten()
        selected_child_index = random.choice(maxIndex)
        # print('selected_child_index', selected_child_index)
        # print('curr_root.children',curr_root.children[selected_child_index])
        # curr_state = curr_root.state
        action = curr_root.children[selected_child_index].action
        return (
            action,
            children_visit,
            curr_root.children[selected_child_index],
            selected_child_index
        )  # next_root

    def initialize_children(self, node, expected_action):
        state = node.state
        initActionPrior = self.get_action_prior()

        for action in self.action_space:
            nextState = self.transition(
                self.agent_id, state, action, expected_action, self.c_init, self.c_base
            )
            # print("child initialized", state, action, nextState)
            Node(
                parent=node,
                id=self.nb_nodes,
                action=action,
                state=nextState,
                num_visited=0,
                sum_value=0,
                action_prior=initActionPrior[action],
                is_expanded=False,
            )
            self.nb_nodes += 1
        return node
