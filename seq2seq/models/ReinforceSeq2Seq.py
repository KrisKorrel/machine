class ReinforceSeq2Seq(nn.Module):

    def __init__(self):
        super(ReinforceSeq2Seq, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = Variable(state)
        probs = self.forward(state)
        cat_dist = Categorical(probs)
        action = cat_dist.sample()
        self.saved_log_probs.append(cat_dist.log_prob(action))
        action = action.data[0]

        return action

    def get_episode_loss(self):
        # Calculate discounted reward of entire episode
        R = 0
        discounted_rewards = []
        for r in self.rewards[::-1]:
            R = r + args.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.Tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
            (discounted_rewards.std() + np.finfo(np.float32).eps)

        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Reset episode
        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss
