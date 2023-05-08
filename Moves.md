```python

        case 0:  # Move down
            new_pos = (self.agent_pos[i][0],
                        min(max_y, self.agent_pos[i][1] + 1))
        case 1:  # Move up
            new_pos = (self.agent_pos[i][0],
                        max(0, self.agent_pos[i][1] - 1))
        case 2:  # Move left
            new_pos = (max(0, self.agent_pos[i][0] - 1),
                        self.agent_pos[i][1])
        case 3:  # Move right
            new_pos = (min(max_x, self.agent_pos[i][0] + 1),
                        self.agent_pos[i][1])
        case 4:  # Stand still
            new_pos = (self.agent_pos[i][0],
                        self.agent_pos[i][1])
        case _:
            raise ValueError(f"Provided action {action} for agent {i} "
                 f"is not one of the possible actions.")
```
