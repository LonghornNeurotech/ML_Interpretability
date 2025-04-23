import numpy as np

class SmoothProstheticController:
    # filter stable for alpha on (0, 2), but alpha < 1 preferred to ensure monotonic, pos, dec impulse response
    # greater alpha means more weight to current prediction, while lower alpha weights feedback/past outputs more heavily
    def __init__(self, alpha=0.3, u_threshold=0.9, l_threshold=0.1):
        self.output = (u_threshold + l_threshold) / 2
        self.alpha = alpha
        self.upper = u_threshold
        self.lower = l_threshold
        self.state = 0  # Initial state, 0 = closed, 1 = open

    def update(self, input_prob):
        # IIR filter that exponentially decays past prediction influence
        self.output = self.alpha * input_prob + (1 - self.alpha) * self.output

        # Hysteresis threshold
        if self.state == 0 and self.output > self.upper:
            self.state = 1  # Open
        elif self.state == 1 and self.output < self.lower:
            self.state = 0  # Close

        return self.state

    def reset(self):
        self.output = 0.5
        self.state = 0


class MultiClassSmoothController:
    # filter stable for alpha on (0, 2), but alpha < 1 preferred to ensure monotonic, pos, dec impulse response
    # greater alpha means more weight to current prediction, while lower alpha weights feedback/past outputs more heavily
    def __init__(self, n_classes=2, alpha=0.3, threshold=0.9):
        self.n = n_classes
        self.output = np.full(self.n, 1 / self.n)
        self.alpha = alpha
        self.threshold = threshold
        self.state = 0

    def update(self, prob_vector):
        # IIR filter that exponentially decays past prediction influence
        self.output = self.alpha * prob_vector + (1 - self.alpha) * self.output

        # Hysteresis threshold
        if np.max(self.output) > self.threshold:
          self.state = np.argmax(self.output)

        return self.state

    def reset(self):
        self.output = np.full(self.n, 1 / self.n)
        self.state = 0

