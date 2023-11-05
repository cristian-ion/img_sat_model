class EarlyStopper:
    def __init__(self, patience=3, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def state_dict(self) -> dict:
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'counter': self.counter,
            'min_validation_loss': self.min_validation_loss,
        }

    def load_state_dict(self, d: dict):
        self.patience = d['patience']
        self.min_delta = d['min_delta']
        self.counter = d['counter']
        self.min_validation_loss = d['min_validation_loss']
