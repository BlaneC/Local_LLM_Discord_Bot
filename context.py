class ChatContext:
    def __init__(self, initial_context, encoder, max_length):
        self.context = [initial_context]
        self.initial_context = initial_context
        self.encoder = encoder
        self.max_length = max_length
        self.len = 0

    def add_user_input(self, user_input):
        self.context.append(" USER: " + user_input)
        self.context.append(" ASSISTANT:")
        self.len += 2
        self.prune_context()

    def add_assistant_output(self, output):
        self.context[self.len] += output

    def prune_context(self):
        while len(self.encoder.encode("".join(self.context))) > self.max_length:
            self.context.pop(1)  # Pop the oldest request (not including model pre-amble)
            self.len -= 1

    def get_context_str(self):
        return "".join(self.context)
    
    def reset_context(self):
        self.context = [self.initial_context]
        self.len = 0

    
    def set_initial_context(self, new_context):
        self.context = [new_context]                # Reset context and set it to just the new initial context
        self.initial_context = new_context          # Set initial context variable
        self.len = 0

