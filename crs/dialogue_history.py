class DialogueHistory:
    def __init__(self):
        self.history = []  

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_history(self):
        return self.history

    def __str__(self):
        return "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.history])
    
    def color_str(self):
        formatted_history = []
        for idx, entry in enumerate(self.history, start=1):
            if entry['role'] == 'user':
                formatted_history.append(f"\033[34m[Round {idx}] User:\033[0m {entry['content']}")
            elif entry['role'] == 'assistant':
                formatted_history.append(f"\033[32m[Round {idx}] Assistant:\033[0m {entry['content']}")
        return "\n".join(formatted_history)

    def get_last_message(self):
        if self.history:
            return self.history[-1]
        else:
            return None

    def get_last_user_message(self):
        for message in reversed(self.history):
            if message['role'] == 'user':
                return message
        return None

    def get_last_assistant_message(self):
        for message in reversed(self.history):
            if message['role'] == 'assistant':
                return message
        return None

    def clear_history(self):
        self.history = []
