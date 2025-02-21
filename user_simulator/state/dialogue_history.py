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

    def get_swapped_history(self):
        """
        Return a new list with 'user' and 'system' roles swapped in each message.
        """
        swapped_history = []
        for entry in self.history:
            swapped_entry = entry.copy()  # 创建每条消息的副本
            if swapped_entry['role'] == 'user':
                swapped_entry['role'] = 'assistant'
            elif swapped_entry['role'] == 'assistant':
                swapped_entry['role'] = 'user'
            swapped_history.append(swapped_entry)
        return swapped_history

