# CRS Variable Overview

In the CRS internal inference process, three key variables are maintained during each LLM inference session:

- **scratchpad** – Keeps track of the internal inference state.  
- **reply_count** – Represents the number of inference steps taken.  
- **inter_log** – Logs all input-output pairs for fine-tuning purposes.  

## Dialogue History
To manage conversation flow, CRS maintains:  

- **dialogue_history** – Stores the conversation history for each response.  
- **step_n** – Tracks the number of dialogue turns.  

This structured approach ensures consistent and traceable inference within CRS.
