# TITLE: A CPS-task performed by generative agents

#----------------------------
# Setup
#----------------------------

import openai
from openai import OpenAI
import os # Operating System
import numpy as np # For handling of numbers
import pandas as pd # for dataframe and data storage
from datetime import datetime # for adaquete log-storing
from dotenv import load_dotenv
import os
import random

load_dotenv()  # This loads the environment variables



openai_api_key = os.getenv('OPENAI_API_KEY')  # Use environment variable
openai.api_key = openai_api_key # Set the API key
client = OpenAI()
if openai_api_key:
    print("Breakpoint: API Key is set.")
else:
    print("Breakpoint: API Key is not set. Please check your environment variables.")
    try:
    # Make a simple API call to check connectivity
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("API is reachable. Response:", response)
    
    except Exception as e:
        print("Error connecting to the API:", e)

# Constants
TOTAL_AGENTS = 5 # Adjust as necessary
no_simulations = 100 # Adjust as necessary
turn_takings = 10 # Adjust as necessary
turn_takings_count = 0 
temperature = 0.8 # Adjust as necessary
defined_model = "gpt-4o-mini" # Adjust as necessary


simulation_no = 0

#Agents and assistants
assistants = {}
agents = {}
agent_names = ["High_neuro", "High_extra", "High_open", "High_agree", "High_consc"]

# Mapping of traits to the respective agent names
trait_to_agent_mapping = {
    "Neuroticism": "High_neuro",
    "Extraversion": "High_extra",
    "Openness": "High_open",
    "Agreeableness": "High_agree",
    "Conscientiousness": "High_consc"
}

# Randomize the order of traits
randomized_traits = random.sample(list(trait_to_agent_mapping.keys()), len(trait_to_agent_mapping))

#Capturing print statements
conversation_outputs = [] # Initialize a list to store individual outputs
final_dataframe = pd.DataFrame()


# Define the folder name
folder_name = "conversation_history"

# Ensure the folder exists
os.makedirs(folder_name, exist_ok=True)


#Prompts
introduction = f"""You are tasked with planning a 6-hour birthday party that includes dinner and games."
                    You will be planning the party with {TOTAL_AGENTS-1} other people. 
                    The goal is to make it fun and memorable for everyone attending. 
                    The party should include a structured schedule to fit within the 6-hour timeframe, with a smooth flow between activities like dinner and games. 
                    If there’s a theme, it should tie into the decorations, music, and party favors. 
                    Feel free to suggest creative touches to make the event stand out. """

guidelines = f"""
- **Active Listening**: Take turns listening actively to your co-planners, understanding their arguments, and respecting their distinct perspectives.
- **Focus**: Please stay on topic and avoid irrelevant conversations. Your goal is planning a birthday party.
- **Consensus Decision**: You have {turn_takings} combined turns to reach a consensus on the ranking. Ensure that all team members understand and compromise on the final plan.
- **Clear Communication**: While discussing the birthday party, provide reasoning for your choices, especially what you think is important for the guests.
- **Word Introduction**: If you are asked with introducing a word in the description, thdn remember to do so.
"""
# High, Low, and Base Descriptions for Big Five Personality Traits

# Neuroticism
high_neuro = "You often feel anxious, self-doubting, and worried."
base_neuro = "You’re calm but occasionally stressed."

# Extraversion
high_extra = "You’re outgoing, engaging, and thrive in the spotlight."
base_extra = "You’re social but avoid the spotlight."

# Openness to Experience
high_open = "You’re imaginative, curious, and love exploring complex ideas."
base_open = "You balance curiosity with routine."

# Agreeableness
high_agree = "You trust others, collaborate easily, and value fairness."
base_agree = "You cooperate but assert when needed."

# Conscientiousness
high_consc = "You’re meticulous, organized, and meet deadlines reliably."
base_consc = "You’re dependable but sometimes miss details."

# add new word
new_word = """You MUST use "Glowchum"(a glowing party accessory like glowsticks or LED necklaces) in conversation. Ensure as many as possible use it."""

#Personality types 
High_extra = high_extra + base_consc + base_agree + base_neuro + base_open
High_neuro = high_neuro + base_consc + base_agree + base_extra + base_open
High_open = high_open + base_consc + base_agree + base_neuro + base_extra
High_agree = high_agree + base_consc + base_neuro + base_extra + base_open
High_consc = high_consc + base_agree + base_neuro + base_extra + base_open




#------------------------------
# Defining the agent class  
#------------------------------

class Agent():
    def __init__(self, agent_name): #constructor
        self.name = agent_name
        if self.name == "High_extra":
            self.traits = High_extra
        if self.name == "High_neuro":
            self.traits = High_neuro
        if self.name == "High_open":
            self.traits = High_open
        if self.name == "High_agree":
            self.traits = High_agree
        if self.name == "High_consc":
            self.traits = High_consc      

    def get_agent_name(self):
        return self.name
    
    def get_agent_traits(self):
        return self.traits

    def instructions_system_personality(self):
        return f"""
        {introduction}
        Use your assigned personality traits, especially the first sentence herin, and role as a party planner to discuss a detailed 6-hour schedule for the birthday party, prioritizing guest enjoyment and smooth transitions between activities. 
        Focus on balancing a delightful dining experience with engaging games, ensuring the flow is logical and aligned with the group’s preferences. 
        Incorporate a creative or thematic touch if applicable. Review the schedules provided by co-planners independently and incorporate ideas that improve your own schedule. 
        """

    def start_task_system(self):
        #This schedule should be crafted in chronological order, prioritizing smooth transitions between activities and keeping the guests’ enjoyment at the forefront. 
        return f"""
        You work in a company, and today you and your {TOTAL_AGENTS} co-workers  are tasked to engage in a team-building exercise unrelated to your usual work.
        Your team’s goal is to collaboratively design a detailed 6-hour schedule for a birthday party. None of you are professional party planners, but your aim is to create an enjoyable and well-structured event that balances a delightful dining experience with engaging games.
        Discuss and negotiate to persuade others to consider your reasoning behind the timing and placement of each activity.
        Your decisions should focus on creating a cohesive and memorable event while aligning with the group’s preferences and potential creative or thematic touches. 
        Collaboratively agree on a final schedule that satisfies the group, ensuring everyone feels confident in the timeline.
        Your decision should still focus on providing the guests with an enjoyable event.
        """

    def interactive_system_personality(self):
        if new_word in self.get_agent_traits():
            intro_word = new_word
            return f"""
            Continue the collaborative party planning task discussion based on the previous context. 
            Be aware that you have a maximum of {turn_takings} turns combined to finalize the schedule.
            The task introduction remains: {introduction}. It is of utmost importance that you introduce {intro_word} into the conversation without explicitly adding it to the schedule.
            These guidelines for collaboration remain: {guidelines}.
            Use your personality profile, especially the first sentence herein, to guide your behavior, communication style, and approach to the task. ONLY assume the identity of {self.get_agent_name()}
            # Output format: When you all agree on a finalized 6-hour schedule, please state "This is our final schedule:" followed by the activities listed with time slots in chronological order, and end with the statement ‘schedule_complete.’ An example of the format is: This is our final schedule: 10:00 am - Guest arrival and welcome drinks 10:30 am - Icebreaker activity 11:00 am - Outdoor team game 12:00 pm - Lunch is served 1:00 pm - Group trivia game 2:00 pm - Cake cutting and party favors distributed 2:30 pm - Closing remarks and farewells schedule_complete "
            """
        else:
            intro_word = None
            return f"""
            Continue the collaborative party planning task discussion based on the previous context. 
            Be aware that you have a maximum of {turn_takings} turns combined to finalize the schedule.
            The task introduction remains: {introduction}.
            These guidelines for collaboration remain: {guidelines}.
            Use your personality profile, especially the first sentence herein, to guide your behavior, communication style, and approach to the task. ONLY assume the identity of {self.get_agent_name()}
            # Output format: When you all agree on a finalized 6-hour schedule, please state "This is our final schedule:" followed by the activities listed with time slots in chronological order, and end with the statement ‘schedule_complete.’ An example of the format is: This is our final schedule: 10:00 am - Guest arrival and welcome drinks 10:30 am - Icebreaker activity 11:00 am - Outdoor team game 12:00 pm - Lunch is served 1:00 pm - Group trivia game 2:00 pm - Cake cutting and party favors distributed 2:30 pm - Closing remarks and farewells schedule_complete "
            """
print("Breakpoint: class Agent setup done")
# --------------------------------------------------------------
# Create assistant class
# --------------------------------------------------------------

class Assistant():
    def __init__(self, agent_name, agent_traits): #constructor
        self.assistant = client.beta.assistants.create(
            name=agent_name,
            temperature= temperature,
            model= defined_model,
            description= f"""Your name is {agent_name}, and you have the following personality profile: {agent_traits}.
            DO NOT assume any other identities other than {agent_name}."""
        )  
        self.thread = self.create_thread()
    
    def get_assistant_id(self):
        self.assistant = openai.beta.assistants.retrieve(self.assistant.id) # Retrieve the Assistant
        return self.assistant.id
    
    def create_thread(self):
        self.thread = client.beta.threads.create()
        return self.thread
    
    def get_thread_id(self):
        self.thread = openai.beta.threads.retrieve(self.thread.id)
        return self.thread.id
    
    def message(self, message_content):
        client.beta.threads.messages.create( #create input message
            thread_id=self.thread.id,
            role="user", # Message is not from the assistant
            content= message_content
        ) 
    def run_assistant(self, instructions):
        run = client.beta.threads.runs.create_and_poll( #create run
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions= instructions
        )

        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc"
            )
            self.message_text = messages.data[0].content[0].__getattribute__('text').__getattribute__('value') 
            self.message_text= str( self.message_text)
            return self.message_text 
        else:
            print(f"{self.assistant.name} is creating output message. Status: {run.status}")

    def delete_assistant(self):
        client.beta.assistants.delete(self.assistant.id)

    def delete_thread(self):
        client.beta.threads.delete(self.thread.id)

print("Breakpoint: class Assistant setup done")


#------------------------------ 
# Defining the world class
#------------------------------
class World():
    def __init__(self): #constructor
        self.group_agents = []
        self.group_assistants = []  # Create a dictionary to store assistants

    def create_group(self):
        # Creating agents and assistants in a randomized trait order
        for trait in randomized_traits:
            agent_name = trait_to_agent_mapping[trait]
            agent_name_in_randomized_list = next(name for name in agent_names if name == agent_name)
            # Create agent
            agent = Agent(agent_name_in_randomized_list)
            self.group_agents.append(agent)
            
            # Create assistant
            assistant = Assistant(agent.get_agent_name(), agent.get_agent_traits())
            self.group_assistants.append(assistant)

        # Print the order of created agents
        print(f"Breakpoint: Group created with the following agents in order: {', '.join([agent.name for agent in self.group_agents])}")

    def run_once(self):        
        print("Starting task: initial_prompt")

        for i in range(TOTAL_AGENTS): #second part
            assistant = self.group_assistants[i]
            agent = self.group_agents[i]
            assistant.message(agent.start_task_system())
        print("Breakpoint: done with initial_prompt")
        
        output = None

        print("Starting second part: Group task")
        conversation_outputs.append("Starting Group task:")

        i=0
        while i<turn_takings: #third part
            global turn_takings_count
            turn_takings_count = i+1
            print(f"Starting interactive_task no. {i+1}")
            number = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4] #2 rounds
            assistant = self.group_assistants[number[i]]
            assistant_other1 = self.group_assistants[number[i+1]]
            assistant_other2 = self.group_assistants[number[i+2]]
            assistant_other3 = self.group_assistants[number[i+3]]
            assistant_other4 = self.group_assistants[number[i+4]]
            
            agent = self.group_agents[number[i]]
            output = assistant.run_assistant(agent.interactive_system_personality())            
            final_output=(f"Agent {agent.name} has responded: {output}")
            conversation_outputs.append(f"(Turn {i+1}) {final_output}")
            # if output is not None and isinstance(output, str):
            #     if "schedule_complete" in output:
            #         start = output.find("This is our final schedule:")
            #         end = output.find("schedule_complete.")
            #         conversation_outputs.append(f"Note: Task finished as concensus was reached within {i+1} turntakings.")
            #         print(f"Note: Task finished as concensus was reached within {i+1} turntakings.")
            #         break            
            assistant_other1.message(final_output)
            assistant_other2.message(final_output)
            assistant_other3.message(final_output)
            assistant_other4.message(final_output)
            i+=1
        if i == turn_takings:
            if output is not None and isinstance(output, str):
                if "schedule_complete" in output:
                    start = output.find("This is our final schedule:")
                    end = output.find("schedule_complete")
            conversation_outputs.append(f"Note: Task finished")
            print(f"Note: Task finished as concensus was not reached within {turn_takings} turntakings.")
        print("Breakpoint: done with interactive_task")

    def delete_assistants(self):
        for i in range(TOTAL_AGENTS):
            assistant = self.group_assistants[i]
            assistant.delete_thread()
            assistant.delete_assistant()
        print("Breakpoint: assistants deleted")

    def save_outputs_conversation(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"run_{current_time}.csv") # Creating file

        #csv_path_outputs = root+ '\\' +conversation_history+ '\\'  +filename
        df_conversation = pd.DataFrame(conversation_outputs)  # Create DataFrame
        df_conversation.to_csv(filename, index=False)  # Save DataFrame to CSV

    def save_outputs_final(self, agent_turns):    
        global final_dataframe
        global simulation_no
        global turn_takings_count
        global conversation_outputs
        list ={
            "Agent_list": agent_turns,
            "Introducing_agent": agent_turns[1],
            "Receiving_agent": agent_turns[2],
            "Conversation": conversation_outputs,
            "turn_takings": turn_takings_count
            }
        new_df = pd.DataFrame([list], columns=[
            "Agent_list",
            "Introducing_agent",
            "Receiving_agent",
            "Conversation",
            "turn_takings"
        ])
        final_dataframe = pd.concat([final_dataframe, new_df], ignore_index=True)

#------------------------------ 
# Running simulations
#------------------------------

for i in range(no_simulations):
    global sim_no
    sim_no = i + 1
    conversation_outputs = []  # resetting output documents
    model = World()

    # Provided code begins here
    agent_names = ["High_neuro", "High_extra", "High_open", "High_agree", "High_consc"]

    # Mapping of traits to the respective agent names
    trait_to_agent_mapping = {
        "Neuroticism": "High_neuro",
        "Extraversion": "High_extra",
        "Openness": "High_open",
        "Agreeableness": "High_agree",
        "Conscientiousness": "High_consc"
    }

    # Randomize the order of traits
    randomized_traits = random.sample(list(trait_to_agent_mapping.keys()), len(trait_to_agent_mapping))

    # Find the second trait in the randomized order
    second_trait = randomized_traits[1]
    # Get the corresponding agent name
    second_agent_name = trait_to_agent_mapping[second_trait]

    # Create a list of agent names based on the randomized traits
    agent_names_in_randomized_order = [trait_to_agent_mapping[trait] for trait in randomized_traits]

    # Creating turn_takings_list
    agent_turns = agent_names_in_randomized_order * 2

    # Initialize or reset agent variables before modification
    High_neuro, High_extra, High_open, High_agree, High_consc = "", "", "", "", ""

    High_extra = high_extra + base_consc + base_agree + base_neuro + base_open
    High_neuro = high_neuro + base_consc + base_agree + base_extra + base_open
    High_open = high_open + base_consc + base_agree + base_neuro + base_extra
    High_agree = high_agree + base_consc + base_neuro + base_extra + base_open
    High_consc = high_consc + base_agree + base_neuro + base_extra + base_open


    # Dynamically modify the second agent's formula
    if second_agent_name == "High_extra":
        High_extra += new_word
        print("printing high_extra:")
        print(High_extra)
    elif second_agent_name == "High_neuro":
        High_neuro += new_word
        print("printing High_neuro:")
        print(High_neuro)
    elif second_agent_name == "High_open":
        High_open += new_word
        print("printing High_open:")
        print(High_open)
    elif second_agent_name == "High_agree":
        High_agree += new_word
        print("printing High_agree:")
        print(High_agree)
    elif second_agent_name == "High_consc":
        High_consc += new_word
        print("printing High_consc:")
        print(High_consc)

    print(f"The second trait is {second_trait}, corresponding to {second_agent_name}, and it has been updated.")
    # Provided code ends here

    model.create_group()
    model.run_once()
    model.save_outputs_final(agent_turns=agent_turns)
    model.delete_assistants()
    print(f"Model run {i + 1} of {no_simulations} complete.")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the full file path
    filename = os.path.join(folder_name, f"run_{current_time}.csv")
    # Save the DataFrame to CSV
    final_dataframe.to_csv(filename, index=False)

    print(f"File saved to: {filename}") 
