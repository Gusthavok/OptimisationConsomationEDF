

class Agent:
    name = ""
    results_dir = "" ## path where the agent writes its load profiles (in KW) and COST  after executing its function,
                ## as load_date_yyyymmdd_hhmm.csv and cost_yyyymmdd_hhmm.csv
    prices_path = "" ## path where the agent READS the prices sent by the central aggregator
    exe_function = '' ## string defining command to launch the optim function of the agent

    def __init__(self, name: str, input_path: str, output_path: str, working_dir: str):
        self.name = name
        self.input_path = input_path
        self.output_path = output_path

        self.working_dir = working_dir
        # self.consumer_directory = consumer_directory

