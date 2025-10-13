### Getting set up with this project:
1. Clone the git repository using 
`git clone https://github.com/luckyowl20/PSS_CUAS.git`

2. Set up a virtual environment called `.pss` using this command 
`python -m venv .pss`

3. Activate your virtual environment (windows command)
`.\.pss\Scripts\activate`

4. Install requirements from `requirements.txt`, this may take a minute.
`pip install -r requirements.txt`

5. Run `pip install -e .` to install local packages used for analysis and other in house packages. 

6. Run code!


### Running the launcher orientation kinematics script
1. Make sure python is installed and the above setup is complete

2. Navigate to the `src/kinematics` directory from the command line

3. Type `python kinematics_sim.py` or `python3 kinematics_sim.py` to run the sim.