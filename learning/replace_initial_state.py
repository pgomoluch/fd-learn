import sys

problem_file = open(sys.argv[1])
problem = problem_file.read()
problem_file.close()

state_file = open(sys.argv[2])
state = state_file.read().rstrip()
state_file.close()

# Extract the new state from the random_walker output
begin = state.find('begin_state')
end = state.find('end_state') + len('end_state')
state = state[begin:end]

begin = problem.find('begin_state')
end = problem.find('end_state') + len('end_state')

new_problem = problem[:begin] + state + problem[end:]

problem_file = open(sys.argv[1], 'w')
problem_file.write(new_problem)
problem_file.close()
