# Learning utilities of FD-Learn

## Learning search policies with the parametrized search routine

Learning search policies according to [Gomoluch et al. 2020](https://ojs.aaai.org//index.php/ICAPS/article/view/6748) is performed with the [`param_search.py`](param_search.py) script. Usage:

`
python param_search.py config.cfg
`

For an example configuration with comments, see [`param_search_example.cfg`](param_search_example.cfg).

### Training problems

The script can be used without problem generation, instead training on user-supplied PDDL problems. This mode is used when the `problem_generator` setting is not specified in the configuration file.

To generate new problems at every training iteration, a problem generator needs to be provided. The `problem_generators` module contains wrappers for problem generators of the IPC domains of <em>Transport</em>, <em>Parking</em>, <em>Elevators</em>, <em>No-mystery</em> and <em>Floortile</em>. The wrappers require the original generators from the [IPC-2014 learning track](https://www.cs.colostate.edu/~ipc2014/). They are assumed to be located in a directory sibling to FD-Learn. At the moment, using a different location requires updating the relative paths in the respective generator modules.

New problem generators can be added by subclassing the `base_generator`. A new generator can be imported in the training script (`param_search.py`) and added to the `generator_dict` dictionary.

### Feature scaling

The neural search policies represent the current state of the planner using a number of high-level features ([Gomoluch et al. 2020](https://ojs.aaai.org//index.php/ICAPS/article/view/6748)). To ensure appropriate scaling of the features, a text file containing space-separated scaling factors for each of the features needs to be provided. By default, it will be the `scales.txt` file in the working directory. Typical expected or reasonable-maximum values for each of the features are both reasonable choices. [Gomoluch et al. 2020](https://ojs.aaai.org//index.php/ICAPS/article/view/6748)) used the maximum values of the features recorded when running the parametrized planner with default parameters on a separate batch of problems following the training distribution.

The feature representation is computed in [`ParametrizedSearch::get_state_features()`](../src/search/search_engines/parametrized_search.cc). The current features, in order, are:
- the initial value of the heuristic,
- the lowest observed value of the heuristic,
- time elapsed (in seconds),
- the number of node expansions since the lowest heuristic value last dropped,
- the number of states generated in the search,
- the number of states evaluated in the search,
- the number of states expanded in the search.
