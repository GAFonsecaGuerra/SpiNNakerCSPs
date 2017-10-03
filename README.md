# SpiNNakerCSPs

**A Spiking Neural Network Solver for Constraint Satisfaction Problems.**

This repository will help you create a spiking neural network whose structure represents a constraint satisfaction problem and whose dynamics implements a stochastic search of its solution.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
In order to use this repository you should have a working installation of the PyNN implementation for SpiNNaker,
you can follow the [user installation instructions](http://spinnakermanchester.github.io/) in:

```
http://spinnakermanchester.github.io/
```
Note: we currently use [sPyNNaker7](https://github.com/SpiNNakerManchester/sPyNNaker7),
but will eventually upgrade to be compatible with [sPyNNaker8](https://github.com/SpiNNakerManchester/sPyNNaker8).

you will also require:

* numpy
* matplotlib
* simplejson

### Installing
1. Just clone the project:

        git clone https://github.com/GAFonsecaGuerra/SpiNNakerCSPs.git
        cd SpiNNakerCSPs
        
2. Run some of the examples e.g. escargot.py file to test its working:

        python escargot.py
### Usage

1. Edit the CSP_template.py file to implement your own SNN solver for a CSP of your interest. It simply creates an instance of the CSP class available in the snn_creator.py file and implements the methods therein. An initial analysis is possible with the tools in analysis.py.

2. You can use different strategies of noise implementation to improve the stochastic search.

## Files

* snn_creator.py
* analysis.py

* spin2csp.py
* sudoku2csp.py     
* tsp2csp.py
* world_border_countries.py

* australia_cmp.py
* world_cmp.py
* sudoku_easy.py
* sudoku_hard.py
* escargot.py
* spin_system.py 

* README.md
* CODE_OF_CONDUCT.md
* __init__.py
* spynnaker.cfg

## Authors

* **Gabriel Fonseca**  - [GAFonsecaGuerra](https://github.com/GAFonsecaGuerra)
* **Steve Furber** -  [sfurber](https://github.com/sfurber)

See also the list of [contributors](https://github.com/GAFonsecaGuerra/SpiNNakerCSPs/graphs/contributors) who participated in this project.


## Contributing

If you want to contribute to the development of this repository please read [CODE_OF_CONDUCT.md]( https://github.com/GAFonsecaGuerra/SpiNNakerCSPs/blob/master/CODE_OF_CONDUCT.md) for details on our code of conduct, and [CONTRIBUTING.md]() to know the process for submitting pull requests to us.
