# EFTModelsStudies

Repository containing tools to study SMEFT models and, more specifically, how the variation of Wilson coefficients changes the shape of the Higgs pt.

## ```pt_shape_bokeh.py```

Based on the JSON files produced in the last part of a typical [EFT2Obs](https://github.com/ajgilbert/EFT2Obs) workflow, when running the ```scripts/get_scaling.py``` script. These files are basically dictionaris with the following items:
- edges: list of 2-lists (low and high edge of the bin)
- bins: list containing a list for each bin; each list contains a list in the form [term, term_err, parameter, *other_or_same_parameter] (e.g. ```[3.9, 0.08, 'chw']``` or ```0.35, 0.02, 'cww', 'cb'```)
- parameters: list of Wilson coefficients available (probably useless, since they can be inferred from the previous key)
- areas: list containing the area of each bin in the SM case (this are probably already normalized?)

The assumption is that, in narrow width approximation, the cross section of a Higgs boson produced through gluon fusion and decaying into a certain particle X can be written as:

<img src="https://render.githubusercontent.com/render/math?math=\sigma(gg\rightarrow%20H%20\rightarrow%20X)%20=%20\sigma(gg%20\rightarrow%20H)%20BR(H%20\rightarrow%20X)%20=%20\sigma(gg%20\rightarrow%20H)%20\frac{\Gamma^{H\rightarrow%20X}}{\Gamma^{TOT}}">

In order to provide a SMEFT interpretation of the pt shape, the three parts that appear in the last part of the equation have to be parametrized. It can all be done within EFT2Obs, meaning that for each of these parts we end up with a JSON file with the structure described above (details concerning the procedure followed to produce something meaningful can be found [here](https://gist.github.com/maxgalli/7407c634d7d5fa2ab5043ee0e434ba7c)). 

Given this structure, the script accepts three arguments: ```production```, ```br-num``` and ```br-den```. The first one is the only required one, and all three of them accept a path to a JSON file with the structure explained above.

### Run it

If working from a remote server, do the following:

Remote:
```
bokeh serve pt_shape_bokeh.py --allow-websocket-origin=localhost:8006 --args --production tests/HiggsTemplateCrossSections_pT_V.json
```

Then from the local machine, you have to ssh-forward through the port 8006 with the following command:
```
ssh -Y -N -f -L localhost:8006:localhost:5006 gallim@t3ui01.psi.ch
```
(note that 5006 is the remote port where bokeh connects, it is displayed when the server starts). At this point, the interactive plot can be seen when typing in the browser ```http://localhost:8006/pt_shape_bokeh```.

### Preview

![Alt Text](docs/imgs/eft-2.gif)


## ```plot_shape_matplotlib.py```

Same idea as before, except that here it is not interactive and shapes are plotted. 

Example of command:

```
python plot_shape_matplotlib.py --production tests/HiggsTemplateCrossSections_pT_V.json --observable pTH --output-dir /work/gallim/DifferentialCombination_home/outputs/EFTModelStudies_out
```


<p float="left">
  <img src="docs/imgs/pTH_ca.png" width="300" />
  <img src="docs/imgs/sf_pTH_ca.png" width="300" /> 
</p>


## ```plot_shape_matt_predictions.py```

This is probably the ultimate one to be used. It relies on parametrizations found in a directory with a structure like [this](https://github.com/maxgalli/EFTScalingEquations/tree/differentials_220506/equations/CMS-prelim-SMEFT-topU3l_22_05_05) in the specific repo [EFTScalingEquations](https://github.com/maxgalli/EFTScalingEquations). Run with e.g.:
```
python plot_shape_matt_predictions.py --input-dir /work/gallim/DifferentialCombination_home/EFTScalingEquations/equations/CMS-prelim-SMEFT-topU3l_22_05_05 --output-dir /eos/home-g/gallim/www/plots/DifferentialCombination/CombinationRun2/EFTModelsStudies/test
```

## ```chi_square_fitter.py```

## ```pca.py```

Here we perform a Principal Component Analysis in order to find out along which axes the information is "flat" and discard them. Apart from plotting the rotation matrix and the new SMEFT coefficients for the eigenvectors, the first part attempts to plot a matrix similar to the one shown at page 34 of [this paper](https://arxiv.org/pdf/2105.00006.pdf). 

An example of command is 
```
python pca.py --prediction-dir /work/gallim/DifferentialCombination_home/EFTScalingEquations/equations/CMS-prelim-SMEFT-topU3l_22_05_05 --model-yaml /work/gallim/DifferentialCombination_home/DifferentialCombinationRun2/metadata/SMEFT/220926Atlas.yml --channels hgg hzz --output-dir /eos/home-g/gallim/www/plots/DifferentialCombination/CombinationRun2/EFTModelsStudies/SMEFT/PCA/CMS-prelim-SMEFT-topU3l_22_05_05-220926Atlas
```

where the arguments are hearby described:
- ```--prediction-dir```: path to one of the directories in [EFTScalingEquations](https://github.com/maxgalli/EFTScalingEquations/tree/differentials_220506)
- ```--model-yaml```: path to a yaml file like the ones stored in [the main project repo](https://gitlab.cern.ch/magalli/DifferentialCombinationRun2), which can be found under ```DifferentialCombinationRun2/metadata/SMEFT``` with the conventions described in the documentation there
- ```channels```: list of decay channels
- ```--output-dir```: where the matrices are going to be dumped

Note that a new directory will be created next to the one specified in ```--prediction-dir```.

# Full Preliminar Study of a Model

This whole thing can be used to perform a preliminar study of SMEFT (but also kappa-framework) models that are then fitted in Combine using the full likelihood. In order to do this, we refer to the [Run2 Differential Combination](https://gitlab.cern.ch/magalli/DifferentialCombinationRun2) and in particular the files in ```metadata/SMEFT```. These files define subgroups of POIs and their ranges to be used together with the above mentioned EFTScalingEquations.
