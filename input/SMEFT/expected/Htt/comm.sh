combine /work/gallim/DifferentialCombination_home/DifferentialCombinationRun2/CombinedWorkspaces/SM/smH_PTH/Htt.root -M MultiDimFit -m 125 --robustFit=1 --X-rtd MINIMIZER_analytic --algo=singles --cl=0.68 --setParameters r_smH_PTH_0_45=1,r_smH_PTH_45_80=1,r_smH_PTH_80_120=1,r_smH_PTH_120_140=1,r_smH_PTH_140_170=1,r_smH_PTH_170_200=1,r_smH_PTH_200_350=1,r_smH_PTH_350_450=1,r_smH_PTH_GT450=1 --setParameterRanges r_smH_PTH_0_45=-5,5:r_smH_PTH_45_80=-5,5:r_smH_PTH_80_120=-5,5:r_smH_PTH_200_350=-5,5:r_smH_PTH_350_450=-5,5:r_smH_PTH_GT450=-5,5 --redefineSignalPOIs r_smH_PTH_0_45,r_smH_PTH_45_80,r_smH_PTH_80_120,r_smH_PTH_120_140,r_smH_PTH_140_170,r_smH_PTH_170_200,r_smH_PTH_200_350,r_smH_PTH_350_450,r_smH_PTH_GT450 --cminDefaultMinimizerStrategy=0 --floatOtherPOIs=1 --toys -1 --saveFitResult --robustHesse 1
