# tum-adlr-ss23-06
## Repository for ADLR for Matthias Kreiner and Samuel Zeitler
<p> Clone via  </p>

    git clone --recurse-submodules

<p> to includes SB3 as a submodule in the PEARL folder!  </p>


### cont_env
<p> Code for the continuous environment with the naive SAC from SB3 </p>
<p> For creation of a virtual environment and installation of SB3 and requirements, run: </p>

    source ./install.sh

<p> To activate the environment run: </p>

    source ./init.sh

<p> Configuration of the training can be done in <strong>train_sac.py</strong> </p>

<p> To train models without CLI output run: </p>

    source ./train_sac.sh

<p> To train models with CLI output run: </p>

    source ./train_sac_wo.sh

<p> To test models run:  </p>

    python3 test_trained.py

### PEARL
<p> Code for the continuous environment including the source code for implementation of the PEARL algorithm (Submodule/ SB3 fork)</p>
<p> For creation of a virtual environment and installation of SB3 and requirements, run: </p>

    source ./install.sh

<p> To activate the environment run: </p>

    source ./init.sh

<p> Configuration of the training can be done in <strong>train_pearl_gce/local.py</strong> </p>

<p> To train models without CLI output run: </p>

    source ./train_pearl.sh

<p> To train models with CLI output run: </p>

    source ./wo_train_pearl.sh

<p> To test models run:  </p>

    python3 pearl_test_trained.py
