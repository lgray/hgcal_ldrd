# hgcal_ldrd
Code repository for HGCal LDRD

You will need to:
```
conda create --name hgcal-env python=3.6
source activate hgcal-env
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install pandas matplotlib jupyter nbconvert==5.4.1
conda install -c conda-forge tqdm
pip install uproot scipy sklearn --user
```
